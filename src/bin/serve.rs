//! OpenAI-compatible inference server. See `serve --help`.

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use clap::Parser;
use eyre::{Result, WrapErr};
use fastokens::DecodeStream;
use gabriel_laevis_0::Compute;
use gabriel_laevis_0::chat::{Message, Role, SpecialToken, render};
use gabriel_laevis_0::config::{ModelConfig, artifact_dir};
use gabriel_laevis_0::model::GabrielLaevis;
use gabriel_laevis_0::model::lm::Sampling;
use gabriel_laevis_0::utils::load_tokenizer;
use kanal::AsyncReceiver;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

type Device = burn::tensor::Device<Compute>;

const DEFAULT_MAX_NEW: usize = 256;
const DEFAULT_TEMPERATURE: f64 = 0.8;
const CHANNEL_CAP: usize = 64;
const NAME: &str = "gabriel-laevis";

/// OpenAI-compatible inference server.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Checkpoint name in the model directory.
    #[arg(default_value = "model")]
    checkpoint: String,
    /// Port to bind.
    #[arg(default_value_t = 8080)]
    port: u16,
}

struct AppState {
    model: GabrielLaevis<Compute>,
    tokenizer: Arc<fastokens::Tokenizer>,
    device: Device,
    stop_token: Option<i64>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// OpenAI Chat Completions `reasoning_effort`. Accepted verbatim so any standard client works.
/// This model is trained on a binary thinking toggle, so effort collapses to on/off internally:
/// `none` is off, any real effort is on. Graded effort can be honored later with no API change.
#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl ReasoningEffort {
    /// Whether this effort enables the thinking channel.
    fn enable_thinking(self) -> bool {
        !matches!(self, ReasoningEffort::None)
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Vec<String>,
    #[serde(default)]
    reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let Cli { checkpoint, port } = Cli::parse();

    let device = Device::default();

    let artifact = artifact_dir();

    let tokenizer = Arc::new(load_tokenizer().wrap_err("loading tokenizer")?);
    let stop_token = Some(gabriel_laevis_0::chat::TURN_END_ID);

    let cfg = ModelConfig::load(artifact.join("config.json")).wrap_err("loading config")?;
    cfg.validate()?;

    let model = GabrielLaevis::<Compute>::new(&cfg, &device)
        .load_file(artifact.join(&checkpoint), &CompactRecorder::new(), &device)
        .wrap_err_with(|| format!("loading checkpoint '{checkpoint}'"))?;

    let state = Arc::new(AppState {
        model,
        tokenizer,
        device,
        stop_token,
    });

    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");

    println!("serving '{checkpoint}' on http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .wrap_err("binding listener")?;

    axum::serve(listener, app).await.wrap_err("serving")?;

    Ok(())
}

/// Spawn blocking generation, streaming token ids over a kanal channel
fn token_channel(
    state: Arc<AppState>,
    prompt_ids: Vec<i64>,
    max_new: usize,
    sampling: Sampling,
) -> AsyncReceiver<i64> {
    let (tx, rx) = kanal::bounded::<i64>(CHANNEL_CAP);

    tokio::task::spawn_blocking(move || {
        let result =
            state
                .model
                .generate_with(&prompt_ids, max_new, &sampling, &state.device, |t| {
                    tx.send(t).is_ok() // closed channel (client gone) → stop
                });
        if let Err(e) = result {
            eprintln!("generation failed: {e}"); // channel closes → client sees end of stream
        }
    });

    rx.to_async()
}

fn encode(state: &AppState, text: &str) -> Result<Vec<i64>, (axum::http::StatusCode, String)> {
    state
        .tokenizer
        .encode(text)
        .map(|v| v.into_iter().map(|x| x as i64).collect())
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("encode: {e}"),
            )
        })
}

/// `GET /v1/models` — most OpenAI clients query this on connect.
async fn models() -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": NAME, "object": "model", "owned_by": "local" }],
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    // Standard OpenAI `reasoning_effort` mapped to the model's binary thinking toggle: absent is
    // on, `none` is off, any real effort is on.
    let enable_thinking = req
        .reasoning_effort
        .map(ReasoningEffort::enable_thinking)
        .unwrap_or(true);

    // Map request turns to the template; the `<|think|>` cue is emitted by the template from
    // `enable_thinking`, so request system messages other than user/assistant are dropped here.
    let history: Vec<Message> = req
        .messages
        .iter()
        .filter_map(|m| match Role::from_chatml(&m.role) {
            Some(Role::User) => Some(Message::user(m.content.clone())),
            Some(Role::Model) => Some(Message::assistant(m.content.clone())),
            _ => None,
        })
        .collect();
    let history = if history.is_empty() {
        vec![Message::user(String::new())]
    } else {
        history
    };

    let prompt = render(&history, true, enable_thinking);

    let ids = match encode(&state, &prompt) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };

    let sampling = Sampling {
        temperature: req.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        top_k: req.top_k,
        top_p: req.top_p,
        min_p: req.min_p,
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        stop_token: state.stop_token,
        seed: req.seed,
    };

    let rx = token_channel(
        state.clone(),
        ids,
        req.max_tokens.unwrap_or(DEFAULT_MAX_NEW),
        sampling,
    );

    respond(state, rx, req.stream, true, req.stop).await
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let ids = match encode(&state, &req.prompt) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };

    let sampling = Sampling {
        temperature: req.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        top_k: req.top_k,
        top_p: req.top_p,
        min_p: req.min_p,
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        stop_token: state.stop_token,
        seed: req.seed,
    };

    let rx = token_channel(
        state.clone(),
        ids,
        req.max_tokens.unwrap_or(DEFAULT_MAX_NEW),
        sampling,
    );

    respond(state, rx, req.stream, false, req.stop).await
}

/// Drain the token channel into either an SSE stream of OpenAI delta chunks
/// (`stream`) or a single completion response. `chat` selects message/delta vs
/// text field shapes.
///
/// For chat, the thought channel is returned to the client in a separate `reasoning_content`
/// field: tokens between `<|channel>` and `<channel|>`, detected by their special-token ids and
/// never decoded, go to `reasoning_content` while the answer goes to `content`. Raw
/// `/v1/completions` has no such field, so the thought is omitted there. Stop strings match the
/// answer, not the thinking. `reasoning_content` is the DeepSeek/vLLM convention; OpenAI itself
/// does not return reasoning in chat completions.
async fn respond(
    state: Arc<AppState>,
    rx: AsyncReceiver<i64>,
    stream: bool,
    chat: bool,
    stop: Vec<String>,
) -> Response {
    let channel_open = SpecialToken::ChannelOpen as i64;
    let channel_close = SpecialToken::ChannelClose as i64;

    if stream {
        let tok = state.tokenizer.clone();

        let object = if chat {
            "chat.completion.chunk"
        } else {
            "text_completion"
        };

        let s = async_stream::stream! {
            let mut dec = DecodeStream::new(Vec::new(), true);
            let mut content = String::new();
            let mut in_thought = false;
            while let Ok(t) = rx.recv().await {
                // Channel markers are atomic special tokens; flip state, don't decode them.
                if t == channel_open { in_thought = true; continue; }
                if t == channel_close { in_thought = false; continue; }
                let Ok(Some(text)) = dec.step(tok.as_ref(), vec![t as u32]) else { continue };
                let choice = if in_thought {
                    // Thought: chat surfaces it as `reasoning_content`; completions has no such
                    // field, so drop it from the visible text.
                    if !chat { continue; }
                    json!({"index": 0, "delta": {"reasoning_content": text}, "finish_reason": null})
                } else if chat {
                    content.push_str(&text);
                    json!({"index": 0, "delta": {"content": text}, "finish_reason": null})
                } else {
                    content.push_str(&text);
                    json!({"index": 0, "text": text, "finish_reason": null})
                };
                let chunk = json!({"id": NAME, "object": object, "model": NAME, "choices": [choice]});
                yield Ok::<Event, std::convert::Infallible>(Event::default().data(chunk.to_string()));
                if !in_thought && stop.iter().any(|s| !s.is_empty() && content.contains(s)) {
                    break;
                }
            }

            let last = if chat {
                json!({"index": 0, "delta": {}, "finish_reason": "stop"})
            } else {
                json!({"index": 0, "text": "", "finish_reason": "stop"})
            };

            let chunk = json!({"id": NAME, "object": object, "model": NAME, "choices": [last]});

            yield Ok(Event::default().data(chunk.to_string()));

            yield Ok(Event::default().data("[DONE]"));
        };

        Sse::new(s).into_response()
    } else {
        let mut dec = DecodeStream::new(Vec::new(), true);
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut in_thought = false;

        while let Ok(t) = rx.recv().await {
            if t == channel_open {
                in_thought = true;
                continue;
            }
            if t == channel_close {
                in_thought = false;
                continue;
            }
            let Ok(Some(text)) = dec.step(state.tokenizer.as_ref(), vec![t as u32]) else {
                continue;
            };
            if in_thought {
                reasoning.push_str(&text);
            } else {
                content.push_str(&text);
                if stop.iter().any(|s| !s.is_empty() && content.contains(s)) {
                    break;
                }
            }
        }

        let choice = if chat {
            let mut message = json!({"role": "assistant", "content": content});
            if !reasoning.is_empty() {
                message["reasoning_content"] = json!(reasoning);
            }
            json!({"index": 0, "message": message, "finish_reason": "stop"})
        } else {
            json!({"index": 0, "text": content, "finish_reason": "stop"})
        };

        let object = if chat {
            "chat.completion"
        } else {
            "text_completion"
        };

        Json(json!({"id": NAME, "object": object, "model": NAME, "choices": [choice]}))
            .into_response()
    }
}
