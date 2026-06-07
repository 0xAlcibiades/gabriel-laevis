//! Conversation template
//!
//! Leverages Gemma-4-style control tokens: turns, roles, rendering.
//!
//! - A turn is `<|turn>{role}\n…<turn|>`.
//! - Roles render as plain words.
//! - Optional reasoning renders in a thought channel (`<|channel>thought\n…\n<channel|>`)
//!   before the content.
//!
//! TODO:
//! Only the text + thinking path is rendered today. The other reserved control tokens
//! (tool-calling, image/audio/video — see [`SpecialToken`]) are kept in the vocab on purpose
//! so adding those capabilities later needs no from-scratch re-pretrain; their rendering
//! lands with the tool-use loop / multimodal work.

/// The reserved special-token vocabulary, in id order. `train::tokenizer` reserves these in
/// exactly this declaration order, so each variant's discriminant is its vocab id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumIter)]
pub enum SpecialToken {
    Pad,
    Bos,
    Eos,
    TurnOpen,
    TurnClose,
    Think,
    ChannelOpen,
    ChannelClose,
    ToolOpen,
    ToolClose,
    ToolCallOpen,
    ToolCallClose,
    ToolResponseOpen,
    ToolResponseClose,
    StringDelim,
    ImageOpen,
    ImageClose,
    AudioOpen,
    AudioClose,
    VideoOpen,
    VideoClose,
    Image,
    Audio,
    Video,
}

impl SpecialToken {
    /// The literal token string the tokenizer reserves and `render` emits.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pad => "<pad>",
            Self::Bos => "<bos>",
            Self::Eos => "<eos>",
            Self::TurnOpen => "<|turn>",
            Self::TurnClose => "<turn|>",
            Self::Think => "<|think|>",
            Self::ChannelOpen => "<|channel>",
            Self::ChannelClose => "<channel|>",
            Self::ToolOpen => "<|tool>",
            Self::ToolClose => "<tool|>",
            Self::ToolCallOpen => "<|tool_call>",
            Self::ToolCallClose => "<tool_call|>",
            Self::ToolResponseOpen => "<|tool_response>",
            Self::ToolResponseClose => "<tool_response|>",
            Self::StringDelim => "<|\"|>",
            Self::ImageOpen => "<|image>",
            Self::ImageClose => "<image|>",
            Self::AudioOpen => "<|audio>",
            Self::AudioClose => "<audio|>",
            Self::VideoOpen => "<|video>",
            Self::VideoClose => "<video|>",
            Self::Image => "<|image|>",
            Self::Audio => "<|audio|>",
            Self::Video => "<|video|>",
        }
    }
}

/// Loss-ignore / pad target id. `<pad>` is id 0, so it never appears as a real target: the
/// cross-entropy loss drops it (`with_pad_tokens`), masking prompt and padding.
pub const IGNORE_ID: usize = SpecialToken::Pad as usize;

/// Turn terminator `<turn|>`, used as the generation stop token (completions end at the turn
/// boundary, not the length cap). Known by construction from [`SpecialToken`].
pub const TURN_END_ID: i64 = SpecialToken::TurnClose as i64;

/// Conversation role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    Developer,
    User,
    Model,
    Tool,
}

impl Role {
    #[inline(always)]
    const fn tag(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::Developer => "developer",
            Role::User => "user",
            Role::Model => "model",
            Role::Tool => "tool",
        }
    }

    /// Map a standard chat-API role name (ChatML / OpenAI) to a [`Role`]. `assistant` maps to
    /// `Model`; an unknown name returns `None`. Shared by request handling and dataset ingestion
    /// so the role vocabulary has one source.
    pub fn from_chatml(role: &str) -> Option<Role> {
        Some(match role {
            "system" => Role::System,
            "developer" => Role::Developer,
            "user" => Role::User,
            "assistant" => Role::Model,
            "tool" => Role::Tool,
            _ => return None,
        })
    }
}

/// One conversation turn.
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub thinking: Option<String>,
}

impl Message {
    #[inline]
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            thinking: None,
        }
    }

    // Ergonomic constructors
    pub fn system(c: impl Into<String>) -> Self {
        Self::new(Role::System, c)
    }
    pub fn developer(c: impl Into<String>) -> Self {
        Self::new(Role::Developer, c)
    }
    pub fn user(c: impl Into<String>) -> Self {
        Self::new(Role::User, c)
    }
    pub fn assistant(c: impl Into<String>) -> Self {
        Self::new(Role::Model, c)
    }
    pub fn tool(c: impl Into<String>) -> Self {
        Self::new(Role::Tool, c)
    }

    pub fn with_thinking(mut self, t: impl Into<String>) -> Self {
        self.thinking = Some(t.into());
        self
    }
}

/// The gemma-4-derived chat template: the single source of the conversation format. serve and the
/// training data path both render prompts through it, and it is written beside the trained
/// tokenizer so external tooling formats conversations identically. The compiled environment is a
/// process-wide singleton in [`crate::config::chat_template_env`].
pub(crate) const CHAT_TEMPLATE: &str = include_str!("../templates/chat_template.jinja");

/// The embedded chat template source, written beside the trained tokenizer so external tooling
/// renders conversations identically to `render`.
pub fn chat_template() -> &'static str {
    CHAT_TEMPLATE
}

/// The thought channel's name, emitted as `<|channel>{THOUGHT_CHANNEL}\n…<channel|>`.
pub(crate) const THOUGHT_CHANNEL: &str = "thought";

/// The message fields the chat template reads off each turn.
pub(crate) const MSG_ROLE: &str = "role";
pub(crate) const MSG_CONTENT: &str = "content";
pub(crate) const MSG_REASONING: &str = "reasoning";

/// A `Message` as the template consumes it: `role`, `content`, and `reasoning`. The template
/// re-renders `reasoning` from history only when interleaved with tool calls.
fn message_value(m: &Message) -> minijinja::Value {
    let mut obj = serde_json::Map::new();
    obj.insert(MSG_ROLE.into(), m.role.tag().into());
    obj.insert(MSG_CONTENT.into(), m.content.clone().into());
    if let Some(t) = &m.thinking {
        obj.insert(MSG_REASONING.into(), t.clone().into());
    }
    minijinja::Value::from_serialize(serde_json::Value::Object(obj))
}

/// Render a conversation to the model's prompt format via the chat template. `enable_thinking`
/// emits the `<|think|>` cue in the system turn; `add_generation_prompt` appends the open model
/// turn. `bos_token` is empty: the tokenizer's encode prepends `<bos>`, so the template must not
/// duplicate it.
pub fn render(messages: &[Message], add_generation_prompt: bool, enable_thinking: bool) -> String {
    let msgs: Vec<minijinja::Value> = messages.iter().map(message_value).collect();
    crate::config::chat_template_env()
        .get_template("chat")
        .expect("chat template")
        .render(minijinja::context! {
            bos_token => "",
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
            messages => msgs,
        })
        .expect("chat template render")
}

/// The assistant turn's generated completion, appended after a prompt's open model turn: an
/// optional `<|channel>thought>` block, the answer, then the turn terminator. Built from
/// [`SpecialToken`]: it is the model's generation format, which the template does not re-render
/// from history. Training uses it to construct the supervised target.
pub fn assistant_completion(content: &str, thinking: Option<&str>) -> String {
    let mut s = String::new();
    if let Some(t) = thinking {
        s.push_str(SpecialToken::ChannelOpen.as_str());
        s.push_str(THOUGHT_CHANNEL);
        s.push('\n');
        s.push_str(t);
        s.push('\n');
        s.push_str(SpecialToken::ChannelClose.as_str());
    }
    s.push_str(content);
    s.push_str(SpecialToken::TurnClose.as_str());
    s.push('\n');
    s
}

/// The prompt the model must continue for a single user turn, with no thinking cue.
pub fn render_prompt(user: &str) -> String {
    render(&[Message::user(user)], true, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_prefix_of_full() {
        // The supervised target is appended after the prompt, so the prompt is a literal prefix.
        let p = render_prompt("hi");
        let f = format!("{p}{}", assistant_completion("yo", None));
        assert!(f.starts_with(&p), "prompt must prefix full for SFT masking");
    }

    // Token/role strings come from the enum, never inline literals.
    fn turn_open(role: Role) -> String {
        format!("{}{}\n", SpecialToken::TurnOpen.as_str(), role.tag())
    }

    #[test]
    fn render_emits_turns_and_think_cue() {
        // enable_thinking emits the `<|think|>` cue in the leading system turn; a system message
        // renders into that same turn; generation prompt opens a model turn.
        let on = render(
            &[Message::system("be nice"), Message::user("2+2?")],
            true,
            true,
        );
        let think_cue = format!(
            "{}{}",
            turn_open(Role::System),
            SpecialToken::Think.as_str()
        );
        assert!(on.contains(&think_cue), "think cue: {on:?}");
        assert!(on.contains("be nice"), "system content: {on:?}");
        let user_turn = format!(
            "{}2+2?{}",
            turn_open(Role::User),
            SpecialToken::TurnClose.as_str()
        );
        assert!(on.contains(&user_turn), "user turn: {on:?}");
        assert!(
            on.ends_with(&turn_open(Role::Model)),
            "gen prompt tail: {on:?}"
        );

        // enable_thinking off: no `<|think|>` cue.
        let off = render(&[Message::user("2+2?")], true, false);
        assert!(
            !off.contains(SpecialToken::Think.as_str()),
            "no think cue when off: {off:?}"
        );
    }

    #[test]
    fn assistant_completion_builds_thought_then_answer() {
        let with = format!(
            "{}{}\nadd them\n{}4{}\n",
            SpecialToken::ChannelOpen.as_str(),
            THOUGHT_CHANNEL,
            SpecialToken::ChannelClose.as_str(),
            SpecialToken::TurnClose.as_str(),
        );
        assert_eq!(assistant_completion("4", Some("add them")), with);
        assert_eq!(
            assistant_completion("4", None),
            format!("4{}\n", SpecialToken::TurnClose.as_str())
        );
    }

    /// Render the gemma-4 template through minijinja+pycompat, covering the namespace/loop
    /// machinery and the tools path macros, `dictsort`, and `format_argument`.
    #[test]
    fn minijinja_renders_gemma_template() {
        let tmpl = crate::config::chat_template_env()
            .get_template("chat")
            .unwrap();

        // Reasoning prompt, no tools: core loop, namespace, think cue, roles.
        let out = tmpl
            .render(serde_json::json!({
                "bos_token": "",
                "add_generation_prompt": true,
                "enable_thinking": true,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            }))
            .expect("render reasoning prompt");
        let think_cue = format!(
            "{}{}",
            turn_open(Role::System),
            SpecialToken::Think.as_str()
        );
        assert!(out.contains(&think_cue), "think cue: {out:?}");
        let user_turn = format!(
            "{}What is 2+2?{}",
            turn_open(Role::User),
            SpecialToken::TurnClose.as_str()
        );
        assert!(out.contains(&user_turn), "user turn: {out:?}");
        assert!(
            out.ends_with(&turn_open(Role::Model)),
            "gen prompt tail: {out:?}"
        );

        // Tools path: declaration macros, dictsort, format_argument.
        let out2 = tmpl
            .render(serde_json::json!({
                "bos_token": "",
                "add_generation_prompt": true,
                "tools": [{"function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "City"}},
                        "required": ["city"]
                    }
                }}],
                "messages": [{"role": "user", "content": "weather in Paris?"}],
            }))
            .expect("render tools prompt");
        assert!(
            out2.contains(SpecialToken::ToolOpen.as_str()) && out2.contains("get_weather"),
            "tool declaration: {out2:?}"
        );
    }

    /// The vendored template must stay consistent with our typed format: it references the token
    /// strings, the thought-channel name, the message fields, and the model role we render through
    /// it. Catches silent drift between `chat_template.jinja` and `SpecialToken`/`Role`.
    #[test]
    fn template_matches_typed_format() {
        for tok in [
            SpecialToken::TurnOpen,
            SpecialToken::TurnClose,
            SpecialToken::Think,
            SpecialToken::ChannelOpen,
            SpecialToken::ChannelClose,
            SpecialToken::ToolOpen,
            SpecialToken::ToolClose,
            SpecialToken::ToolCallOpen,
            SpecialToken::ToolCallClose,
            SpecialToken::ToolResponseOpen,
            SpecialToken::ToolResponseClose,
            SpecialToken::StringDelim,
            SpecialToken::Image,
            SpecialToken::Audio,
            SpecialToken::Video,
        ] {
            assert!(
                CHAT_TEMPLATE.contains(tok.as_str()),
                "template missing token {:?}",
                tok.as_str()
            );
        }
        assert!(
            CHAT_TEMPLATE.contains(&format!(
                "{}{THOUGHT_CHANNEL}",
                SpecialToken::ChannelOpen.as_str()
            )),
            "template missing the thought channel"
        );
        for field in [MSG_ROLE, MSG_CONTENT, MSG_REASONING] {
            assert!(
                CHAT_TEMPLATE.contains(field),
                "template missing message field {field}"
            );
        }
        assert!(
            CHAT_TEMPLATE.contains(Role::Model.tag()),
            "template missing the model role"
        );
    }
}
