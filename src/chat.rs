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
//! The tokenizer reserves the full modern control-token set (tool-calling, image/audio/video);
//! only the text + thinking path is rendered here. This needs an update accordingly.

/// `<pad>` is a reserved special token with id 0, so it never appears as a real target.
pub const IGNORE_ID: usize = 0;

/// Token id of the turn terminator `<turn|>`, used as the generation stop token.
///
/// Returns `None` if the tokenizer doesn't map it to a single id.
///
/// TODO:
/// However, it is a known ID by construction, so we should choose here instead
/// the known numeric id from tokenizer.rs
pub fn turn_end_id(tok: &fastokens::Tokenizer) -> Option<i64> {
    let ids = tok.encode("<turn|>").ok()?;
    (ids.len() == 1).then(|| ids[0] as i64)
}

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

/// Render a conversation to the turn format.
pub fn render(messages: &[Message], add_generation_prompt: bool) -> String {
    // Estimate the buffer size to avoid spurious reallocations.
    let est_len: usize = messages.iter().map(|m| m.content.len() + 64).sum::<usize>()
        + if add_generation_prompt { 16 } else { 0 };

    let mut out = String::with_capacity(est_len);

    for m in messages {
        out.push_str("<|turn>");
        out.push_str(m.role.tag());
        out.push('\n');

        if let Some(ref t) = m.thinking {
            out.push_str("<|channel>thought\n");
            out.push_str(t);
            out.push_str("\n<channel|>");
        }

        out.push_str(&m.content);
        out.push_str("<turn|>\n");
    }

    if add_generation_prompt {
        out.push_str("<|turn>model\n");
    }
    out
}

/// The prompt which the model must continue.
pub fn render_prompt(user: &str) -> String {
    render(&[Message::user(user)], true)
}

/// The full templated example for SFT: user turn + model response, closed.
///
/// NOTE:
/// `render_prompt(user)` is a prefix of this, which the response-loss masking in
/// `build_sft_row` relies on.
pub fn render_full(user: &str, assistant: &str) -> String {
    render(&[Message::user(user), Message::assistant(assistant)], false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_prefix_of_full() {
        let p = render_prompt("hi");
        let f = render_full("hi", "yo");
        assert!(f.starts_with(&p), "prompt must prefix full for SFT masking");
    }

    #[test]
    fn renders_roles_and_thinking() {
        let s = render(
            &[
                Message::system("be nice"),
                Message::user("2+2?"),
                Message::assistant("4").with_thinking("add them"),
            ],
            true,
        );
        assert!(s.contains("<|turn>system\nbe nice<turn|>"));
        assert!(s.contains("<|turn>model\n<|channel>thought\nadd them\n<channel|>4<turn|>"));
        assert!(s.ends_with("<|turn>model\n"));
    }
}
