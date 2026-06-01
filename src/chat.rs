//! ChatML conversation template — roles, messages, and rendering.

/// Loss-ignore target id (SmolLM2 `<|endoftext|>` = 0). Targets set to this are
/// dropped by the cross-entropy loss (`with_pad_tokens`), so SFT trains only on
/// the assistant response. It never appears as a target in normal text, so it's
/// a no-op for pretraining.
pub const IGNORE_ID: usize = 0;

/// Conversation role. `Developer` and `Tool` cover the instruction-hierarchy and
/// tool-calling conventions; they render as plain ChatML role tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

impl Role {
    #[inline(always)]
    const fn tag(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::Developer => "developer",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One conversation turn.
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Optional reasoning trace, rendered as `<think>…</think>` before content.
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
        Self::new(Role::Assistant, c)
    }
    pub fn tool(c: impl Into<String>) -> Self {
        Self::new(Role::Tool, c)
    }

    pub fn with_thinking(mut self, t: impl Into<String>) -> Self {
        self.thinking = Some(t.into());
        self
    }
}

/// Render a conversation to ChatML. With `add_generation_prompt`, append an open
/// assistant turn for the model to continue. Sized up front into a single buffer.
pub fn render(messages: &[Message], add_generation_prompt: bool) -> String {
    // Estimate the buffer size to avoid reallocations (tags + content per turn).
    let est_len: usize = messages.iter().map(|m| m.content.len() + 64).sum::<usize>()
        + if add_generation_prompt { 32 } else { 0 };

    let mut out = String::with_capacity(est_len);

    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(m.role.tag());
        out.push('\n');

        if let Some(ref t) = m.thinking {
            out.push_str("<think>");
            out.push_str(t);
            out.push_str("</think>");
        }

        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// The prompt (user turn + open assistant turn) the model must continue.
pub fn render_prompt(user: &str) -> String {
    render(&[Message::user(user)], true)
}

/// The full templated SFT example: user turn + assistant response, closed. Note
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
        assert!(s.contains("<|im_start|>system\nbe nice<|im_end|>"));
        assert!(s.contains("<|im_start|>assistant\n<think>add them</think>4<|im_end|>"));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }
}
