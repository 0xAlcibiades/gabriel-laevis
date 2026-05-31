//! Held-out solve-rate metric: the honest "did this stage help?" gate, plotted live.
//!
//! Training loss/accuracy says nothing about whether sampled completions actually
//! *solve* held-out problems. So a generative stage's validation step reports
//! [`SolveCounts`] — greedy pass@1 and sampled pass@k over its disjoint valid set —
//! and [`SolveRateMetric`] renders them in the Learner TUI alongside the loss. The
//! pass@1-vs-pass@k gap over training is the RL mode-collapse signal (Cobbe et al.,
//! GSM8K, arXiv:2110.14168, Fig. 3): pass@1 climbs while pass@k craters as the
//! policy's coverage narrows.
//!
//! The metric is task-agnostic — it consumes counts, not GSM8K specifics — so any
//! verifiable task feeds it by supplying its own `correct(completion, reference)`
//! predicate (here, [`gsm8k_correct`]).

use std::sync::Arc;

use burn::train::metric::state::{FormatOptions, NumericMetricState};
use burn::train::metric::{
    Adaptor, ItemLazy, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric,
    NumericAttributes, NumericEntry, SerializedEntry,
};

/// Last signed integer appearing in `s`, or `None` if it has no digits. The verifiable
/// signal for GSM8K (answers are integers) and the GRPO reward.
pub fn extract_last_int(s: &str) -> Option<String> {
    let mut last = None;
    let mut cur = String::new();
    let has_digit = |t: &str| t.chars().any(|c| c.is_ascii_digit());
    for ch in s.chars() {
        if ch.is_ascii_digit() || (ch == '-' && cur.is_empty()) {
            cur.push(ch);
        } else {
            if has_digit(&cur) {
                last = Some(cur.clone());
            }
            cur.clear();
        }
    }
    if has_digit(&cur) {
        last = Some(cur);
    }
    last
}

/// GSM8K correctness: the completion's last integer equals the answer.
pub fn gsm8k_correct(completion: &str, answer: &str) -> bool {
    extract_last_int(completion).as_deref() == Some(answer)
}

// --- Live solve-rate metric (plotted in the Learner TUI) ---

/// Per-validation-batch solve counts: a training stage's `InferenceStep` returns this
/// so the held-out pass@1 / pass@k render live as numeric metrics. Carries no tensors,
/// so [`ItemLazy`] is a no-op — the work (generation) already happened in the step.
#[derive(Debug, Clone, Copy)]
pub struct SolveCounts {
    /// Prompts solved by the single greedy sample.
    pub pass1: usize,
    /// Prompts solved by at least one of the k sampled completions.
    pub pass_k: usize,
    /// Prompts in this batch.
    pub total: usize,
}

impl ItemLazy for SolveCounts {
    type ItemSync = SolveCounts;
    fn sync(self) -> Self::ItemSync {
        self
    }
}

/// [`SolveRateMetric`] input — full per-batch counts; the metric picks pass@1 vs
/// pass@k by its [`SolveKind`], so both metrics share one [`Adaptor`] impl.
pub struct SolveRateInput {
    pass1: usize,
    pass_k: usize,
    total: usize,
}

impl Adaptor<SolveRateInput> for SolveCounts {
    fn adapt(&self) -> SolveRateInput {
        SolveRateInput {
            pass1: self.pass1,
            pass_k: self.pass_k,
            total: self.total,
        }
    }
}

#[derive(Clone, Copy)]
enum SolveKind {
    Pass1,
    PassK,
}

/// A held-out solve-rate metric (a percentage, higher is better), plottable in the
/// TUI. One instance tracks one rate — pass@1 or pass@k — selected at construction.
#[derive(Clone)]
pub struct SolveRateMetric {
    name: MetricName,
    state: NumericMetricState,
    kind: SolveKind,
}

impl SolveRateMetric {
    /// Greedy pass@1.
    pub fn pass1() -> Self {
        Self {
            name: Arc::new("Pass@1".to_string()),
            state: NumericMetricState::default(),
            kind: SolveKind::Pass1,
        }
    }
    /// Sampled pass@k (`any of k correct`).
    pub fn pass_k(k: usize) -> Self {
        Self {
            name: Arc::new(format!("Pass@{k}")),
            state: NumericMetricState::default(),
            kind: SolveKind::PassK,
        }
    }
}

impl Metric for SolveRateMetric {
    type Input = SolveRateInput;

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let solved = match self.kind {
            SolveKind::Pass1 => input.pass1,
            SolveKind::PassK => input.pass_k,
        };
        let pct = solved as f64 / input.total.max(1) as f64 * 100.0;
        self.state.update(
            pct,
            input.total,
            FormatOptions::new(self.name()).unit("%").precision(1),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for SolveRateMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }
    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_trailing_answer() {
        assert_eq!(extract_last_int("so 2+2 = 4").as_deref(), Some("4"));
        assert_eq!(
            extract_last_int("steps: 10, 20, then -5").as_deref(),
            Some("-5")
        );
        assert_eq!(extract_last_int("no digits here"), None);
    }

    #[test]
    fn gsm8k_correct_matches_last_int() {
        assert!(gsm8k_correct("the answer is #### 42", "42"));
        assert!(!gsm8k_correct("the answer is 41", "42"));
        // Trailing prose after the number still resolves to the last integer.
        assert!(gsm8k_correct("42 dollars total, so 7 dozen", "7"));
    }
}
