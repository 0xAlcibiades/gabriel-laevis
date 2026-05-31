//! Training stages and shared context. The `train` binary dispatches into these.

pub mod context;
pub mod dpo;
pub mod grpo;
pub mod pretrain;
pub mod sft;

pub use context::TrainingContext;

/// Every stage checkpoints (and resumes) on this fixed step interval, so a crash at
/// hour N of an overnight run loses at most this many steps, not the whole run.
pub const STEPS_PER_CHECKPOINT: usize = 1000;

/// How a run is sliced for checkpointing. Burn checkpoints once per epoch over
/// fixed-size epochs, so we run whole `STEPS_PER_CHECKPOINT`-step "checkpoint epochs".
pub struct Schedule {
    /// Number of checkpoint-epochs (i.e. number of saves).
    pub ckpt_epochs: usize,
    /// Samples drawn per checkpoint-epoch (with replacement via `SamplerDataset`).
    pub epoch_samples: usize,
    /// Total optimizer steps across the whole run (for LR schedules).
    pub total_steps: usize,
}

/// Slice a run of `passes` over `n_items` (batched by `batch`) into equal
/// `STEPS_PER_CHECKPOINT`-step checkpoint-epochs. The interval is capped at the
/// requested length so a short run isn't inflated up to a full interval. Epochs sample
/// with replacement, so one "pass" is ~63% unique-item coverage, not an exact sweep —
/// fine at these scales, and the only mechanism Burn offers for sub-pass checkpoints.
pub fn schedule(n_items: usize, batch: usize, passes: usize) -> Schedule {
    let steps_per_pass = (n_items / batch.max(1)).max(1);
    let requested_steps = (steps_per_pass * passes.max(1)).max(1);
    let steps_per_ckpt = STEPS_PER_CHECKPOINT.min(requested_steps);
    let ckpt_epochs = requested_steps.div_ceil(steps_per_ckpt).max(1);
    Schedule {
        ckpt_epochs,
        epoch_samples: steps_per_ckpt * batch,
        total_steps: ckpt_epochs * steps_per_ckpt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_caps_interval_and_rounds_up() {
        // Short run: interval capped at the requested length, single checkpoint.
        let s = schedule(800, 8, 1); // 100 steps/pass
        assert_eq!(s.ckpt_epochs, 1);
        assert_eq!(s.total_steps, 100);
        assert_eq!(s.epoch_samples, 100 * 8);

        // Long run: sliced into 1000-step checkpoint-epochs, rounded up.
        let s = schedule(40_000, 8, 1); // 5000 steps/pass
        assert_eq!(s.ckpt_epochs, 5);
        assert_eq!(s.total_steps, 5000);
        assert_eq!(s.epoch_samples, 1000 * 8);

        // Non-divisible: rounds up to cover the request (1850 -> 2 x 1000).
        let s = schedule(7400, 4, 1); // 1850 steps/pass
        assert_eq!(s.ckpt_epochs, 2);
        assert_eq!(s.total_steps, 2000);
    }

    #[test]
    fn schedule_handles_degenerate_inputs() {
        let s = schedule(0, 0, 0);
        assert_eq!(s.ckpt_epochs, 1);
        assert_eq!(s.total_steps, 1);
    }
}
