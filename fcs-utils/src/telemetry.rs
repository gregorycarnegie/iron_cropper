//! Lightweight timing utilities for optional performance tracing.
//!
//! The helpers in this module provide a simple RAII guard that records the
//! elapsed duration of a scoped operation and logs it when the guard is dropped.
//! Logging only occurs when both the requested log level is enabled and the
//! caller explicitly opts in (via [`timing_guard_if`]). This keeps the overhead
//! negligible when tracing is disabled.

use log::{Level, LevelFilter, log, log_enabled};
use std::{
    borrow::Cow,
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    time::{Duration, Instant},
};

static TELEMETRY_ENABLED: AtomicBool = AtomicBool::new(false);
static TELEMETRY_LEVEL: AtomicU8 = AtomicU8::new(LevelFilter::Off as u8);

/// RAII helper that logs how long an operation took when dropped.
///
/// Guards are usually created via [`timing_guard`] or [`timing_guard_if`] so
/// most callers do not need to interact with this type directly.
pub struct TimingGuard {
    label: Cow<'static, str>,
    level: Level,
    start: Instant,
    active: bool,
}

impl TimingGuard {
    /// Create a guard with an explicit activation flag.
    fn new(label: Cow<'static, str>, level: Level, active: bool) -> Self {
        Self {
            label,
            level,
            start: Instant::now(),
            active,
        }
    }

    /// Returns `true` when the guard will emit a log entry on drop.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Returns the elapsed duration since the guard was created.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Consume the guard and return the elapsed duration without logging.
    pub fn finish(mut self) -> Duration {
        let duration = self.start.elapsed();
        self.active = false;
        duration
    }
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        if self.active {
            let duration = self.start.elapsed();
            log!(
                target: "fcs::telemetry",
                self.level,
                "{} completed in {:.2?}",
                self.label,
                duration
            );
        }
    }
}

/// Create a timing guard that logs at the provided level when that level is enabled.
///
/// Logging only occurs when the global logger allows the provided level (e.g. via
/// `RUST_LOG=fcs=debug`). This is the preferred helper when the guard should
/// activate automatically based on the current log filter.
pub fn timing_guard(label: impl Into<Cow<'static, str>>, level: Level) -> TimingGuard {
    timing_guard_if(label, level, true)
}

/// Create a timing guard that also respects an explicit boolean flag.
///
/// This variant gives callers the ability to toggle telemetry at runtime (e.g.
/// via configuration) in addition to the global log filter.
pub fn timing_guard_if(
    label: impl Into<Cow<'static, str>>,
    level: Level,
    enabled: bool,
) -> TimingGuard {
    let label = label.into();
    let active =
        enabled && telemetry_allows(level) && log_enabled!(target: "fcs::telemetry", level);
    TimingGuard::new(label, level, active)
}

/// Configure the global telemetry state.
///
/// Callers should invoke this whenever user preferences change so guards can
/// pick up the new settings.
pub fn configure(enabled: bool, level: LevelFilter) {
    TELEMETRY_ENABLED.store(enabled, Ordering::Relaxed);
    TELEMETRY_LEVEL.store(filter_index(level), Ordering::Relaxed);
}

/// Returns whether telemetry logging is currently enabled.
pub fn telemetry_enabled() -> bool {
    TELEMETRY_ENABLED.load(Ordering::Relaxed)
}

/// Returns the maximum telemetry logging level.
pub fn telemetry_level() -> LevelFilter {
    filter_from_index(TELEMETRY_LEVEL.load(Ordering::Relaxed))
}

/// Returns `true` when telemetry is enabled and the provided level is within
/// the configured threshold.
pub fn telemetry_allows(level: Level) -> bool {
    if !telemetry_enabled() {
        return false;
    }
    let threshold = TELEMETRY_LEVEL.load(Ordering::Relaxed);
    level_index(level) <= threshold
}

fn level_index(level: Level) -> u8 {
    match level {
        Level::Error => 1,
        Level::Warn => 2,
        Level::Info => 3,
        Level::Debug => 4,
        Level::Trace => 5,
    }
}

fn filter_index(filter: LevelFilter) -> u8 {
    match filter {
        LevelFilter::Off => 0,
        LevelFilter::Error => 1,
        LevelFilter::Warn => 2,
        LevelFilter::Info => 3,
        LevelFilter::Debug => 4,
        LevelFilter::Trace => 5,
    }
}

fn filter_from_index(value: u8) -> LevelFilter {
    match value {
        1 => LevelFilter::Error,
        2 => LevelFilter::Warn,
        3 => LevelFilter::Info,
        4 => LevelFilter::Debug,
        5 => LevelFilter::Trace,
        _ => LevelFilter::Off,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reset global telemetry state between tests so they don't interfere.
    fn reset() {
        configure(false, LevelFilter::Off);
    }

    #[test]
    fn configure_roundtrips_enabled_flag_and_level() {
        configure(true, LevelFilter::Debug);
        assert!(telemetry_enabled());
        assert_eq!(telemetry_level(), LevelFilter::Debug);

        configure(false, LevelFilter::Off);
        assert!(!telemetry_enabled());
        assert_eq!(telemetry_level(), LevelFilter::Off);
    }

    #[test]
    fn telemetry_allows_respects_enabled_flag() {
        reset();
        // Disabled → never allowed regardless of level.
        assert!(!telemetry_allows(Level::Error));

        configure(true, LevelFilter::Info);
        assert!(telemetry_allows(Level::Error));
        assert!(telemetry_allows(Level::Warn));
        assert!(telemetry_allows(Level::Info));
        assert!(!telemetry_allows(Level::Debug));
        assert!(!telemetry_allows(Level::Trace));

        reset();
    }

    #[test]
    fn telemetry_allows_all_levels_at_trace() {
        configure(true, LevelFilter::Trace);
        for level in [
            Level::Error,
            Level::Warn,
            Level::Info,
            Level::Debug,
            Level::Trace,
        ] {
            assert!(telemetry_allows(level), "expected {level:?} to be allowed");
        }
        reset();
    }

    #[test]
    fn timing_guard_finish_suppresses_log_and_returns_duration() {
        // finish() should mark the guard inactive so Drop does not log.
        let guard = timing_guard_if("test_op", Level::Debug, false);
        assert!(!guard.is_active());
        let elapsed = guard.finish();
        // We can't assert an exact value, but it should be a valid Duration.
        assert!(elapsed.as_nanos() < 1_000_000_000); // < 1 second
    }

    #[test]
    fn timing_guard_elapsed_increases_over_time() {
        let guard = timing_guard_if("op", Level::Debug, false);
        let d1 = guard.elapsed();
        // Spin briefly to ensure time advances.
        let start = std::time::Instant::now();
        while start.elapsed().as_nanos() < 1_000 {}
        let d2 = guard.elapsed();
        assert!(d2 >= d1);
        drop(guard);
    }

    #[test]
    fn timing_guard_if_inactive_when_disabled() {
        let guard = timing_guard_if("op", Level::Debug, false);
        assert!(!guard.is_active());
    }
}
