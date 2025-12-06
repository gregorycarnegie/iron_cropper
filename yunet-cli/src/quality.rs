//! Quality filter construction and management.

use yunet_utils::{QualityFilter, config::QualityAutomationSettings};

/// Construct a `QualityFilter` from persistent automation settings.
pub fn build_quality_filter(settings: &QualityAutomationSettings) -> QualityFilter {
    let mut filter = QualityFilter::new(settings.min_quality);
    filter.auto_select = settings.auto_select_best_face;
    filter.auto_skip_no_high = settings.auto_skip_no_high_quality;
    filter.suffix_enabled = settings.quality_suffix;
    filter
}

#[cfg(test)]
mod tests {
    use super::*;
    use yunet_utils::Quality;

    #[test]
    fn build_quality_filter_reflects_settings() {
        let settings = QualityAutomationSettings {
            auto_select_best_face: true,
            min_quality: Some(Quality::High),
            auto_skip_no_high_quality: true,
            quality_suffix: true,
        };

        let filter = build_quality_filter(&settings);
        assert_eq!(filter.min_quality, Some(Quality::High));
        assert!(filter.auto_select);
        assert!(filter.auto_skip_no_high);
        assert!(filter.suffix_enabled);
    }

    #[test]
    fn build_quality_filter_defaults_to_none() {
        let settings = QualityAutomationSettings::default();
        let filter = build_quality_filter(&settings);
        assert_eq!(filter.min_quality, None);
        assert!(!filter.auto_select);
        assert!(!filter.auto_skip_no_high);
        assert!(!filter.suffix_enabled);
    }
}
