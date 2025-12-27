#[cfg(test)]
mod tests {
    use crate::shape::{CropShape, PolygonCornerStyle, outline_points_for_rect};

    #[test]
    fn test_bezier_polygon_generation() {
        let shape = CropShape::Polygon {
            sides: 4,
            rotation_deg: 0.0,
            corner_style: PolygonCornerStyle::Bezier { tension: 0.5 },
        };

        let points = outline_points_for_rect(100.0, 100.0, &shape);

        // We expect points to be generated.
        // 4 sides * 16 segments = 64 points roughly
        assert!(!points.is_empty());
        assert_eq!(points.len(), 64);

        // Check bounds roughly
        for (x, y) in &points {
            assert!(*x >= 0.0 && *x <= 100.0);
            assert!(*y >= 0.0 && *y <= 100.0);
        }
    }

    #[test]
    fn test_bezier_polygon_zero_tension() {
        let shape = CropShape::Polygon {
            sides: 3,
            rotation_deg: 0.0,
            corner_style: PolygonCornerStyle::Bezier { tension: 0.0 },
        };

        let points = outline_points_for_rect(100.0, 100.0, &shape);

        // Zero tension should return original vertices (3 for triangle)
        assert_eq!(points.len(), 3);
    }

    #[test]
    fn test_star_generation() {
        let shape = CropShape::Star {
            points: 5,
            inner_radius_pct: 0.5,
            rotation_deg: 0.0,
        };

        let points = outline_points_for_rect(100.0, 100.0, &shape);

        // 5 points * 2 (inner + outer) = 10 vertices
        assert_eq!(points.len(), 10);
    }

    #[test]
    fn test_vignette_mask_application() {
        use crate::shape::{CropShape, apply_shape_mask};
        use image::RgbaImage;
        let mut img = RgbaImage::new(100, 100);
        apply_shape_mask(
            &mut img,
            &CropShape::Ellipse,
            0.1,
            1.0,
            crate::color::RgbaColor::opaque(0, 0, 0),
        );
    }

    /// Test that shape masking produces transparent pixels outside the shape
    /// regardless of vignette_intensity setting.
    #[test]
    fn test_ellipse_crop_transparent_outside_zero_intensity() {
        use crate::shape::{CropShape, apply_shape_mask};
        use image::{Rgba, RgbaImage};

        // Create a 100x100 white opaque image
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));

        // Apply ellipse mask with vignette_intensity = 0 (no vignette color)
        apply_shape_mask(
            &mut img,
            &CropShape::Ellipse,
            0.0, // no softness - hard edge
            0.0, // zero intensity - should still crop!
            crate::color::RgbaColor::opaque(0, 0, 0),
        );

        // Corner pixel (0,0) is definitely outside the ellipse - must be transparent
        let corner = img.get_pixel(0, 0);
        assert_eq!(
            corner[3], 0,
            "Corner pixel should be fully transparent (alpha=0), got alpha={}",
            corner[3]
        );

        // Center pixel (50,50) is inside the ellipse - must remain opaque
        let center = img.get_pixel(50, 50);
        assert_eq!(
            center[3], 255,
            "Center pixel should remain fully opaque (alpha=255), got alpha={}",
            center[3]
        );
    }

    /// Test that shape masking with softness produces transparent edges
    /// regardless of vignette_intensity setting. With softness, edges have
    /// gradual alpha falloff, but RGB should not be affected by vignette color
    /// when intensity is 0.
    #[test]
    fn test_ellipse_crop_transparent_outside_with_softness_zero_intensity() {
        use crate::shape::{CropShape, apply_shape_mask};
        use image::{Rgba, RgbaImage};

        // Create a 100x100 white opaque image
        let mut img = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));

        // Apply ellipse mask with softness but vignette_intensity = 0
        apply_shape_mask(
            &mut img,
            &CropShape::Ellipse,
            0.1, // small softness so corner is clearly outside
            0.0, // zero intensity - should still crop!
            crate::color::RgbaColor::opaque(0, 0, 0),
        );

        // Corner pixel (0,0) with small softness should be nearly transparent
        let corner = img.get_pixel(0, 0);
        assert!(
            corner[3] < 50,
            "Corner pixel should be mostly transparent with softness, got alpha={}",
            corner[3]
        );

        // RGB should remain white (255,255,255) since intensity=0 means no color mixing
        assert_eq!(
            (corner[0], corner[1], corner[2]),
            (255, 255, 255),
            "Corner RGB should remain white when intensity=0, got ({},{},{})",
            corner[0], corner[1], corner[2]
        );

        // Center pixel (50,50) is inside the ellipse - must remain opaque
        let center = img.get_pixel(50, 50);
        assert_eq!(
            center[3], 255,
            "Center pixel should remain fully opaque (alpha=255), got alpha={}",
            center[3]
        );
    }

    /// Test that vignette_intensity only affects color mixing, not alpha.
    #[test]
    fn test_vignette_intensity_affects_color_not_alpha() {
        use crate::shape::{CropShape, apply_shape_mask};
        use image::{Rgba, RgbaImage};

        // Test with intensity=0.0
        let mut img_zero = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        apply_shape_mask(
            &mut img_zero,
            &CropShape::Ellipse,
            0.0,
            0.0,
            crate::color::RgbaColor::opaque(255, 0, 0), // red vignette
        );

        // Test with intensity=1.0
        let mut img_full = RgbaImage::from_pixel(100, 100, Rgba([255, 255, 255, 255]));
        apply_shape_mask(
            &mut img_full,
            &CropShape::Ellipse,
            0.0,
            1.0,
            crate::color::RgbaColor::opaque(255, 0, 0), // red vignette
        );

        // Both should have same alpha at corners (transparent)
        let corner_zero = img_zero.get_pixel(0, 0);
        let corner_full = img_full.get_pixel(0, 0);
        assert_eq!(
            corner_zero[3], corner_full[3],
            "Alpha should be same regardless of intensity: zero={}, full={}",
            corner_zero[3], corner_full[3]
        );
        assert_eq!(corner_zero[3], 0, "Corner should be transparent");

        // Both should have same alpha at center (opaque)
        let center_zero = img_zero.get_pixel(50, 50);
        let center_full = img_full.get_pixel(50, 50);
        assert_eq!(center_zero[3], 255, "Center should be opaque");
        assert_eq!(center_full[3], 255, "Center should be opaque");
    }
}
