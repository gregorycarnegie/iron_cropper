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
}
