use crate::YuNetApp;
use crate::ui::widgets;
use egui::{ComboBox, Ui};
use fcs_utils::{CropShape, PolygonCornerStyle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShapeVariant {
    Rectangle,
    RoundedRect,
    ChamferRect,
    Ellipse,
    PolygonSharp,
    PolygonRounded,
    PolygonChamfered,
    PolygonBezier,
    Star,
    KochPolygon,
    KochRectangle,
}

fn shape_variant(shape: &CropShape) -> ShapeVariant {
    match shape {
        CropShape::Rectangle => ShapeVariant::Rectangle,
        CropShape::RoundedRectangle { .. } => ShapeVariant::RoundedRect,
        CropShape::ChamferedRectangle { .. } => ShapeVariant::ChamferRect,
        CropShape::Ellipse => ShapeVariant::Ellipse,
        CropShape::Polygon { corner_style, .. } => match corner_style {
            PolygonCornerStyle::Sharp => ShapeVariant::PolygonSharp,
            PolygonCornerStyle::Rounded { .. } => ShapeVariant::PolygonRounded,
            PolygonCornerStyle::Chamfered { .. } => ShapeVariant::PolygonChamfered,
            PolygonCornerStyle::Bezier { .. } => ShapeVariant::PolygonBezier,
        },
        CropShape::Star { .. } => ShapeVariant::Star,
        CropShape::KochPolygon { .. } => ShapeVariant::KochPolygon,
        CropShape::KochRectangle { .. } => ShapeVariant::KochRectangle,
    }
}

fn shape_variant_label(variant: ShapeVariant) -> &'static str {
    match variant {
        ShapeVariant::Rectangle => "Rectangle",
        ShapeVariant::RoundedRect => "Rounded rectangle",
        ShapeVariant::ChamferRect => "Chamfered rectangle",
        ShapeVariant::Ellipse => "Ellipse",
        ShapeVariant::PolygonSharp => "Polygon",
        ShapeVariant::PolygonRounded => "Polygon (rounded)",
        ShapeVariant::PolygonChamfered => "Polygon (chamfered)",
        ShapeVariant::PolygonBezier => "Polygon (bezier)",
        ShapeVariant::Star => "Star",
        ShapeVariant::KochPolygon => "Koch Polygon",
        ShapeVariant::KochRectangle => "Koch Rectangle",
    }
}

fn default_shape_for_variant(variant: ShapeVariant) -> CropShape {
    match variant {
        ShapeVariant::Rectangle => CropShape::Rectangle,
        ShapeVariant::RoundedRect => CropShape::RoundedRectangle { radius_pct: 0.12 },
        ShapeVariant::ChamferRect => CropShape::ChamferedRectangle { size_pct: 0.12 },
        ShapeVariant::Ellipse => CropShape::Ellipse,
        ShapeVariant::PolygonSharp => CropShape::Polygon {
            sides: 6,
            rotation_deg: 0.0,
            corner_style: PolygonCornerStyle::Sharp,
        },
        ShapeVariant::PolygonRounded => CropShape::Polygon {
            sides: 6,
            rotation_deg: 0.0,
            corner_style: PolygonCornerStyle::Rounded { radius_pct: 0.1 },
        },
        ShapeVariant::PolygonChamfered => CropShape::Polygon {
            sides: 6,
            rotation_deg: 0.0,
            corner_style: PolygonCornerStyle::Chamfered { size_pct: 0.1 },
        },
        ShapeVariant::PolygonBezier => CropShape::Polygon {
            sides: 6,
            rotation_deg: 0.0,
            corner_style: PolygonCornerStyle::Bezier { tension: 0.5 },
        },
        ShapeVariant::Star => CropShape::Star {
            points: 5,
            inner_radius_pct: 0.5,
            rotation_deg: 0.0,
        },
        ShapeVariant::KochPolygon => CropShape::KochPolygon {
            sides: 3,
            rotation_deg: 0.0,
            iterations: 3,
        },
        ShapeVariant::KochRectangle => CropShape::KochRectangle { iterations: 3 },
    }
}

fn clamp_rounded_rect_radius_pct(radius_pct: f32) -> f32 {
    radius_pct.clamp(0.0, 0.5)
}

fn clamp_chamfered_rect_size_pct(size_pct: f32) -> f32 {
    size_pct.clamp(0.0, 0.5)
}

fn clamp_polygon_sides(value: u32) -> u8 {
    value.clamp(3, 24) as u8
}

fn polygon_corner_radius_max_pct(sides: u32) -> f32 {
    0.5 * (std::f32::consts::PI / sides as f32).cos()
}

fn polygon_corner_radius_display_max(sides: u32) -> f32 {
    polygon_corner_radius_max_pct(sides) * 100.0
}

fn clamp_polygon_corner_radius_pct(radius_pct: f32, sides: u32) -> f32 {
    radius_pct.clamp(0.0, polygon_corner_radius_max_pct(sides))
}

fn polygon_chamfer_size_max_pct(sides: u32) -> f32 {
    0.5 * (std::f32::consts::PI / sides as f32).sin()
}

fn polygon_chamfer_size_display_max(sides: u32) -> f32 {
    polygon_chamfer_size_max_pct(sides) * 100.0
}

fn clamp_polygon_chamfer_size_pct(size_pct: f32, sides: u32) -> f32 {
    size_pct.clamp(0.0, polygon_chamfer_size_max_pct(sides))
}

fn clamp_star_points(value: u32) -> u8 {
    value.clamp(3, 24) as u8
}

fn clamp_star_inner_radius_pct(inner_radius_pct: f32) -> f32 {
    inner_radius_pct.clamp(0.0, 1.0)
}

fn clamp_koch_iterations(value: u32) -> u8 {
    value.clamp(0, 5) as u8
}

/// Shape controls extracted from edit_shape_controls method.
pub fn edit_shape_controls(app: &mut YuNetApp, ui: &mut Ui) -> bool {
    let mut shape = app.settings.crop.shape.clone();
    let mut changed = false;
    let mut variant = shape_variant(&shape);

    let mut variant_changed = false;
    ComboBox::from_label("Shape")
        .selected_text(shape_variant_label(variant))
        .show_ui(ui, |ui| {
            let mut select_variant = |label: &str, target: ShapeVariant| {
                let selected = variant == target;
                if ui.selectable_label(selected, label).clicked() && !selected {
                    variant = target;
                    variant_changed = true;
                }
            };
            select_variant("Rectangle", ShapeVariant::Rectangle);
            select_variant("Rounded rectangle", ShapeVariant::RoundedRect);
            select_variant("Chamfered rectangle", ShapeVariant::ChamferRect);
            select_variant("Ellipse", ShapeVariant::Ellipse);
            select_variant("Polygon", ShapeVariant::PolygonSharp);
            select_variant("Polygon (rounded)", ShapeVariant::PolygonRounded);
            select_variant("Polygon (chamfered)", ShapeVariant::PolygonChamfered);
            select_variant("Polygon (bezier)", ShapeVariant::PolygonBezier);
            select_variant("Star", ShapeVariant::Star);
            select_variant("Koch Polygon", ShapeVariant::KochPolygon);
            select_variant("Koch Rectangle", ShapeVariant::KochRectangle);
        });

    if variant_changed {
        shape = default_shape_for_variant(variant);
        changed = true;
    }

    match &mut shape {
        CropShape::RoundedRectangle { radius_pct } => {
            let mut radius = (*radius_pct * 100.0).clamp(0.0, 50.0);
            crate::constrained_slider_row!(
                ui,
                &mut radius,
                0.0..=50.0,
                "Corner radius (%)",
                1.0,
                None,
                None,
                {
                    *radius_pct = clamp_rounded_rect_radius_pct(radius / 100.0);
                    changed = true;
                }
            );
        }
        CropShape::ChamferedRectangle { size_pct } => {
            let mut size = (*size_pct * 100.0).clamp(0.0, 50.0);
            crate::constrained_slider_row!(
                ui,
                &mut size,
                0.0..=50.0,
                "Chamfer size (%)",
                1.0,
                None,
                None,
                {
                    *size_pct = clamp_chamfered_rect_size_pct(size / 100.0);
                    changed = true;
                }
            );
        }
        CropShape::Polygon {
            sides,
            rotation_deg,
            corner_style,
        } => {
            let mut sides_u32 = *sides as u32;
            if widgets::integer_input(ui, &mut sides_u32, 3..=24, 60.0, None)
                .on_hover_text("Number of sides")
                .changed()
            {
                *sides = clamp_polygon_sides(sides_u32);
                changed = true;
            }
            crate::constrained_slider_row!(
                ui,
                rotation_deg,
                -180.0..=180.0,
                "Rotation (°)",
                1.0,
                None,
                None,
                {
                    changed = true;
                }
            );

            match corner_style {
                PolygonCornerStyle::Sharp => {}
                PolygonCornerStyle::Rounded { radius_pct } => {
                    let max_radius_display = polygon_corner_radius_display_max(sides_u32);

                    let mut radius = (*radius_pct * 100.0).clamp(0.0, max_radius_display);
                    crate::constrained_slider_row!(
                        ui,
                        &mut radius,
                        0.0..=max_radius_display,
                        "Corner radius (%)",
                        0.1,
                        None,
                        None,
                        {
                            *radius_pct =
                                clamp_polygon_corner_radius_pct(radius / 100.0, sides_u32);
                            changed = true;
                        }
                    );
                }
                PolygonCornerStyle::Chamfered { size_pct } => {
                    let max_size_display = polygon_chamfer_size_display_max(sides_u32);

                    let mut size = (*size_pct * 100.0).clamp(0.0, max_size_display);
                    crate::constrained_slider_row!(
                        ui,
                        &mut size,
                        0.0..=max_size_display,
                        "Chamfer size (%)",
                        0.1,
                        None,
                        None,
                        {
                            *size_pct = clamp_polygon_chamfer_size_pct(size / 100.0, sides_u32);
                            changed = true;
                        }
                    );
                }
                PolygonCornerStyle::Bezier { tension } => {
                    crate::constrained_slider_row!(
                        ui,
                        tension,
                        0.0..=2.0,
                        "Tension",
                        0.01,
                        Some(
                            "Adjusts the curvature of the corners. 0 is sharp, higher values are smoother."
                        ),
                        None,
                        {
                            changed = true;
                        }
                    );
                }
            }
        }
        CropShape::Star {
            points,
            inner_radius_pct,
            rotation_deg,
        } => {
            let mut points_u32 = *points as u32;
            if widgets::integer_input(ui, &mut points_u32, 3..=24, 60.0, None)
                .on_hover_text("Number of points")
                .changed()
            {
                *points = clamp_star_points(points_u32);
                changed = true;
            }

            let mut inner = (*inner_radius_pct * 100.0).clamp(10.0, 90.0);
            crate::constrained_slider_row!(
                ui,
                &mut inner,
                10.0..=90.0,
                "Inner radius (%)",
                1.0,
                None,
                None,
                {
                    *inner_radius_pct = clamp_star_inner_radius_pct(inner / 100.0);
                    changed = true;
                }
            );

            crate::constrained_slider_row!(
                ui,
                rotation_deg,
                -180.0..=180.0,
                "Rotation (°)",
                1.0,
                None,
                None,
                {
                    changed = true;
                }
            );
        }
        CropShape::KochPolygon {
            sides,
            rotation_deg,
            iterations,
        } => {
            let mut sides_u32 = *sides as u32;
            if widgets::integer_input(ui, &mut sides_u32, 3..=24, 60.0, None)
                .on_hover_text("Number of sides")
                .changed()
            {
                *sides = clamp_polygon_sides(sides_u32);
                changed = true;
            }
            crate::constrained_slider_row!(
                ui,
                rotation_deg,
                -180.0..=180.0,
                "Rotation (°)",
                1.0,
                None,
                None,
                {
                    changed = true;
                }
            );
            let mut iter = *iterations as u32;
            if widgets::integer_input(ui, &mut iter, 0..=5, 60.0, None)
                .on_hover_text("Iterations")
                .changed()
            {
                *iterations = clamp_koch_iterations(iter);
                changed = true;
            }
        }
        CropShape::KochRectangle { iterations } => {
            let mut iter = *iterations as u32;
            if widgets::integer_input(ui, &mut iter, 0..=5, 60.0, None)
                .on_hover_text("Iterations")
                .changed()
            {
                *iterations = clamp_koch_iterations(iter);
                changed = true;
            }
        }
        CropShape::Rectangle | CropShape::Ellipse => {}
    }

    ui.add_space(4.0);
    let mut softness = app.settings.crop.vignette_softness * 100.0;
    crate::constrained_slider_row!(
        ui,
        &mut softness,
        0.0..=100.0,
        "Vignette softness (%)",
        1.0,
        Some("Feathers the edges of the crop shape."),
        None,
        {
            app.settings.crop.vignette_softness = (softness / 100.0).clamp(0.0, 1.0);
            changed = true;
        }
    );

    let mut intensity = app.settings.crop.vignette_intensity * 100.0;
    crate::constrained_slider_row!(
        ui,
        &mut intensity,
        0.0..=100.0,
        "Vignette intensity (%)",
        1.0,
        Some("Controls the maximum opacity of the vignette."),
        None,
        {
            app.settings.crop.vignette_intensity = (intensity / 100.0).clamp(0.0, 1.0);
            changed = true;
        }
    );

    ui.horizontal(|ui| {
        ui.label("Vignette color");
        let mut color = [
            app.settings.crop.vignette_color.red,
            app.settings.crop.vignette_color.green,
            app.settings.crop.vignette_color.blue,
            app.settings.crop.vignette_color.alpha,
        ];
        if ui
            .color_edit_button_srgba_unmultiplied(&mut color)
            .changed()
        {
            app.settings.crop.vignette_color = fcs_utils::RgbaColor {
                red: color[0],
                green: color[1],
                blue: color[2],
                alpha: color[3],
            };
            changed = true;
        }
    })
    .response
    .on_hover_text("Choose the color of the vignette effect.");

    let sanitized = shape.sanitized();
    if sanitized != app.settings.crop.shape {
        app.settings.crop.shape = sanitized;
        changed = true;
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_variant_round_trips_through_default_shape() {
        let variants = [
            ShapeVariant::Rectangle,
            ShapeVariant::RoundedRect,
            ShapeVariant::ChamferRect,
            ShapeVariant::Ellipse,
            ShapeVariant::PolygonSharp,
            ShapeVariant::PolygonRounded,
            ShapeVariant::PolygonChamfered,
            ShapeVariant::PolygonBezier,
            ShapeVariant::Star,
            ShapeVariant::KochPolygon,
            ShapeVariant::KochRectangle,
        ];

        for variant in variants {
            assert_eq!(shape_variant(&default_shape_for_variant(variant)), variant);
            assert!(!shape_variant_label(variant).is_empty());
        }
    }

    #[test]
    fn clamp_helpers_enforce_expected_ranges() {
        assert_eq!(clamp_rounded_rect_radius_pct(0.75), 0.5);
        assert_eq!(clamp_chamfered_rect_size_pct(-0.25), 0.0);
        assert_eq!(clamp_polygon_sides(2), 3);
        assert_eq!(clamp_polygon_sides(99), 24);
        assert_eq!(clamp_star_points(1), 3);
        assert_eq!(clamp_star_inner_radius_pct(1.5), 1.0);
        assert_eq!(clamp_koch_iterations(99), 5);
    }

    #[test]
    fn polygon_corner_helpers_cap_values_by_side_count() {
        let sides = 6;

        assert!(polygon_corner_radius_display_max(sides) > 0.0);
        assert!(polygon_chamfer_size_display_max(sides) > 0.0);
        assert_eq!(
            clamp_polygon_corner_radius_pct(10.0, sides),
            polygon_corner_radius_max_pct(sides)
        );
        assert_eq!(
            clamp_polygon_chamfer_size_pct(10.0, sides),
            polygon_chamfer_size_max_pct(sides)
        );
    }

    #[test]
    fn polygon_corner_radius_max_increases_with_more_sides() {
        // formula: 0.5 * cos(π/sides) — cos(π/sides) grows toward cos(0)=1 as sides → ∞
        let r3 = polygon_corner_radius_max_pct(3);
        let r6 = polygon_corner_radius_max_pct(6);
        let r12 = polygon_corner_radius_max_pct(12);
        assert!(r3 < r6, "hexagon should allow larger radius than triangle");
        assert!(r6 < r12, "12-gon should allow larger radius than hexagon");
    }

    #[test]
    fn polygon_chamfer_size_max_increases_with_more_sides() {
        // formula: 0.5 * sin(π/sides) — sin(π/sides) → 0 as sides → ∞, so it DECREASES
        let c3 = polygon_chamfer_size_max_pct(3);
        let c6 = polygon_chamfer_size_max_pct(6);
        let c12 = polygon_chamfer_size_max_pct(12);
        assert!(c3 > c6, "triangle should allow larger chamfer than hexagon");
        assert!(c6 > c12, "hexagon should allow larger chamfer than 12-gon");
    }

    #[test]
    fn polygon_display_max_is_max_pct_scaled_by_100() {
        for sides in [3, 4, 5, 6, 8, 12] {
            let expected_radius = polygon_corner_radius_max_pct(sides) * 100.0;
            let expected_chamfer = polygon_chamfer_size_max_pct(sides) * 100.0;
            assert!(
                (polygon_corner_radius_display_max(sides) - expected_radius).abs() < 1e-5,
                "display_max mismatch for radius at sides={sides}"
            );
            assert!(
                (polygon_chamfer_size_display_max(sides) - expected_chamfer).abs() < 1e-5,
                "display_max mismatch for chamfer at sides={sides}"
            );
        }
    }
}
