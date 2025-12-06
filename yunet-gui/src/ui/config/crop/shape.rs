use crate::YuNetApp;
use crate::ui::widgets;
use egui::{ComboBox, Ui};
use yunet_utils::{CropShape, PolygonCornerStyle};

/// Shape controls extracted from edit_shape_controls method.
pub fn edit_shape_controls(app: &mut YuNetApp, ui: &mut Ui) -> bool {
    let mut shape = app.settings.crop.shape.clone();
    let mut changed = false;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Variant {
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

    let mut variant = match &shape {
        CropShape::Rectangle => Variant::Rectangle,
        CropShape::RoundedRectangle { .. } => Variant::RoundedRect,
        CropShape::ChamferedRectangle { .. } => Variant::ChamferRect,
        CropShape::Ellipse => Variant::Ellipse,
        CropShape::Polygon { corner_style, .. } => match corner_style {
            PolygonCornerStyle::Sharp => Variant::PolygonSharp,
            PolygonCornerStyle::Rounded { .. } => Variant::PolygonRounded,
            PolygonCornerStyle::Chamfered { .. } => Variant::PolygonChamfered,
            PolygonCornerStyle::Bezier { .. } => Variant::PolygonBezier,
        },
        CropShape::Star { .. } => Variant::Star,
        CropShape::KochPolygon { .. } => Variant::KochPolygon,
        CropShape::KochRectangle { .. } => Variant::KochRectangle,
    };

    let current_label = match variant {
        Variant::Rectangle => "Rectangle",
        Variant::RoundedRect => "Rounded rectangle",
        Variant::ChamferRect => "Chamfered rectangle",
        Variant::Ellipse => "Ellipse",
        Variant::PolygonSharp => "Polygon",
        Variant::PolygonRounded => "Polygon (rounded)",
        Variant::PolygonChamfered => "Polygon (chamfered)",
        Variant::PolygonBezier => "Polygon (bezier)",
        Variant::Star => "Star",
        Variant::KochPolygon => "Koch Polygon",
        Variant::KochRectangle => "Koch Rectangle",
    };

    let mut variant_changed = false;
    ComboBox::from_label("Shape")
        .selected_text(current_label)
        .show_ui(ui, |ui| {
            let mut select_variant = |label: &str, target: Variant| {
                let selected = variant == target;
                if ui.selectable_label(selected, label).clicked() && !selected {
                    variant = target;
                    variant_changed = true;
                }
            };
            select_variant("Rectangle", Variant::Rectangle);
            select_variant("Rounded rectangle", Variant::RoundedRect);
            select_variant("Chamfered rectangle", Variant::ChamferRect);
            select_variant("Ellipse", Variant::Ellipse);
            select_variant("Polygon", Variant::PolygonSharp);
            select_variant("Polygon (rounded)", Variant::PolygonRounded);
            select_variant("Polygon (chamfered)", Variant::PolygonChamfered);
            select_variant("Polygon (bezier)", Variant::PolygonBezier);
            select_variant("Star", Variant::Star);
            select_variant("Koch Polygon", Variant::KochPolygon);
            select_variant("Koch Rectangle", Variant::KochRectangle);
        });

    if variant_changed {
        shape = match variant {
            Variant::Rectangle => CropShape::Rectangle,
            Variant::RoundedRect => CropShape::RoundedRectangle { radius_pct: 0.12 },
            Variant::ChamferRect => CropShape::ChamferedRectangle { size_pct: 0.12 },
            Variant::Ellipse => CropShape::Ellipse,
            Variant::PolygonSharp => CropShape::Polygon {
                sides: 6,
                rotation_deg: 0.0,
                corner_style: PolygonCornerStyle::Sharp,
            },
            Variant::PolygonRounded => CropShape::Polygon {
                sides: 6,
                rotation_deg: 0.0,
                corner_style: PolygonCornerStyle::Rounded { radius_pct: 0.1 },
            },
            Variant::PolygonChamfered => CropShape::Polygon {
                sides: 6,
                rotation_deg: 0.0,
                corner_style: PolygonCornerStyle::Chamfered { size_pct: 0.1 },
            },
            Variant::PolygonBezier => CropShape::Polygon {
                sides: 6,
                rotation_deg: 0.0,
                corner_style: PolygonCornerStyle::Bezier { tension: 0.5 },
            },
            Variant::Star => CropShape::Star {
                points: 5,
                inner_radius_pct: 0.5,
                rotation_deg: 0.0,
            },
            Variant::KochPolygon => CropShape::KochPolygon {
                sides: 3,
                rotation_deg: 0.0,
                iterations: 3,
            },
            Variant::KochRectangle => CropShape::KochRectangle { iterations: 3 },
        };
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
                    *radius_pct = (radius / 100.0).clamp(0.0, 0.5);
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
                    *size_pct = (size / 100.0).clamp(0.0, 0.5);
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
                *sides = sides_u32.clamp(3, 24) as u8;
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
                    let angle = std::f32::consts::PI / sides_u32 as f32;
                    let max_radius_pct = 0.5 * angle.cos();
                    let max_radius_display = max_radius_pct * 100.0;

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
                            *radius_pct = (radius / 100.0).clamp(0.0, max_radius_pct);
                            changed = true;
                        }
                    );
                }
                PolygonCornerStyle::Chamfered { size_pct } => {
                    let angle = std::f32::consts::PI / sides_u32 as f32;
                    let max_size_pct = 0.5 * angle.sin();
                    let max_size_display = max_size_pct * 100.0;

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
                            *size_pct = (size / 100.0).clamp(0.0, max_size_pct);
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
                *points = points_u32.clamp(3, 24) as u8;
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
                    *inner_radius_pct = (inner / 100.0).clamp(0.1, 0.9);
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
                *sides = sides_u32.clamp(3, 24) as u8;
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
                *iterations = iter.clamp(0, 5) as u8;
                changed = true;
            }
        }
        CropShape::KochRectangle { iterations } => {
            let mut iter = *iterations as u32;
            if widgets::integer_input(ui, &mut iter, 0..=5, 60.0, None)
                .on_hover_text("Iterations")
                .changed()
            {
                *iterations = iter.clamp(0, 5) as u8;
                changed = true;
            }
        }
        CropShape::Rectangle | CropShape::Ellipse => {}
    }

    let sanitized = shape.sanitized();
    if sanitized != app.settings.crop.shape {
        app.settings.crop.shape = sanitized;
        changed = true;
    }

    changed
}
