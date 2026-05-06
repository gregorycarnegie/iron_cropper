//! Windows-11-style title bar.

use crate::theme::P;
use crate::types::App2;
use egui::{Color32, Frame, Sense, Stroke, Ui, Vec2};

pub fn show(ui: &mut Ui, app: &mut App2) {
    egui::Panel::top("titlebar")
        .exact_size(36.0)
        .frame(Frame::new()
            .fill(P::BG)
            .inner_margin(egui::Margin::ZERO))
        .show_inside(ui, |ui| {
            ui.painter().line_segment(
                [egui::pos2(ui.min_rect().min.x, ui.min_rect().max.y - 1.0),
                 egui::pos2(ui.min_rect().max.x, ui.min_rect().max.y - 1.0)],
                Stroke::new(1.0, P::RULE),
            );

            ui.horizontal_centered(|ui| {
                ui.set_height(36.0);

                // Left: logo + name
                ui.add_space(12.0);
                draw_logo(ui);
                ui.add_space(8.0);
                ui.label(egui::RichText::new("Face Crop Studio").size(12.5).color(P::INK).strong());
                ui.add_space(6.0);
                ui.label(egui::RichText::new("—").size(12.5).color(P::RULE2));
                ui.add_space(6.0);

                // Current file
                if let Some(path) = &app.preview.image_path.clone() {
                    let name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");
                    ui.label(egui::RichText::new(name).size(11.5).color(P::INK3).family(egui::FontFamily::Monospace));
                    if app.is_busy {
                        ui.label(egui::RichText::new(" ●").size(11.5).color(P::PEACH));
                    }
                } else {
                    ui.label(egui::RichText::new("No file").size(11.5).color(P::INK3).family(egui::FontFamily::Monospace));
                }

                // Right: Windows chrome buttons
                let _full_w = ui.max_rect().max.x;
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    win_btn(ui, "✕", true);
                    win_btn(ui, "□", false);
                    win_btn(ui, "─", false);
                });
            });
        });
}

fn draw_logo(ui: &mut egui::Ui) {
    let (resp, painter) = ui.allocate_painter(Vec2::splat(18.0), Sense::hover());
    let c = resp.rect.center();
    let r = 9.0;
    // Outer ring with gradient effect — approximate with arcs
    painter.circle_filled(c, r, P::PEACH);
    painter.circle_filled(c, r * 0.8, P::ROSE);
    painter.circle_filled(c, r * 0.55, P::BG);
    painter.circle_filled(c, r * 0.28, P::PEACH);
}

fn win_btn(ui: &mut egui::Ui, label: &str, is_close: bool) {
    let (resp, painter) = ui.allocate_painter(Vec2::new(46.0, 36.0), Sense::click());
    let bg = if resp.hovered() {
        if is_close { Color32::from_rgb(0xe8, 0x11, 0x23) } else { P::white_alpha(15) }
    } else {
        Color32::TRANSPARENT
    };
    if bg != Color32::TRANSPARENT {
        painter.rect_filled(resp.rect, 0.0, bg);
    }
    let text_color = if resp.hovered() && is_close { Color32::WHITE } else { P::INK2 };
    painter.text(
        resp.rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(11.0),
        text_color,
    );
}
