//! Centre canvas column — image viewport + face overlays + mini-log.

use crate::rendering::paint::{
    draw_confidence_badge, draw_drag_handle, draw_face_box, draw_landmark_dot,
};
use crate::theme::P;
use crate::types::{App2, LogKind};
use crate::ui::widgets::{ctl_pill, face_chip};
use egui::{Color32, Frame, Sense, Stroke, Ui, Vec2};
use egui::epaint::{Mesh, Vertex};
use fcs_core::calculate_crop_region;
use fcs_utils::outline_points_for_rect;

pub fn show(ui: &mut Ui, app: &mut App2) {
    ui.set_min_size(ui.available_size());

    // Canvas header
    canvas_header(ui, app);

    // Stage (image + overlays) — takes remaining space minus the bottom bar
    let avail = ui.available_size();
    let bottom_bar_h = 44.0;
    let _stage_h = (avail.y - bottom_bar_h).max(100.0);

    egui::Panel::bottom("canvas_bottom_bar")
        .exact_size(bottom_bar_h)
        .show_separator_line(false)
        .frame(Frame::new().fill(P::black_alpha(50)))
        .show_inside(ui, |ui| {
            canvas_bottom_bar(ui, app);
        });

    egui::CentralPanel::default()
        .frame(Frame::new().fill(P::BG1))
        .show_inside(ui, |ui| {
            stage(ui, app);
        });
}

fn canvas_header(ui: &mut Ui, app: &mut App2) {
    let h = 42.0;
    let (resp, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), h), Sense::hover());
    let r = resp.rect;
    painter.rect_filled(r, 0.0, P::black_alpha(38));
    painter.line_segment(
        [egui::pos2(r.min.x, r.max.y), egui::pos2(r.max.x, r.max.y)],
        Stroke::new(1.0, P::RULE),
    );

    // Control chips on the right
    let chips_x = r.max.x - 360.0;

    // Filename + dimensions — clipped so they never overlap with the pills
    let name = app
        .preview
        .image_path
        .as_deref()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("No image");
    let dims = app
        .preview
        .image_size
        .map(|(w, h)| format!("{w} × {h}"))
        .unwrap_or_default();

    let text_clip = egui::Rect::from_min_max(
        egui::pos2(r.min.x, r.min.y),
        egui::pos2(chips_x - 10.0, r.max.y),
    );
    let left_painter = painter.with_clip_rect(text_clip);

    left_painter.text(
        egui::pos2(r.min.x + 14.0, r.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        egui::FontId::proportional(13.0),
        P::INK,
    );
    if !dims.is_empty() {
        let name_w = left_painter
            .layout_no_wrap(name.to_string(), egui::FontId::proportional(13.0), P::INK)
            .size()
            .x;
        left_painter.text(
            egui::pos2(r.min.x + 14.0 + name_w + 12.0, r.center().y),
            egui::Align2::LEFT_CENTER,
            &dims,
            egui::FontId::monospace(11.0),
            P::INK3,
        );
    }
    let clip = egui::Rect::from_min_max(egui::pos2(chips_x, r.min.y), r.max);
    // We draw them inline via a separate child ui
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(clip)
            .layout(egui::Layout::right_to_left(egui::Align::Center)),
    );
    child.add_space(12.0);
    let conf = app.settings.detection.score_threshold;
    let preset = app.settings.crop.preset.clone();
    let face_h = app.settings.crop.face_height_pct;
    let aspect = format!(
        "{}:{}",
        app.settings.crop.output_width, app.settings.crop.output_height
    );

    ctl_pill(&mut child, "preset ", &preset, Some(P::CYAN));
    child.add_space(4.0);
    ctl_pill(&mut child, "conf ", &format!("{conf:.2}"), Some(P::PEACH));
    child.add_space(4.0);
    ctl_pill(&mut child, "aspect ", &aspect, None);
    child.add_space(4.0);
    ctl_pill(&mut child, "face-h ", &format!("{:.0}%", face_h), None);
}

/// Maps a normalized image coordinate (0..1) to screen space, accounting for canvas rotation.
fn norm_to_screen_rotated(nx: f32, ny: f32, rect: egui::Rect, rotation: u32) -> egui::Pos2 {
    let (rx, ry) = match rotation {
        90  => (1.0 - ny, nx),
        180 => (1.0 - nx, 1.0 - ny),
        270 => (ny, 1.0 - nx),
        _   => (nx, ny),
    };
    egui::pos2(rect.min.x + rx * rect.width(), rect.min.y + ry * rect.height())
}

/// Converts a BoundingBox in image pixel coords to an axis-aligned screen rect, accounting for rotation.
fn rotated_bbox_screen_rect(
    bx: f32, by: f32, bw: f32, bh: f32,
    iw: f32, ih: f32,
    rect: egui::Rect,
    rotation: u32,
) -> egui::Rect {
    let corners = [
        (bx / iw,        by / ih),
        ((bx + bw) / iw, by / ih),
        ((bx + bw) / iw, (by + bh) / ih),
        (bx / iw,        (by + bh) / ih),
    ];
    let pts: Vec<egui::Pos2> = corners
        .iter()
        .map(|(nx, ny)| norm_to_screen_rotated(*nx, *ny, rect, rotation))
        .collect();
    let min_x = pts.iter().map(|p| p.x).fold(f32::INFINITY,     f32::min);
    let min_y = pts.iter().map(|p| p.y).fold(f32::INFINITY,     f32::min);
    let max_x = pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
    let max_y = pts.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
    egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y))
}

/// Builds a textured mesh quad that renders `texture_id` into `dest` with the given rotation.
fn image_shape(texture_id: egui::TextureId, dest: egui::Rect, rotation: u32) -> egui::Shape {
    // Screen-space quad corners: TL, TR, BR, BL
    let pos = [
        dest.min,
        egui::pos2(dest.max.x, dest.min.y),
        dest.max,
        egui::pos2(dest.min.x, dest.max.y),
    ];
    // UV corners — what original UV does each display corner sample?
    // (derived from standard 90°/180°/270° CW rotation math)
    let uvs: [[f32; 2]; 4] = match rotation {
        90  => [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        180 => [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
        270 => [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        _   => [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    };
    let mut mesh = Mesh::with_texture(texture_id);
    for (p, uv) in pos.iter().zip(uvs.iter()) {
        mesh.vertices.push(Vertex {
            pos: *p,
            uv: egui::pos2(uv[0], uv[1]),
            color: Color32::WHITE,
        });
    }
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    egui::Shape::mesh(mesh)
}

fn stage(ui: &mut Ui, app: &mut App2) {
    let avail = ui.available_rect_before_wrap();
    let pad = 18.0;
    let stage_outer = avail.shrink(pad);

    // Scroll-to-zoom (read before allocating sense, so it applies to the whole stage area)
    let scroll = ui.ctx().input(|i| i.smooth_scroll_delta.y);
    if scroll.abs() > 0.1 {
        let factor = (1.0 + scroll * 0.003).clamp(0.85, 1.20_f32);
        app.zoom = (app.zoom * factor).clamp(0.25, 8.0);
    }

    // Compute the "fit" rect (zoom = 1, pan = 0), accounting for rotation's effect on aspect ratio
    let fit_rect = if let Some((iw, ih)) = app.preview.image_size {
        // For 90°/270° the displayed aspect ratio is the inverse of the image's
        let (disp_w, disp_h) = match app.canvas_rotation {
            90 | 270 => (ih as f32, iw as f32),
            _        => (iw as f32, ih as f32),
        };
        let img_ar  = disp_w / disp_h;
        let slot_ar = stage_outer.width() / stage_outer.height();
        if img_ar > slot_ar {
            let w = stage_outer.width();
            let h = w / img_ar;
            egui::Rect::from_center_size(stage_outer.center(), Vec2::new(w, h))
        } else {
            let h = stage_outer.height();
            let w = h * img_ar;
            egui::Rect::from_center_size(stage_outer.center(), Vec2::new(w, h))
        }
    } else {
        stage_outer
    };

    // Apply zoom and pan relative to the fit rect center
    let image_rect = egui::Rect::from_center_size(
        fit_rect.center() + app.pan,
        fit_rect.size() * app.zoom,
    );

    // Stage background + border drawn at the fit rect (the "frame")
    let painter = ui.painter();
    painter.rect_filled(fit_rect, 12.0, P::BG);
    painter.rect_stroke(
        fit_rect,
        12.0,
        Stroke::new(1.0, P::RULE),
        egui::StrokeKind::Outside,
    );

    // Clip all image/overlay painting to the fit rect so zoom overflow is hidden
    let painter = painter.with_clip_rect(fit_rect);

    if let Some(texture) = &app.preview.texture {
        painter.add(image_shape(texture.id(), image_rect, app.canvas_rotation));
    } else if app.is_busy {
        painter.text(
            image_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Detecting faces…",
            egui::FontId::proportional(14.0),
            P::INK3,
        );
    } else {
        painter.text(
            image_rect.center() - Vec2::new(0.0, 12.0),
            egui::Align2::CENTER_CENTER,
            "Drop an image to begin",
            egui::FontId::proportional(14.0),
            P::INK3,
        );
        painter.text(
            image_rect.center() + Vec2::new(0.0, 12.0),
            egui::Align2::CENTER_CENTER,
            "or click Detect faces →",
            egui::FontId::monospace(11.0),
            P::INK3,
        );
    }

    // Face detection overlays
    if let Some((img_w, img_h)) = app.preview.image_size {
        let iw = img_w as f32;
        let ih = img_h as f32;
        let rot = app.canvas_rotation;
        let dets: Vec<_> = app.preview.detections.iter().enumerate().collect();

        for (i, det) in &dets {
            let bbox = det.active_bbox();
            let screen_rect = rotated_bbox_screen_rect(
                bbox.x, bbox.y, bbox.width, bbox.height, iw, ih, image_rect, rot,
            );

            let selected = app.selected_faces.contains(i);
            let color = if !selected {
                P::INK3
            } else if i % 2 == 0 {
                P::PEACH
            } else {
                P::CYAN
            };

            draw_face_box(&painter, screen_rect, color, selected);

            draw_confidence_badge(
                &painter,
                &format!("{:.2}", det.detection.score),
                screen_rect,
                color,
            );

            for lm in &det.detection.landmarks {
                let lm_pos = norm_to_screen_rotated(lm.x / iw, lm.y / ih, image_rect, rot);
                draw_landmark_dot(&painter, lm_pos);
            }

            if selected {
                let corners = [
                    screen_rect.min,
                    egui::pos2(screen_rect.max.x, screen_rect.min.y),
                    egui::pos2(screen_rect.min.x, screen_rect.max.y),
                    screen_rect.max,
                ];
                for corner in corners {
                    draw_drag_handle(&painter, corner, color);
                }
            }
        }
    }

    // Crop region + shape outline overlay
    if app.show_crop_overlay
        && let Some((img_w, img_h)) = app.preview.image_size
    {
        let crop_settings = app.build_crop_settings();
        let crop_stroke = Stroke::new(2.0, P::LIME);
        let iw = img_w as f32;
        let ih = img_h as f32;
        let rot = app.canvas_rotation;
        for det in &app.preview.detections {
            let bbox = det.active_bbox();
            let region = calculate_crop_region(img_w, img_h, bbox, &crop_settings);
            let rx = region.x as f32;
            let ry = region.y as f32;
            let rw = region.width as f32;
            let rh = region.height as f32;

            if rot == 0 {
                // Full shape-outline support at 0°
                let sx = image_rect.min.x + (rx / iw) * image_rect.width();
                let sy = image_rect.min.y + (ry / ih) * image_rect.height();
                let sw = (rw / iw) * image_rect.width();
                let sh = (rh / ih) * image_rect.height();
                let crop_rect = egui::Rect::from_min_size(egui::pos2(sx, sy), Vec2::new(sw, sh));
                let shape_pts = outline_points_for_rect(sw, sh, &app.settings.crop.shape);
                if shape_pts.len() >= 2 {
                    let outline: Vec<egui::Pos2> = shape_pts
                        .iter()
                        .map(|(x, y)| egui::pos2(crop_rect.min.x + x, crop_rect.min.y + y))
                        .collect();
                    painter.add(egui::Shape::closed_line(outline, crop_stroke));
                } else {
                    painter.rect_stroke(crop_rect, 4.0, crop_stroke, egui::StrokeKind::Inside);
                }
            } else {
                // For rotated views draw a simple rect outline via the 4 rotated corners
                let screen_rect = rotated_bbox_screen_rect(rx, ry, rw, rh, iw, ih, image_rect, rot);
                painter.rect_stroke(screen_rect, 4.0, crop_stroke, egui::StrokeKind::Inside);
            }
        }
    }

    // Allocate the visible stage area for click (face selection) and drag (pan)
    let resp = ui.allocate_rect(fit_rect, Sense::click_and_drag());

    if resp.dragged() {
        app.pan += resp.drag_delta();
    }

    if resp.clicked()
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some((img_w, img_h)) = app.preview.image_size
    {
        let iw = img_w as f32;
        let ih = img_h as f32;
        let rot = app.canvas_rotation;
        let mut clicked_any = false;
        for (i, det) in app.preview.detections.iter().enumerate() {
            let bbox = det.active_bbox();
            let sr = rotated_bbox_screen_rect(
                bbox.x, bbox.y, bbox.width, bbox.height, iw, ih, image_rect, rot,
            );
            if sr.expand(4.0).contains(pos) {
                if app.selected_faces.contains(&i) {
                    app.selected_faces.remove(&i);
                } else {
                    app.selected_faces.insert(i);
                }
                clicked_any = true;
                break;
            }
        }
        if !clicked_any {
            app.selected_faces.clear();
        }
    }

    // Mini-log overlay
    mini_log_overlay(ui, app, fit_rect);
}

fn mini_log_overlay(ui: &mut Ui, app: &App2, image_rect: egui::Rect) {
    let log_w = 300.0;
    let log_max_lines = 5;
    let line_h = 16.0;
    let pad = 10.0;
    let header_h = 24.0;
    let n = app.log_lines.len().min(log_max_lines);
    let log_h = header_h + n as f32 * line_h + pad * 2.0;

    let log_rect = egui::Rect::from_min_size(
        egui::pos2(image_rect.min.x + 18.0, image_rect.max.y - log_h - 18.0),
        Vec2::new(log_w, log_h),
    );
    let painter = ui.painter();
    painter.rect_filled(log_rect, 9.0, P::SURFACE.linear_multiply(0.85));
    painter.rect_stroke(
        log_rect,
        9.0,
        Stroke::new(1.0, P::RULE),
        egui::StrokeKind::Outside,
    );

    painter.text(
        egui::pos2(log_rect.min.x + pad, log_rect.min.y + 8.0),
        egui::Align2::LEFT_TOP,
        "Pipeline log",
        egui::FontId::proportional(11.0),
        P::INK,
    );
    painter.text(
        egui::pos2(log_rect.max.x - pad, log_rect.min.y + 8.0),
        egui::Align2::RIGHT_TOP,
        "single.run",
        egui::FontId::monospace(9.5),
        P::CYAN,
    );

    let msg_x = log_rect.min.x + 62.0;
    let msg_clip = egui::Rect::from_min_max(
        egui::pos2(msg_x, log_rect.min.y),
        egui::pos2(log_rect.max.x - pad, log_rect.max.y),
    );
    let msg_painter = painter.with_clip_rect(msg_clip);

    let start = app.log_lines.len().saturating_sub(log_max_lines);
    for (i, line) in app.log_lines[start..].iter().enumerate() {
        let y = log_rect.min.y + header_h + i as f32 * line_h;
        let msg_color = match line.kind {
            LogKind::Ok => P::LIME,
            LogKind::Warn => P::PEACH,
            LogKind::Info => P::INK2,
        };
        painter.text(
            egui::pos2(log_rect.min.x + pad, y),
            egui::Align2::LEFT_TOP,
            &line.timestamp,
            egui::FontId::monospace(9.5),
            P::INK3,
        );
        msg_painter.text(
            egui::pos2(msg_x, y),
            egui::Align2::LEFT_TOP,
            &line.message,
            egui::FontId::monospace(9.5),
            msg_color,
        );
    }
}

fn canvas_bottom_bar(ui: &mut Ui, app: &mut App2) {
    ui.painter().line_segment(
        [
            egui::pos2(ui.min_rect().min.x, ui.min_rect().min.y),
            egui::pos2(ui.min_rect().max.x, ui.min_rect().min.y),
        ],
        Stroke::new(1.0, P::RULE),
    );

    ui.horizontal_centered(|ui| {
        ui.set_height(44.0);
        ui.add_space(8.0);

        // Face chips
        let _n = app.preview.detections.len();
        let _selected_faces: Vec<usize> = app.selected_faces.iter().cloned().collect();
        let mut to_toggle: Option<usize> = None;
        for (i, det) in app.preview.detections.iter().enumerate() {
            let label = format!("face_{:03} · {:.2}", i + 1, det.detection.score);
            let selected = app.selected_faces.contains(&i);
            let alt = i % 2 == 1;
            let resp = face_chip(ui, &label, selected, alt);
            if resp.clicked() {
                to_toggle = Some(i);
            }
            ui.add_space(3.0);
        }
        if let Some(i) = to_toggle {
            if app.selected_faces.contains(&i) {
                app.selected_faces.remove(&i);
            } else {
                app.selected_faces.insert(i);
            }
        }

        // Zoom controls on right
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(8.0);
            if zoom_btn(ui, "⛶", "Fit") {
                app.zoom = 1.0;
                app.pan = egui::Vec2::ZERO;
            }
            ui.add_space(2.0);
            if zoom_btn(ui, "+", "Zoom in") {
                app.zoom = (app.zoom * 1.2).min(8.0);
            }
            ui.add_space(2.0);
            zoom_btn(ui, &format!("{:.0}%", app.zoom * 100.0), "Reset");
            ui.add_space(2.0);
            if zoom_btn(ui, "−", "Zoom out") {
                app.zoom = (app.zoom / 1.2).max(0.1);
            }
        });
    });
}

fn zoom_btn(ui: &mut egui::Ui, label: &str, _tip: &str) -> bool {
    let font = egui::FontId::monospace(10.5);
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font, P::INK2);
    let w = (galley.size().x + 14.0).max(26.0);
    let (resp, painter) = ui.allocate_painter(Vec2::new(w, 26.0), Sense::click());
    let r = resp.rect;
    let bg = if resp.hovered() {
        P::white_alpha(20)
    } else {
        P::white_alpha(10)
    };
    painter.rect_filled(r, 6.0, bg);
    painter.rect_stroke(r, 6.0, Stroke::new(1.0, P::RULE), egui::StrokeKind::Outside);
    painter.galley(
        r.min + Vec2::new((w - galley.size().x) / 2.0, (26.0 - galley.size().y) / 2.0),
        galley,
        P::INK2,
    );
    resp.clicked()
}
