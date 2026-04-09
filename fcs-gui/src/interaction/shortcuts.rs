//! Keyboard shortcut handling.

use egui::{Context as EguiContext, Key};

/// Actions that can be triggered by keyboard shortcuts.
#[derive(Default)]
pub struct ShortcutActions {
    pub undo: bool,
    pub redo: bool,
    pub horiz_delta: f32,
    pub vert_delta: f32,
    pub face_height_delta: f32,
    pub preset: Option<&'static str>,
    pub toggle_enhance: bool,
    pub export: bool,
}

/// Captures keyboard input and translates it into shortcut actions.
pub fn capture_shortcut_actions(ctx: &EguiContext, wants_text: bool) -> ShortcutActions {
    let mut actions = ShortcutActions::default();

    ctx.input(|input| {
        let base_step = if input.modifiers.shift { 0.1 } else { 0.05 };
        let face_step = if input.modifiers.shift { 5.0 } else { 1.0 };
        let command = input.modifiers.command;

        // Undo/Redo
        if input.key_pressed(Key::Z) && command {
            if input.modifiers.shift {
                actions.redo = true;
            } else {
                actions.undo = true;
            }
        }
        if input.key_pressed(Key::Y) && command {
            actions.redo = true;
        }

        // Movement and adjustment keys (only when not typing text)
        if !command && !wants_text {
            // Arrow keys for offset adjustment
            if input.key_pressed(Key::ArrowLeft) {
                actions.horiz_delta -= base_step;
            }
            if input.key_pressed(Key::ArrowRight) {
                actions.horiz_delta += base_step;
            }
            if input.key_pressed(Key::ArrowUp) {
                actions.vert_delta -= base_step;
            }
            if input.key_pressed(Key::ArrowDown) {
                actions.vert_delta += base_step;
            }

            // +/- for face height adjustment
            if input.key_pressed(Key::Minus) {
                actions.face_height_delta -= face_step;
            }
            if input.key_pressed(Key::Equals) {
                actions.face_height_delta += face_step;
            }

            // Number keys for presets
            const PRESETS: [(&str, Key); 6] = [
                ("linkedin", Key::Num1),
                ("passport", Key::Num2),
                ("instagram", Key::Num3),
                ("idcard", Key::Num4),
                ("avatar", Key::Num5),
                ("headshot", Key::Num6),
            ];
            for (preset, key) in PRESETS {
                if input.key_pressed(key) {
                    actions.preset = Some(preset);
                }
            }

            // Space to toggle enhancement
            if input.key_pressed(Key::Space) {
                actions.toggle_enhance = true;
            }

            // Enter to export
            if input.key_pressed(Key::Enter) {
                actions.export = true;
            }
        }
    });

    actions
}
