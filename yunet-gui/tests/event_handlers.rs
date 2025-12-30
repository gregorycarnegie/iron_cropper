//! Integration tests for event handlers and input processing.

use egui::{Context, Event, Key, Modifiers, RawInput};
use yunet_gui::YuNetApp;

#[test]
fn test_undo_redo_shortcuts() {
    let mut app = YuNetApp::test_instance();

    // 1. Modify state
    app.adjust_horizontal_offset(0.5);
    assert!(app.settings.crop.horizontal_offset > 0.0);
    assert_eq!(app.crop_history_index, 1); // Initial (0) + 1 change

    // 2. Setup Undo Shortcut (Cmd+Z)
    let ctx = Context::default();
    let mut input = RawInput::default();
    input.modifiers = Modifiers {
        command: true,
        ctrl: true,
        ..Modifiers::default()
    };
    input.events.push(Event::Key {
        key: Key::Z,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: input.modifiers,
    });

    // Begin frame to populate input
    ctx.begin_pass(input);

    // 3. Handle shortcuts
    app.handle_shortcuts(&ctx);

    // 4. Verify Undo (should revert to default 0.0)
    assert_eq!(
        app.settings.crop.horizontal_offset, 0.0,
        "Undo should revert offset"
    );
    assert_eq!(app.crop_history_index, 0);

    // 5. Setup Redo Shortcut (Ctrl+Y or Ctrl+Shift+Z)
    // Testing Ctrl+Y
    let mut input_redo = RawInput::default();
    input_redo.modifiers = Modifiers {
        command: true,
        ctrl: true,
        ..Modifiers::default()
    };
    input_redo.events.push(Event::Key {
        key: Key::Y,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: input_redo.modifiers,
    });
    ctx.begin_pass(input_redo);

    app.handle_shortcuts(&ctx);

    // 6. Verify Redo (should return to 0.5)
    assert_eq!(
        app.settings.crop.horizontal_offset, 0.5,
        "Redo should restore offset"
    );
    assert_eq!(app.crop_history_index, 1);
}

#[test]
fn test_preset_shortcuts() {
    let mut app = YuNetApp::test_instance();

    // 1. Setup Shortcut for "LinkedIn" (Num1)
    let ctx = Context::default();
    let mut input = RawInput::default();
    input.events.push(Event::Key {
        key: Key::Num1,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });

    ctx.begin_pass(input);

    // 2. Handle shortcuts
    app.handle_shortcuts(&ctx);

    // 3. Verify Preset Changed
    assert_eq!(app.settings.crop.preset, "linkedin");
    assert_eq!(app.settings.crop.output_width, 400);

    // Test "Passport" (Num2)
    let mut input_passport = RawInput::default();
    input_passport.events.push(Event::Key {
        key: Key::Num2,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    ctx.begin_pass(input_passport);

    app.handle_shortcuts(&ctx);

    assert_eq!(app.settings.crop.preset, "passport");
}
