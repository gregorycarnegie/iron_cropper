use eframe::{App, Frame, NativeOptions, egui};
use egui::CentralPanel;
use log::info;
use yunet_utils::init_logging;

fn main() -> eframe::Result<()> {
    init_logging(log::LevelFilter::Info).expect("failed to initialize logging");
    let options = NativeOptions::default();

    eframe::run_native(
        "YuNet Desktop",
        options,
        Box::new(|_cc| Box::new(YuNetApp::default())),
    )
}

#[derive(Default)]
struct YuNetApp {
    status: String,
}

impl App for YuNetApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        CentralPanel::default().show(ctx, |ui| {
            ui.heading("YuNet GUI scaffolding");
            ui.label("Face detection coming soon.");
            if ui.button("Log status").clicked() {
                self.status = "User pressed log status".to_owned();
                info!("{}", self.status);
            }
            if !self.status.is_empty() {
                ui.separator();
                ui.label(&self.status);
            }
        });
    }
}
