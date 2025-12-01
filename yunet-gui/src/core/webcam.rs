//! Webcam capture and real-time detection for GUI.

use crate::types::WebcamStatus;
use crate::{JobMessage, YuNetApp};

use anyhow::Result;
use image::DynamicImage;
use log::{error, info, warn};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};
use yunet_core::YuNetDetector;
use yunet_utils::{WebcamCapture, estimate_sharpness};

impl YuNetApp {
    /// Start the webcam capture in a background thread.
    pub fn start_webcam(&mut self) {
        if !matches!(self.webcam_state.status, WebcamStatus::Inactive) {
            return;
        }

        if self.detector.is_none() {
            self.webcam_state.status = WebcamStatus::Error;
            self.webcam_state.error_message = Some("No model loaded".to_string());
            return;
        }

        info!(
            "Starting webcam capture: device={}, {}x{} @ {} fps",
            self.webcam_state.device_index,
            self.webcam_state.width,
            self.webcam_state.height,
            self.webcam_state.fps
        );

        self.webcam_state.status = WebcamStatus::Starting;
        self.webcam_state.error_message = None;
        self.webcam_state.frames_captured = 0;
        self.webcam_state.total_faces = 0;

        let device_index = self.webcam_state.device_index;
        let width = self.webcam_state.width;
        let height = self.webcam_state.height;
        let fps = self.webcam_state.fps;
        let detector = self.detector.clone().unwrap();
        let job_tx = self.job_tx.clone();

        // Create stop flag
        let stop_flag = Arc::new(AtomicBool::new(false));
        self.webcam_state.stop_flag = Some(stop_flag.clone());

        // Spawn webcam capture thread
        thread::spawn(move || {
            let result = run_webcam_loop(
                device_index,
                width,
                height,
                fps,
                detector,
                job_tx,
                stop_flag,
            );
            if let Err(e) = result {
                error!("Webcam capture error: {}", e);
            }
        });
    }

    /// Stop the webcam capture.
    pub fn stop_webcam(&mut self) {
        if matches!(self.webcam_state.status, WebcamStatus::Inactive) {
            return;
        }

        info!("Stopping webcam capture");
        self.webcam_state.status = WebcamStatus::Stopping;

        // Signal the webcam thread to stop
        if let Some(flag) = &self.webcam_state.stop_flag {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Process webcam frame updates from the background thread.
    pub fn process_webcam_frame(
        &mut self,
        ctx: &egui::Context,
        image: DynamicImage,
        frame_number: u32,
        detections: Vec<crate::DetectionWithQuality>,
    ) {
        use crate::core::cache::load_texture_from_image;

        self.webcam_state.status = WebcamStatus::Active;
        self.webcam_state.frames_captured = frame_number;
        self.webcam_state.total_faces += detections.len();

        let (width, height) = (image.width(), image.height());

        // Create texture for display
        let texture = load_texture_from_image(ctx, &image, &mut self.texture_seq);

        // Update preview with the new frame and detections
        self.preview.source_image = Some(Arc::new(image));
        self.preview.image_path = Some(format!("webcam_frame{}", frame_number).into());
        self.preview.texture = Some(texture);
        self.preview.image_size = Some((width, height));
        self.preview.detections = detections;
    }

    /// Handle webcam error from background thread.
    pub fn handle_webcam_error(&mut self, error: String) {
        error!("Webcam error: {}", error);
        self.webcam_state.status = WebcamStatus::Error;
        self.webcam_state.error_message = Some(error);
    }

    /// Handle webcam stop event.
    pub fn handle_webcam_stopped(&mut self) {
        info!("Webcam stopped");
        self.webcam_state.status = WebcamStatus::Inactive;
    }
}

/// Run the webcam capture loop in a background thread.
fn run_webcam_loop(
    device_index: u32,
    width: u32,
    height: u32,
    fps: u32,
    detector: Arc<YuNetDetector>,
    job_tx: mpsc::Sender<JobMessage>,
    stop_flag: Arc<AtomicBool>,
) -> Result<()> {
    use crate::{DetectionOrigin, DetectionWithQuality};

    let mut webcam = WebcamCapture::with_device_index(device_index, width, height, fps)?;

    info!(
        "Webcam opened: {}x{} @ {} fps",
        webcam.resolution().0,
        webcam.resolution().1,
        webcam.frame_rate()
    );

    let frame_duration = Duration::from_millis(1000 / fps.max(1) as u64);
    let mut frame_number = 0u32;
    let mut last_detections = Vec::new();

    loop {
        // Check stop flag
        if stop_flag.load(Ordering::Relaxed) {
            info!("Stop signal received, ending webcam capture");
            break;
        }

        // Capture frame
        let image = match webcam.capture_frame() {
            Ok(img) => img,
            Err(e) => {
                let _ = job_tx.send(JobMessage::WebcamError(e.to_string()));
                break;
            }
        };

        frame_number += 1;

        // Skip every other frame for detection to improve performance
        let detections = if frame_number.is_multiple_of(2) {
            // Reuse detections from previous frame
            last_detections.clone()
        } else {
            // Run detection on this frame
            match detector.detect_image(&image) {
                Ok(output) => {
                    let dets: Vec<DetectionWithQuality> = output
                        .detections
                        .into_iter()
                        .map(|det| {
                            let bbox_region = image.crop_imm(
                                det.bbox.x.max(0.0) as u32,
                                det.bbox.y.max(0.0) as u32,
                                det.bbox.width.max(1.0) as u32,
                                det.bbox.height.max(1.0) as u32,
                            );
                            let (quality_score, quality) = estimate_sharpness(&bbox_region);
                            let bbox = det.bbox;

                            DetectionWithQuality {
                                detection: det,
                                quality_score,
                                quality,
                                thumbnail: None,
                                current_bbox: bbox,
                                original_bbox: bbox,
                                origin: DetectionOrigin::Detector,
                            }
                        })
                        .collect();
                    last_detections = dets.clone();
                    dets
                }
                Err(e) => {
                    warn!("Detection failed on frame {}: {}", frame_number, e);
                    Vec::new()
                }
            }
        };

        // Send frame with detections to GUI
        let _ = job_tx.send(JobMessage::WebcamFrame {
            image,
            frame_number,
            detections,
        });

        // Rate limiting
        thread::sleep(frame_duration);
    }

    webcam.stop()?;
    let _ = job_tx.send(JobMessage::WebcamStopped);

    Ok(())
}
