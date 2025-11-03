# OpenCV Parity Snapshot

Generated with:

```bash
cargo run --release -p yunet-core --example parity_report
```

Results (IoU threshold = 0.5):

- Expected detections: **164**
- YuNet detections: **168**
- Matched detections: **163**
- Average recall: **0.992**
- Average precision: **0.991**
- Mean IoU (matched): **0.983**
- Mean |score delta|: **0.0010**
- Worst IoU across matches: **0.899**
- Worst |score delta|: **0.0058**

## Notable gaps

- `074.jpg` — OpenCV reports no faces but the current pipeline produces a low-score false positive near the background. Tightening the score threshold to `0.92` suppresses it but would also hide valid low-confidence faces elsewhere, so it is documented instead of adjusted.
- `238_g.jpg` — two false positives on dense crowd background (OpenCV sees none). These occur at scores `0.91–0.92`; future work will experiment with crowd-aware NMS or adaptive thresholds.
- `240_g.jpg` — one of the six expected faces is missed (heavy occlusion). The IoU of the remaining matches is ≥0.98, so the gap is isolated to the occluded subject.
- `232_g.webp` — extra detection yielding local precision `0.667`; also linked to background clutter.

All other fixtures reach 100% recall and precision with score deltas within ±0.003. The telemetry example makes it easy to re-run this report after changes to preprocessing or postprocessing.
