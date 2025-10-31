import sys
import cv2

image = cv2.imread('fixtures/images/006.jpg')
if image is None:
    raise SystemExit('missing image')
orig_h, orig_w = image.shape[:2]
if len(sys.argv) >= 3:
    input_size = (int(sys.argv[1]), int(sys.argv[2]))
else:
    input_size = (640, 640)
resized = cv2.resize(image, input_size)
detector = cv2.FaceDetectorYN.create('models/face_detection_yunet_2023mar.onnx', '', input_size, 0.9, 0.3, 5000)
detector.setInputSize(input_size)
_, det = detector.detect(resized)
print('input size:', input_size)
print('raw detections:\n', det)
if det is not None:
    det_scaled = det.copy()
    scale_x = orig_w / float(input_size[0])
    scale_y = orig_h / float(input_size[1])
    for idx in range(det.shape[0]):
        det_scaled[idx, 0] *= scale_x
        det_scaled[idx, 1] *= scale_y
        det_scaled[idx, 2] *= scale_x
        det_scaled[idx, 3] *= scale_y
        for lm in range(5):
            det_scaled[idx, 4 + 2 * lm] *= scale_x
            det_scaled[idx, 5 + 2 * lm] *= scale_y
    print('scaled detections:\n', det_scaled)
else:
    print('no detections')
