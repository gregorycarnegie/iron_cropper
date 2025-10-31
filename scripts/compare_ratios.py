import cv2
from pathlib import Path

samples = ['006.jpg', '190_g.jpg', '002_n.jpg']

for name in samples:
    image = cv2.imread(str(Path('fixtures/images') / name))
    if image is None:
        print(f'skip {name}, missing image')
        continue

    def run(size):
        resized = cv2.resize(image, size)
        detector = cv2.FaceDetectorYN.create('models/face_detection_yunet_2023mar_640.onnx', '', size, 0.9, 0.3, 5000)
        detector.setInputSize(size)
        _, det = detector.detect(resized)
        return det

    det640 = run((640, 640))
    det320 = run((320, 320))
    if det640 is None or det320 is None:
        print(f'skip {name}, no detections')
        continue

    ratios = det640[0] / det320[0]
    diffs = det640[0] - det320[0]
    print(name)
    print('ratios:', ratios[:14], ratios[14])
    print('diffs:', diffs[:14], diffs[14])
