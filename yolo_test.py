from ultralytics import YOLO
import cv2
import math

#video_path = r"input.mp4"
cap = cv2.VideoCapture(0)

model = YOLO("yolov10s.pt")

classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
]
vehicle_classes = [2, 5, 7]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            if cls in vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                print("Class name -->", classNames[cls])

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(
                    img, classNames[cls], org, font, fontScale, color, thickness
                )

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break
