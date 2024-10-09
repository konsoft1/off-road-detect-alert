import tkinter as tk
from tkinter import Text
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import mailtrap as mt
from dotenv import load_dotenv
import os

model = YOLO("yolo11n.pt")

video_path = "input2.mp4"
cap = cv2.VideoCapture(video_path)

OFF_ROAD_ZONES = [
    np.array([[143, 0], [140, 74], [160, 150], [156, 198], [108, 269], [0, 342], [0, 0]], np.int32),
    np.array([[163, 0], [161, 76], [191, 154], [191, 200], [160, 270], [47, 360], [640, 360], [640, 0]], np.int32),
]

load_dotenv('.env')

track_history = defaultdict(lambda: [])
vehicle_states = defaultdict(lambda: {"in_off_road": False, "on_screen": False})

root = tk.Tk()
root.title("Off-Road Detector")

frame = tk.Frame(root)
frame.pack()

video_label = tk.Label(frame)
video_label.grid(row=0, column=0)

log_box = Text(frame, width=40, height=20)
log_box.grid(row=0, column=1, padx=10)

# Function to check if a golf cart is in any off-road zone
def is_in_any_off_road_zone(x, y):
    for zone in OFF_ROAD_ZONES:
        if cv2.pointPolygonTest(zone, (x, y), False) >= 0:
            return True
    return False

client = mt.MailtrapClient(token=os.getenv('MAILTRAP_TOKEN'))

# Function to send an email alert in a separate thread
def send_email_alert(subject, body):
    def email_thread():
        mail = mt.Mail(
            sender=mt.Address(email="hello@demomailtrap.com", name="Mailtrap Test"),
            to=[mt.Address(email=os.getenv('EMAIL_TO'))],
            subject=subject,
            text=body,
            category="Integration Test",
        )
        try:
            client.send(mail)
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    threading.Thread(target=email_thread).start()


TARGET_CLASSES = {2: "", 3: "", 7: ""}

def update_frame():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        detected_vehicle_ids = set()

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xywh.cpu()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if class_id not in TARGET_CLASSES:
                        continue

                    x, y, w, h = box
                    left, top = int(x - w / 2), int(y - h / 2)
                    right, bottom = int(x + w / 2), int(y + h / 2)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    label = f"ID: {track_id}"
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Update track history
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    detected_vehicle_ids.add(track_id)

                    # Check if the golf cart is in any off-road zone
                    in_off_road = is_in_any_off_road_zone(int(x), int(y))

                    # Handle entering or exiting the off-road zone
                    if in_off_road and not vehicle_states[track_id]["in_off_road"]:
                        log_box.insert(tk.END, f"ALERT! ID {track_id} entered the off-road zone.\n")
                        log_box.see(tk.END)
                        send_email_alert(f"ID {track_id} Off-Road Alert", f"ID {track_id} entered the off-road zone.")
                        vehicle_states[track_id]["in_off_road"] = True
                    elif not in_off_road and vehicle_states[track_id]["in_off_road"]:
                        log_box.insert(tk.END, f"ALERT! ID {track_id} left the off-road zone.\n")
                        log_box.see(tk.END)
                        send_email_alert(f"ID {track_id} Off-Road Exit", f"ID {track_id} left the off-road zone.")
                        vehicle_states[track_id]["in_off_road"] = False

                    # Detect when a golf cart first appears on the screen
                    if not vehicle_states[track_id]["on_screen"]:
                        log_box.insert(tk.END, f"INFO: ID {track_id} appeared on the screen.\n")
                        log_box.see(tk.END)
                        vehicle_states[track_id]["on_screen"] = True

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
            else:
                annotated_frame = frame
        else:
            annotated_frame = frame

        # Handle disappearing golf carts
        for track_id in list(vehicle_states.keys()):
            if vehicle_states[track_id]["on_screen"] and track_id not in detected_vehicle_ids:
                log_box.insert(tk.END, f"INFO: ID {track_id} disappeared from the screen.\n")
                log_box.see(tk.END)
                vehicle_states[track_id]["on_screen"] = False
                vehicle_states[track_id]["in_off_road"] = False  # Reset state when off screen

        # Draw all off-road zones on the annotated frame
        overlay = frame.copy()
        for zone in OFF_ROAD_ZONES:
            cv2.fillPoly(overlay, [zone], color=(255, 0, 255))
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
        # Convert the OpenCV image to PIL format for Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

        video_label.after(10, update_frame)
    else:
        cap.release()

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
