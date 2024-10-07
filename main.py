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

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Load the .env file
load_dotenv()

# Open the video file
video_path = "input2.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history and state of vehicles
track_history = defaultdict(lambda: [])
vehicle_states = defaultdict(lambda: {"in_off_road": False, "on_screen": False})

# Create a Tkinter window
root = tk.Tk()
root.title("Off-Road Detector")

# Create a frame for video and logs
frame = tk.Frame(root)
frame.pack()

# Create a label for video output on the left side
video_label = tk.Label(frame)
video_label.grid(row=0, column=0)

# Create a text box for logs on the right side
log_box = Text(frame, width=40, height=20)
log_box.grid(row=0, column=1, padx=10)

# Define multiple off-road zones as a list of polygons
OFF_ROAD_ZONES = [
    np.array([[143, 0], [140, 74], [160, 150], [156, 198], [108, 269], [0, 342], [0, 0]], np.int32),  # First zone
    np.array([[163, 0], [161, 76], [191, 154], [191, 200], [160, 270], [47, 360], [640, 360], [640, 0]], np.int32),    # Second zone
    # Add more polygons as needed
]

# Function to check if a vehicle is in any off-road zone
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
            # Send the email using Mailtrap
            client.send(mail)
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {e}")
    
    # Start a new thread for the email function
    threading.Thread(target=email_thread).start()

# Set the class IDs you want to detect. (COCO class IDs for car and truck are 2 and 7)
TARGET_CLASSES = [2, 7]  # car, truck

# Modify the update_frame function to filter by target classes
def update_frame():
    success, frame = cap.read()

    if success:
        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Track detected vehicles
        detected_vehicle_ids = set()

        # Check if boxes and track IDs exist
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xywh.cpu()
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Get class IDs

            # Ensure the track IDs are not None
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Plot the tracks and detect events, filtering by class ID
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if class_id not in TARGET_CLASSES:
                        continue  # Skip if class ID is not in the target list

                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    detected_vehicle_ids.add(track_id)

                    # Check if the vehicle is in any off-road zone
                    in_off_road = is_in_any_off_road_zone(int(x), int(y))

                    # Handle entering or exiting the off-road zone
                    if in_off_road and not vehicle_states[track_id]["in_off_road"]:
                        log_box.insert(tk.END, f"ALERT! Vehicle ID {track_id} entered the off-road zone.\n")
                        log_box.see(tk.END)
                        send_email_alert(f"Vehicle ID {track_id} Off-Road Alert", f"Vehicle ID {track_id} entered the off-road zone.")
                        vehicle_states[track_id]["in_off_road"] = True
                    elif not in_off_road and vehicle_states[track_id]["in_off_road"]:
                        log_box.insert(tk.END, f"ALERT! Vehicle ID {track_id} left the off-road zone.\n")
                        log_box.see(tk.END)
                        send_email_alert(f"Vehicle ID {track_id} Off-Road Exit", f"Vehicle ID {track_id} left the off-road zone.")
                        vehicle_states[track_id]["in_off_road"] = False

                    # Detect when a vehicle first appears on the screen
                    if not vehicle_states[track_id]["on_screen"]:
                        log_box.insert(tk.END, f"INFO: Vehicle ID {track_id} appeared on the screen.\n")
                        log_box.see(tk.END)
                        vehicle_states[track_id]["on_screen"] = True

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
            else:
                annotated_frame = frame
        else:
            annotated_frame = frame

        # Handle disappearing vehicles
        for track_id in list(vehicle_states.keys()):
            if vehicle_states[track_id]["on_screen"] and track_id not in detected_vehicle_ids:
                log_box.insert(tk.END, f"INFO: Vehicle ID {track_id} disappeared from the screen.\n")
                log_box.see(tk.END)
                vehicle_states[track_id]["on_screen"] = False
                vehicle_states[track_id]["in_off_road"] = False  # Reset state when off screen

        # Draw all off-road zones on the annotated frame
        overlay = annotated_frame.copy()
        for zone in OFF_ROAD_ZONES:
            cv2.fillPoly(overlay, [zone], color=(255, 0, 255))
        alpha = 0.2  # Transparency factor
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
        # Convert the OpenCV image to PIL format for Tkinter
        img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

        # Call this function again to keep updating the frame
        video_label.after(10, update_frame)
    else:
        cap.release()

# Start the video feed
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Clean up when done
cap.release()
cv2.destroyAllWindows()
