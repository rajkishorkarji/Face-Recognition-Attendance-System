from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
import threading
from datetime import datetime
from win32com.client import Dispatch

# ----------------- CONSTANTS TO TUNE ----------------- #
DISTANCE_THRESHOLD =  5000
# Time in seconds to wait before repeating a voice message for the same person.
REPEAT_INTERVAL = 15
# Cooldown specifically for the "Unknown person" message.
UNKNOWN_PERSON_COOLDOWN = 10
# ----------------------------------------------------- #

# ---------------- ASYNC SPEAK ---------------- #
def speak_async(message):
    """Speaks a message in a separate thread to avoid freezing the camera feed."""
    def _speak(msg):
        speaker = Dispatch("SAPI.SpVoice")
        speaker.Speak(msg)
    threading.Thread(target=_speak, args=(message,), daemon=True).start()

# ---------------- ASYNC FILE WRITE ---------------- #
def write_attendance_async(filename, row):
    """Writes a row to the CSV file in a separate thread."""
    def _write(file, row_data):
        # Use 'a+' to create the file if it doesn't exist
        with open(file, "a+", newline="") as csvfile:
            # Move cursor to the beginning to check if the file is empty
            csvfile.seek(0)
            # If the file is empty, write the header first
            if not csvfile.read(1):
                writer = csv.writer(csvfile)
                writer.writerow(["NAME", "TIME"])
            # Append the new attendance record
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
    threading.Thread(target=_write, args=(filename, row), daemon=True).start()

# ---------------- VIDEO & FACE DETECTOR ---------------- #
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# ---------------- LOAD TRAINED DATA ---------------- #
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    print("✅ Training data loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'names.pkl' or 'faces_data.pkl' not found. Please generate the data first.")
    exit()

# Safety check for mismatch
if FACES.shape[0] != len(LABELS):
    print("⚠️ Warning: Mismatch between faces data and labels. Trimming to the smaller size.")
    min_len = min(FACES.shape[0], len(LABELS))
    FACES = FACES[:min_len]
    LABELS = LABELS[:min_len]

# ---------------- TRAIN KNN MODEL ---------------- #
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(FACES, LABELS)
print("✅ KNN model trained.")

# ---------------- LOAD BACKGROUND IMAGE ---------------- #
try:
    imgBackground = cv2.imread("BACKGROUND IMAGE.png")
except:
    imgBackground = None
    print("⚠️ Warning: 'BACKGROUND IMAGE.png' not found. Displaying raw camera feed only.")


# ---------------- SETUP ATTENDANCE TRACKING ---------------- #
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
attendance_file = f"Attendance/Attendance_{date}.csv"

# Create Attendance directory if it doesn't exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')

# Load already marked names for the day
marked = set()
if os.path.isfile(attendance_file):
    with open(attendance_file, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                marked.add(row[0])

# Dictionary to track the last time a message was spoken for each person
last_spoken_time = {}

# ---------------- MAIN LOOP ---------------- #
while True:
    ret, frame = video.read()
    if not ret:
        print("❌ Error: Failed to grab frame from camera.")
        break
        
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
        # Get distances and indices of the nearest neighbors
        distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
        
        # --- IMPROVED ACCURACY CHECK ---
        if distances[0][0] > DISTANCE_THRESHOLD:
            name = "Unknown"
        else:
            name = LABELS[indices[0][0]]

        current_time = time.time()
        timestamp = datetime.fromtimestamp(current_time).strftime("%H:%M:%S")

        # ---- Draw rectangle + name ----
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
        cv2.putText(frame, name, (x+5, y-15),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

        # ---- Attendance & Voice Logic (with Cooldown) ----
        if name == "Unknown":
            if current_time - last_spoken_time.get("Unknown", 0) > UNKNOWN_PERSON_COOLDOWN:
                speak_async("Unknown person")
                last_spoken_time["Unknown"] = current_time
        else:
            if name not in marked:
                marked.add(name)
                speak_async(f"Welcome {name}")
                write_attendance_async(attendance_file, [name, timestamp])
                # Set initial spoken time to prevent immediate repeat message
                last_spoken_time[name] = current_time 
            else:
                # Check if enough time has passed to speak again
                if current_time - last_spoken_time.get(name, 0) > REPEAT_INTERVAL:
                    speak_async(f"Attendance for {name} is already taken.")
                    last_spoken_time[name] = current_time

    # Paste processed frame into background if available
    if imgBackground is not None:
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Attendance System", imgBackground)
    else:
        cv2.imshow("Attendance System", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()