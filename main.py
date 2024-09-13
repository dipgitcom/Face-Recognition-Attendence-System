import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
import logging

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize video capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    logging.error("Failed to open the webcam. Exiting...")
    exit()

# Load known faces with exception handling
def load_face_encodings(file_paths):
    encodings = []
    names = []
    for file_path in file_paths:
        try:
            image = face_recognition.load_image_file(file_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                names.append(os.path.splitext(os.path.basename(file_path))[0])
            else:
                logging.warning(f"No faces found in {file_path}")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
    return encodings, names

face_files = ["faces/Dipraj.jpg", "faces/Popy.jpg"]
known_face_encodings, known_face_names = load_face_encodings(face_files)

if not known_face_encodings:
    logging.error("No face encodings were loaded. Exiting...")
    exit()

# List of expected students
students = known_face_names.copy()

# Initialize lists for face locations and encodings
face_locations = []
face_encodings = []

# Get the current date for the attendance file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create 'Attendance' folder if it doesn't exist
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

# Open the CSV file for writing attendance
csv_filename = f"Attendance/{current_date}.csv"
f = open(csv_filename, "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time In"])

# Frame skipping for performance improvement
frame_count = 0  # Frame skipping counter

while True:
    ret, frame = video_capture.read()
    if not ret:
        logging.error("Failed to capture frame from camera. Exiting...")
        break

    frame_count += 1
    if frame_count % 5 != 0:  # Process every 5th frame
        continue

    # Resize the frame for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV default) to RGB color (required by face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces in the frame using 'hog' model for speed
    try:
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    except Exception as e:
        logging.error(f"Error during face recognition: {e}")
        continue

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            current_time = datetime.now().strftime("%H:%M:%S")

            # Display attendance on the frame
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Text settings for name and time
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"{name} Present", (left, top - 10), font, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {current_time}", (left, top + 20), font, 0.75, (0, 255, 0), 2)

            # Log attendance if the student is present for the first time
            if name in students:
                students.remove(name)
                lnwriter.writerow([name, current_time])
                logging.info(f"{name} marked present at {current_time}")

    # Show the updated frame with bounding boxes and text
    cv2.imshow("Attendance System", frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        logging.info("Attendance session terminated by user.")
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
f.close()
logging.info(f"Attendance saved to {csv_filename}.")
