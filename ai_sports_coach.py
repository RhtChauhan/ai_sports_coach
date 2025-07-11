import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import json
import os
import matplotlib.pyplot as plt

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load or initialize progress
PROGRESS_FILE = "user_progress.json"

if not os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"squat_angles": []}, f)

with open(PROGRESS_FILE, 'r') as f:
    progress_data = json.load(f)

# UI
st.title("🏋️ AI Sports Coach — Free Squat Form Evaluator")
st.write("Upload a short video (5–10 seconds) of your squat, and get feedback!")

uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi"])

# Pose angle calculation
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arccos(
        np.clip(np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1.0, 1.0)
    )
    return np.degrees(radians)

def extract_pose_and_angle(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        angle = calculate_angle(hip, knee, ankle)
        return angle
    return None

def generate_feedback(angle):
    if angle > 160:
        return "You're standing. Start lowering into your squat."
    elif 140 < angle <= 160:
        return "Good depth starting. Keep lowering with control."
    elif 90 <= angle <= 140:
        return "Great squat depth!"
    elif angle < 90:
        return "You're going too deep — may stress your knees."
    else:
        return "Could not detect form clearly."

def draw_angle_plot():
    angles = progress_data["squat_angles"]
    if angles:
        st.subheader("📈 Progress Over Time")
        fig, ax = plt.subplots()
        ax.plot(angles, marker='o')
        ax.set_title("Squat Knee Angle")
        ax.set_ylabel("Degrees")
        ax.set_xlabel("Session")
        st.pyplot(fig)

if uploaded_file is not None:
    st.video(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)
    angle_data = []

    st.info("Analyzing your squat form...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        angle = extract_pose_and_angle(frame)
        if angle:
            angle_data.append(angle)

    cap.release()

    if angle_data:
        average_angle = int(np.mean(angle_data))
        progress_data["squat_angles"].append(average_angle)
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)

        st.success(f"✅ Avg Knee Angle: {average_angle}°")
        st.write(generate_feedback(average_angle))

        draw_angle_plot()
    else:
        st.warning("Couldn't detect any squats. Try uploading a clearer video.")
