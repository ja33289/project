import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import math
import tempfile
import json
import time



def Feedback(video_capture):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    pose_data = []
    start_time = time.time()
    elapsed_time = 0
    frame_number = 0

    while video_capture.isOpened() and elapsed_time < 30:
        ret, frame = video_capture.read()

        if not ret:
            break

        elapsed_time = time.time() - start_time

        pose_frame_data = {'frame_number': frame_number, 'landmarks': []}

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = [[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark]
            pose_frame_data['landmarks'] = landmarks

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        pose_data.append(pose_frame_data)

        frame_number += 1

    video_capture.release()

    # Convert pose_data to a JSON string
    pose_data_json = json.dumps(pose_data)
    return pose_data_json


def connect_landmarks(image, landmarks, shift_x=0, shift_y=0):
    # Define connections between body landmarks
    connections = [(11, 13), (12, 14), (13, 15), (14, 16), (15, 17),  # Connect head to upper body
                   (23, 24), (11, 23), (12, 24),(11, 12),  # Chest
                   (11, 23), (12, 24),  # Connect upper body to arms
                   (23, 25), (25, 27), (24, 26), (26, 28),  # legs
                   (27, 29), (29, 31), (27, 31), (28, 30), (28, 32), (30, 32),  # Feet
                   (23, 11), (24, 12),  # Connect upper body to head
                   ]

    for connection in connections:
        index1, index2 = connection
        if 0 <= index1 < len(landmarks) and 0 <= index2 < len(landmarks):
            landmark1 = landmarks[index1]
            landmark2 = landmarks[index2]
            x1, y1 = int(landmark1[0] * 1280) + shift_x, int(landmark1[1] * 480) + shift_y
            x2, y2 = int(landmark2[0] * 1280) + shift_x, int(landmark2[1] * 480) + shift_y
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 1)

def draw_landmarks(image, landmarks, shift_x=0, shift_y=0, color=(0, 0, 0)):
    for landmark in landmarks:
        x, y = int(landmark[0] * 1280) + shift_x, int(landmark[1] * 480) + shift_y
        cv2.circle(image, (int(x), int(y)), 3, color, -1)

def Overlay(json_input1, json_input2, accuracy_threshold=10):
    landmarks1 = json.loads(json_input1)
    landmarks2 = json.loads(json_input2)

    output_video_path = tempfile.NamedTemporaryFile(suffix='.mov', delete=False).name
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), 30.0, (1280,720))
    #output_frames = []
    shift_x1 = 0
    shift_y1 = 0
    shift_x2 = 0
    shift_y2 = 0
    total_accurate_matches = 0
    total_leg_angles = 0
    total_ankle_angles = 0

    x_min_1 = min(landmark[0] for landmark in landmarks1[0]['landmarks'])
    x_max_1 = max(landmark[0] for landmark in landmarks1[0]['landmarks'])
    y_min_1 = min(landmark[1] for landmark in landmarks1[0]['landmarks'])
    y_max_1 = max(landmark[1] for landmark in landmarks1[0]['landmarks'])

    x_min_2 = min(landmark[0] for landmark in landmarks2[0]['landmarks'])
    x_max_2 = max(landmark[0] for landmark in landmarks2[0]['landmarks'])
    y_min_2 = min(landmark[1] for landmark in landmarks2[0]['landmarks'])
    y_max_2 = max(landmark[1] for landmark in landmarks2[0]['landmarks'])

    # Calculate scaling factors
    scale_x = (x_max_2 - x_min_2) / (x_max_1 - x_min_1)
    scale_y = (y_max_2 - y_min_2) / (y_max_1 - y_min_1)

    # Process frames to create the output video
    for frame_number in range(min(len(landmarks1), len(landmarks2))):
        
        pose_frame_data1 = landmarks1[frame_number]['landmarks']
        pose_frame_data2 = landmarks2[frame_number]['landmarks']

        # Calculate angles for the legs of both sets of body marks
        legangle_pose1 = [(math.degrees(math.atan2(pose_frame_data1[23][1] - pose_frame_data1[25][1], pose_frame_data1[23][0] - pose_frame_data1[25][0])) +
                        math.degrees(math.atan2(pose_frame_data1[25][1] - pose_frame_data1[27][1], pose_frame_data1[25][0] - pose_frame_data1[27][0]))),
                        (math.degrees(math.atan2(pose_frame_data1[24][1] - pose_frame_data1[26][1], pose_frame_data1[24][0] - pose_frame_data1[26][0])) +
                        math.degrees(math.atan2(pose_frame_data1[26][1] - pose_frame_data1[28][1], pose_frame_data1[26][0] - pose_frame_data1[28][0])))]

        legangle_pose2 = [(math.degrees(math.atan2(pose_frame_data2[23][1] - pose_frame_data2[25][1], pose_frame_data2[23][0] - pose_frame_data2[25][0])) +
                        math.degrees(math.atan2(pose_frame_data2[25][1] - pose_frame_data2[27][1], pose_frame_data2[25][0] - pose_frame_data2[27][0]))),
                        (math.degrees(math.atan2(pose_frame_data2[24][1] - pose_frame_data2[26][1], pose_frame_data2[24][0] - pose_frame_data2[26][0])) +
                        math.degrees(math.atan2(pose_frame_data2[26][1] - pose_frame_data2[28][1], pose_frame_data2[26][0] - pose_frame_data2[28][0])))]

                # Calculate angles for the legs of both sets of body marks
        ankleangle_pose1 = [(math.degrees(math.atan2(pose_frame_data1[25][1] - pose_frame_data1[27][1], pose_frame_data1[25][0] - pose_frame_data1[27][0])) +
                        math.degrees(math.atan2(pose_frame_data1[27][1] - pose_frame_data1[31][1], pose_frame_data1[27][0] - pose_frame_data1[31][0]))),
                        (math.degrees(math.atan2(pose_frame_data1[26][1] - pose_frame_data1[28][1], pose_frame_data1[26][0] - pose_frame_data1[28][0])) +
                        math.degrees(math.atan2(pose_frame_data1[28][1] - pose_frame_data1[32][1], pose_frame_data1[28][0] - pose_frame_data1[32][0])))]

        ankleangle_pose2 = [(math.degrees(math.atan2(pose_frame_data2[25][1] - pose_frame_data2[27][1], pose_frame_data2[25][0] - pose_frame_data2[27][0])) +
                        math.degrees(math.atan2(pose_frame_data2[25][1] - pose_frame_data2[27][1], pose_frame_data2[27][0] - pose_frame_data2[25][0]))),
                        (math.degrees(math.atan2(pose_frame_data2[26][1] - pose_frame_data2[28][1], pose_frame_data2[26][0] - pose_frame_data2[28][0])) +
                        math.degrees(math.atan2(pose_frame_data2[28][1] - pose_frame_data2[32][1], pose_frame_data2[28][0] - pose_frame_data2[32][0])))]

        # Compare leg angles between the two poses
        for angle1 in legangle_pose1:
            for angle2 in legangle_pose2:
                if abs(angle1 - angle2) <= accuracy_threshold:
                    total_accurate_matches += 1
                    break  # Exit inner loop once a match is found
        total_leg_angles += len(legangle_pose1)  # Add all angles from the first pose

        # Compare ankle angles between the two poses
        for angle1 in ankleangle_pose1:
            for angle2 in ankleangle_pose2:
                if abs(angle1 - angle2) <= accuracy_threshold:
                    total_accurate_matches += 1
                    break  # Exit inner loop once a match is found
        total_ankle_angles += len(ankleangle_pose1)

    accuracy_percentage = (total_accurate_matches / (total_leg_angles + total_ankle_angles)) * 100 if (total_leg_angles + total_ankle_angles) > 0 else 0
    pose_frame_data1 = landmarks1[frame_number]['landmarks']
    pose_frame_data2 = landmarks2[frame_number]['landmarks']
    pose_frame_data1 = [(landmark[0] * scale_x, landmark[1] * scale_y) for landmark in pose_frame_data1]
    center_x1 = sum(landmark[0] for landmark in pose_frame_data1) / len(pose_frame_data1)
    center_y1 = sum(landmark[1] for landmark in pose_frame_data1) / len(pose_frame_data1)
    center_x2 = sum(landmark[0] for landmark in pose_frame_data2) / len(pose_frame_data2)
    center_y2 = sum(landmark[1] for landmark in pose_frame_data2) / len(pose_frame_data2)
    shift_x_center = 350- ((720 / 2) - (center_x1 + center_x2) / 2)
    shift_y_center = 360-((480 / 2) - (center_y1 + center_y2) / 2)
    shift_x1 += shift_x_center
    shift_y1 += shift_y_center
    shift_x2 += shift_x_center
    shift_y2 += shift_y_center
    # Process frames again to create the output video
    for frame_number in range(min(len(landmarks1), len(landmarks2))):
        white_screen = np.ones((720,1280,3), dtype=np.uint8) * 255

        pose_frame_data1 = landmarks1[frame_number]['landmarks']
        pose_frame_data2 = landmarks2[frame_number]['landmarks']
        pose_frame_data1 = [(landmark[0] * scale_x, landmark[1] * scale_y) for landmark in pose_frame_data1]

        draw_landmarks(white_screen, pose_frame_data1, shift_x=shift_x1, shift_y=shift_y1, color=(255, 0, 0))
        draw_landmarks(white_screen, pose_frame_data2, shift_x=shift_x2, shift_y=shift_y2, color=(0, 0, 255))
        connect_landmarks(white_screen, pose_frame_data1, shift_x=shift_x1, shift_y=shift_y1)
        connect_landmarks(white_screen, pose_frame_data2, shift_x=shift_x2, shift_y=shift_y2)
        # Add key at the bottom left showing "Exercise Marks" in blue and "User Marks" in red
        cv2.putText(white_screen, 'Exercise Marks (Blue)', (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(white_screen, 'User Marks (Red)', (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Add text overlay for overall similarity
        cv2.putText(white_screen, f'Overall Similarity: {accuracy_percentage:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        #output_frames.append(white_screen)
        out.write(white_screen)

    out.release()
    return output_video_path


def main():
    st.title("AI Soccer Skillz")
    st.subheader("Upload two videos to get your percentage of accuracy!")

    exercise_video_file = st.file_uploader("Upload Exercise Video", type=['mp4', 'mov'], key='exercise_video')
    if not exercise_video_file:
        st.warning("Please upload the exercise video.")
        return

    user_video_file = st.file_uploader("Upload User Video", type=['mp4', 'mov'], key='user_video')
    if not user_video_file:
        st.warning("Please upload the user video.")
        return

    if st.button("Ready to Receive Results"):
        with tempfile.NamedTemporaryFile(delete=False) as exercise_temp, tempfile.NamedTemporaryFile(delete=False) as user_temp:
            # Write the contents of the file objects to temporary files
            exercise_temp.write(exercise_video_file.read())
            exercise_temp.seek(0)
            user_temp.write(user_video_file.read())
            user_temp.seek(0)

            # Create video capture objects from the temporary files
            exercise_video_obj = cv2.VideoCapture(exercise_temp.name)
            user_video_obj = cv2.VideoCapture(user_temp.name)

            # Get feedback data for both videos
            feedback_json1 = Feedback(exercise_video_obj)
            feedback_json2 = Feedback(user_video_obj)

        # Overlay the feedback data
        overlay_vid = Overlay(feedback_json1, feedback_json2)
        # Display the resulting video
        st.video(overlay_vid, format='mov', start_time=0)

if __name__ == '__main__':
    main()