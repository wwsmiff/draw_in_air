#!/usr/bin/env python3

import mediapipe as mp
import numpy as np
import cv2

# setup mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# setup CV window
cv2.namedWindow("Draw!", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Draw!", 1280, 720)

# capture webcam
cap = cv2.VideoCapture(0)

# CONSTS
CAPTURE_WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
CAPTURE_HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
CLOSE_THRESHOLD = 85
OPEN_THRESHOLD = 80
EXPORT_PATH = "output.png"

results = None

primary_index = ()
secondary_index = ()

primary_coords = []
secondary_coords = []
start = 0
current_color = (0, 0, 0)


# get coordinates of tracked fingers
def extract_coordinates(landmarks, finger_number):
    return (int(CAPTURE_WIDTH * float(str(landmarks.landmark[finger_number]).split('\n')[0].split(' ')[1])),
            int(CAPTURE_HEIGHT * float(str(landmarks.landmark[finger_number]).split('\n')[1].split(' ')[1])),
            int(CAPTURE_HEIGHT * float(str(landmarks.landmark[finger_number]).split('\n')[2].split(' ')[1])))


def handle_gestures(landmarks):
    index = extract_coordinates(landmarks, 8)
    thumb = extract_coordinates(landmarks, 4)
    middle = extract_coordinates(landmarks, 12)
    ring = extract_coordinates(landmarks, 16)
    pinky = extract_coordinates(landmarks, 20)
    zero = extract_coordinates(landmarks, 0)

    # handle closed hand
    if zero[1] - index[1] < CLOSE_THRESHOLD and \
       zero[1] - thumb[1] < CLOSE_THRESHOLD + 20 and \
       zero[1] - middle[1] < CLOSE_THRESHOLD and \
       zero[1] - ring[1] < CLOSE_THRESHOLD and \
       zero[1] - pinky[1] < CLOSE_THRESHOLD:
        return "gesture_close"

    # handle open hand
    if zero[1] - index[1] > OPEN_THRESHOLD and \
       zero[1] - thumb[1] > OPEN_THRESHOLD - 40 and \
       zero[1] - middle[1] > OPEN_THRESHOLD and \
       zero[1] - ring[1] > OPEN_THRESHOLD and \
       zero[1] - pinky[1] > OPEN_THRESHOLD:
        return "gesture_open"

    # handle showing one and two
    if zero[1] - pinky[1] < CLOSE_THRESHOLD and \
       zero[1] - ring[1] < CLOSE_THRESHOLD and \
       zero[1] - thumb[1] < CLOSE_THRESHOLD:
       if zero[1] - middle[1] < CLOSE_THRESHOLD:
           return "gesture_one"
       return "gesture_two"



with mp_hands.Hands(
    min_detection_confidence=0.80,
    min_tracking_confidence=0.75) as hands:
    while cap.isOpened():
        _, img = cap.read()
        image = img.copy()

        # image processing
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        original_image = image.copy()

        # rendering the line to the screen

        for i in range(start, len(primary_coords) - 1):
            if primary_coords[i] == 0:
                continue
            elif primary_coords[i] == 0 and primary_coords[i + 1] == 0:
                continue
            elif primary_coords[i + 1] == 0:
                continue
            else:
                cv2.line(image, (primary_coords[i][1][0], primary_coords[i][1][1]), (primary_coords[i+1][1][0], primary_coords[i+1][1][1]), color = primary_coords[i][0], thickness = 5)

        # rendering and changing colors according to gesture
        if results.multi_hand_landmarks:
            # one hand
            if len(results.multi_hand_landmarks) == 1:
                # draw
                primary_index = extract_coordinates(results.multi_hand_landmarks[0], 8)
                if handle_gestures(results.multi_hand_landmarks[0]) == "gesture_one":
                    primary_coords.append([current_color, primary_index])
                else:
                    primary_coords.append(0)

                if handle_gestures(results.multi_hand_landmarks[0]) == "gesture_close":
                    primary_coords.clear()

            # two hands
            if len(results.multi_hand_landmarks) == 2:
                # draw
                primary_index = extract_coordinates(results.multi_hand_landmarks[1], 8)
                secondary_index = extract_coordinates(results.multi_hand_landmarks[0], 8)
                if handle_gestures(results.multi_hand_landmarks[1]) == "gesture_one":
                    primary_coords.append([current_color, primary_index])
                else:
                    primary_coords.append(0)


                # change color
                if not handle_gestures(results.multi_hand_landmarks[0]) == "gesture_two":
                    secondary_coords.append([current_color, secondary_index])

                # save image
                if handle_gestures(results.multi_hand_landmarks[0]) == "gesture_two" and \
                   handle_gestures(results.multi_hand_landmarks[1]) == "gesture_two":
                    cv2.imwrite(EXPORT_PATH, image)
                    image = cv2.putText(img = image, text="Image saved!", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), org=(20, 20), thickness=2)

                # clear lines
                if handle_gestures(results.multi_hand_landmarks[1]) == "gesture_close":
                    primary_coords.clear()


            # draw tracked hand(s)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # draw circle to show currently selected color
        image = cv2.circle(image, (int(CAPTURE_WIDTH - 50), int(CAPTURE_HEIGHT - 50)), 20, current_color, -1)
        image = cv2.circle(image, (int(CAPTURE_WIDTH - 50), int(CAPTURE_HEIGHT - 50)), 20, (0, 0, 0), 5)

        # extract color
        if len(secondary_coords) > 0:
            cropped = original_image[max(secondary_coords[-1][1][1] - 15, 15) : secondary_coords[-1][1][1], max(secondary_coords[-1][1][0] - 5, 1) : secondary_coords[-1][1][0] + 5]
            if(len(cropped) > 0):
                current_color = (int(cropped[0][0][0]), int(cropped[0][0][1]), int(cropped[0][0][2]))
            image = cv2.rectangle(image, (max(secondary_coords[-1][1][0] - 10, 1), max(secondary_coords[-1][1][1] - 10, 10)), (secondary_coords[-1][1][0] + 5, secondary_coords[-1][1][1]), color=(0, 0, 255), thickness=2)



        # render everything to the window
        cv2.imshow("Draw!", image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
