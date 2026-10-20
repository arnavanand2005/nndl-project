import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:

    def __init__(self, model_path="models/hand_landmarker.task"):

        # Resolve model path relative to project root
        if not os.path.isabs(model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found:\n{model_path}"
            )

        print(f"Loading MediaPipe model:")
        print(model_path)

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(
            options
        )

        print("MediaPipe Hand Landmarker loaded successfully.")

    def detect(self, frame, timestamp_ms):
        """
        Detect one hand from an OpenCV BGR frame.

        Returns:
            MediaPipe HandLandmarkerResult
        """

        if frame is None:
            return None

        # OpenCV uses BGR
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    def get_landmarks(self, result):
        """
        Extract the first detected hand.

        Returns:
            List of 21 MediaPipe landmarks,
            or None if no hand was detected.
        """

        if result is None:
            return None

        if not result.hand_landmarks:
            return None

        return result.hand_landmarks[0]

    def get_landmark_array(self, result):
        """
        Convert the first detected hand into 63 raw features.

        Order:
            x1, y1, z1,
            x2, y2, z2,
            ...
            x21, y21, z21

        Returns:
            List containing 63 values,
            or None if no hand was detected.
        """

        hand = self.get_landmarks(result)

        if hand is None:
            return None

        features = []

        for landmark in hand:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return features

    def draw_landmarks(self, frame, result):

        if frame is None:
            return frame

        if result is None:
            return frame

        if not result.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        # MediaPipe hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),

            (0, 5), (5, 6), (6, 7), (7, 8),

            (0, 9), (9, 10), (10, 11), (11, 12),

            (0, 13), (13, 14), (14, 15), (15, 16),

            (0, 17), (17, 18), (18, 19), (19, 20),

            (5, 9),
            (9, 13),
            (13, 17)
        ]

        for hand in result.hand_landmarks:

            points = []

            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Keep points inside the image
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))

                points.append((x, y))

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

        return frame

    def close(self):
        """
        Release MediaPipe resources.
        """

        if self.detector is not None:
            self.detector.close()
            self.detector = None