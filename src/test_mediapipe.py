import cv2
import time

from hand_detector import HandDetector


def main():

    detector = HandDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Starting webcam...")
    print("Press Q to quit.")

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        result = detector.detect(
            frame,
            timestamp_ms
        )

        frame = detector.draw_landmarks(
            frame,
            result
        )

        # Status
        if result.hand_landmarks:
            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "NO HAND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "ASL Hand Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    detector.close()

    print("Webcam closed.")


if __name__ == "__main__":
    main()