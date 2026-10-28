
import cv2
import time
from datetime import datetime
from pathlib import Path


def main():
    # Load face detection model
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        print("Error: Face detector could not be loaded.")
        return

    # Create screenshot folder
    screenshot_folder = Path("screenshots")
    screenshot_folder.mkdir(exist_ok=True)

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Real-Time Face Detection Started!")
    print("Press S to save a screenshot.")
    print("Press Q to quit.")

    previous_time = time.time()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Label each detected face
            for index, (x, y, w, h) in enumerate(faces, start=1):
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Face {index}",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - previous_time
            fps = 1 / elapsed if elapsed > 0 else 0
            previous_time = current_time

            # Display face count and FPS
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Real-Time Face Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                filename = (
                    screenshot_folder / f"face_{timestamp}.jpg"
                )

                if cv2.imwrite(str(filename), frame):
                    print(f"Screenshot saved: {filename}")
                else:
                    print("Error: Could not save screenshot.")

            elif key == ord("q"):
                print("Face detection stopped.")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
