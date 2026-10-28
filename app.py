
from flask import Flask, render_template, Response, jsonify, send_from_directory
from pathlib import Path
from datetime import datetime
import cv2
import threading
import time

app = Flask(__name__)

# Screenshot storage
SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

# Camera state
camera = None
camera_lock = threading.Lock()
camera_running = False

face_count = 0
fps = 0.0
latest_frame = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start_camera", methods=["POST"])
def start_camera():
    global camera, camera_running, latest_frame
    global face_count, fps

    with camera_lock:
        if camera_running and camera is not None:
            return jsonify({
                "status": "started",
                "message": "Camera is already running."
            })

        new_camera = cv2.VideoCapture(0)

        if not new_camera.isOpened():
            new_camera.release()
            return jsonify({
                "status": "error",
                "message": "Camera could not be opened."
            }), 500

        camera = new_camera
        camera_running = True
        latest_frame = None
        face_count = 0
        fps = 0.0

    return jsonify({
        "status": "started",
        "message": "Camera started successfully."
    })


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global camera, camera_running, latest_frame
    global face_count, fps

    with camera_lock:
        camera_running = False

        if camera is not None:
            camera.release()
            camera = None

        latest_frame = None
        face_count = 0
        fps = 0.0

    return jsonify({
        "status": "stopped",
        "message": "Camera stopped."
    })


def generate_frames():
    global face_count, fps, latest_frame

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    previous_time = time.time()

    while True:
        with camera_lock:
            if not camera_running or camera is None:
                break

            success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_count = len(faces)

        for number, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Face {number}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        current_time = time.time()
        elapsed = current_time - previous_time

        if elapsed > 0:
            fps = round(1 / elapsed, 1)

        previous_time = current_time

        with camera_lock:
            latest_frame = frame.copy()

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def stats():
    with camera_lock:
        current_faces = face_count
        current_fps = fps
        running = camera_running

    return jsonify({
        "faces": current_faces,
        "fps": current_fps,
        "camera": "Online" if running else "Offline"
    })


@app.route("/save_screenshot", methods=["POST"])
def save_screenshot():
    with camera_lock:
        if not camera_running or camera is None:
            return jsonify({
                "status": "error",
                "message": "Please start the camera first."
            }), 400

        if latest_frame is None:
            return jsonify({
                "status": "error",
                "message": "Camera frame is not ready. Try again."
            }), 503

        frame_to_save = latest_frame.copy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"face_{timestamp}.jpg"
    file_path = SCREENSHOT_DIR / filename

    saved = cv2.imwrite(str(file_path), frame_to_save)

    if not saved:
        return jsonify({
            "status": "error",
            "message": "Could not save screenshot."
        }), 500

    return jsonify({
        "status": "success",
        "message": f"Screenshot saved: screenshots/{filename}",
        "filename": filename
    })


# Screenshot Gallery
@app.route("/gallery")
def gallery():
    images = sorted(
        SCREENSHOT_DIR.glob("*.jpg"),
        key=lambda file: file.stat().st_mtime,
        reverse=True
    )

    return render_template(
        "gallery.html",
        images=[image.name for image in images]
    )


# Download an individual screenshot
@app.route("/download/<filename>")
def download_screenshot(filename):
    if (
        not filename.lower().endswith(".jpg")
        or Path(filename).name != filename
    ):
        return "Invalid file type", 400

    file_path = (SCREENSHOT_DIR / filename).resolve()

    if (
        file_path.parent != SCREENSHOT_DIR.resolve()
        or not file_path.is_file()
    ):
        return "Screenshot not found", 404

    return send_from_directory(
        SCREENSHOT_DIR,
        filename,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False
    )
