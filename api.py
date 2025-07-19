import os
import shutil
import time
from flask import Flask, request, jsonify, send_from_directory,url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load configuration from environment variables
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'runs/detect/track')
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'mp4,mov,avi').split(','))
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/yolov8n.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.6))

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to clear and recreate the output folder with error handling
def clear_and_create_output_folder():
    if os.path.exists(OUTPUT_FOLDER):
        try:
            # Try to remove all files and subdirectories
            for filename in os.listdir(OUTPUT_FOLDER):
                file_path = os.path.join(OUTPUT_FOLDER, filename)
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                except PermissionError:
                    print(f"Permission error while deleting {file_path}. Skipping...")
            # Remove the directory itself
            os.rmdir(OUTPUT_FOLDER)
        except Exception as e:
            print(f"Error clearing folder {OUTPUT_FOLDER}: {e}")
    
    # Recreate the output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Clear and recreate the output folder before processing
        clear_and_create_output_folder()

        # Process the video with YOLOv8
        start_time = time.time()
        model = YOLO(YOLO_MODEL_PATH)  # Load YOLO model from the .env path
        results = model.track(
            source=filepath,  # Input video path
            save=True,
            classes=[0],  # Track only the "person" class
            conf=CONFIDENCE_THRESHOLD,  # Confidence threshold from .env
            iou=IOU_THRESHOLD,  # IoU threshold from .env
            stream=True
        )
        end_time = time.time()

        # Initialize metrics
        total_frames = 0
        tracked_ids = set()
        continuity_issues = 0

        # Parse results
        for result in results:
            total_frames += 1
            if result.boxes and result.boxes.id is not None:
                ids = result.boxes.id.cpu().tolist()
                for track_id in ids:
                    if track_id not in tracked_ids:
                        tracked_ids.add(track_id)
                    else:
                        continuity_issues += 1  # Simplified placeholder for continuity issues

        # Calculate FPS
        total_time = end_time - start_time
        fps = total_frames / total_time if total_time > 0 else 0

        # Prepare metrics
        output_filename = f"{os.path.splitext(filename)[0]}.avi"  # Ensure output is .avi
        output_video_path = os.path.join(OUTPUT_FOLDER, output_filename)

        metrics = {
            'total_time': f"{total_time:.2f}",
            'total_frames': total_frames,
            'fps': f"{fps:.2f}",
            'output_video': output_filename  # Pass only the filename, not the full path
        }

        # Return the processed video info and metrics as JSON
        return jsonify({
            "message": "Video processed successfully.",
            "metrics": metrics,
            "download_link": url_for('download_video', filename=output_filename, _external=True)
        }), 200

    return jsonify({"error": "Invalid file format. Allowed formats: mp4, mov, avi."}), 400

@app.route('/api/download/<filename>', methods=['GET'])
def download_video(filename):
    # Ensure to serve from the right directory
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
