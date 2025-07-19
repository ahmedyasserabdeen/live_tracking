import os
import shutil
import time
import cv2
import numpy as np
import json
import threading
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import queue
import base64

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Define paths and configurations from .env file
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'runs/detect/track')
FRAMES_FOLDER = os.getenv('FRAMES_FOLDER', 'static/frames')
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'mp4,mov,avi').split(','))

# Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.6))
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/yolov8n.pt')

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

# Function to extract first frame from video
def extract_first_frame(video_path, output_path):
    """Extract the first frame from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        cap.release()
        return True
    cap.release()
    return False

# Global variables for live tracking
current_video_path = None
tracking_active = False
current_target_id = None
frame_queue = queue.Queue(maxsize=10)
video_ended = False
video_status_message = ""
current_frame_count = 0
total_video_frames = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Create frames folder if it doesn't exist
        os.makedirs(FRAMES_FOLDER, exist_ok=True)
        
        # Extract first frame for bounding box selection
        frame_filename = f"{os.path.splitext(filename)[0]}_first_frame.jpg"
        frame_path = os.path.join(FRAMES_FOLDER, frame_filename)
        
        if extract_first_frame(filepath, frame_path):
            return render_template('select_object.html', 
                                 video_filename=filename,
                                 frame_filename=frame_filename)
        else:
            return "Error extracting first frame", 400

    return redirect(url_for('index'))

@app.route('/select_object')
def select_object():
    video_filename = request.args.get('video_filename')
    frame_filename = request.args.get('frame_filename')
    if not video_filename or not frame_filename:
        return redirect(url_for('index'))
    
    return render_template('select_object.html', 
                         video_filename=video_filename,
                         frame_filename=frame_filename)

@app.route('/get_detections/<video_filename>')
def get_detections(video_filename):
    """Get YOLO detections for the first frame"""
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    # Get the first frame
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'error': 'Could not read video'}), 400
    
    # Run YOLO detection
    model = YOLO(YOLO_MODEL_PATH)
    results = model(first_frame, classes=[0], conf=CONFIDENCE_THRESHOLD)  # class 0 = person
    
    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
        confidences = results[0].boxes.conf.cpu().numpy()
        
        frame_height, frame_width = first_frame.shape[:2]
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            detections.append({
                'id': i,
                'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # [x, y, width, height]
                'confidence': float(conf),
                'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            })
    
    return jsonify({
        'detections': detections,
        'frame_dimensions': {'width': frame_width, 'height': frame_height}
    })

@app.route('/track_with_selection', methods=['POST'])
def track_with_selection():
    data = request.get_json()
    video_filename = data.get('video_filename')
    bounding_box = data.get('bounding_box')  # [x, y, width, height]
    
    if not video_filename or not bounding_box:
        return jsonify({'error': 'Missing video filename or bounding box'}), 400
    
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404

    # Clear and recreate the output folder before processing
    clear_and_create_output_folder()

    # Process the video with YOLOv8 using the selected bounding box
    start_time = time.time()
    model = YOLO(YOLO_MODEL_PATH)
    
    # Get the first frame to find the target object
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'error': 'Could not read video'}), 400
    
    # Convert bounding box to YOLO format if needed
    # The bounding box from frontend is [x, y, width, height] in pixels
    x, y, w, h = bounding_box
    
    # Run detection on first frame to find objects
    results = model(first_frame, classes=[0], conf=CONFIDENCE_THRESHOLD)  # class 0 = person
    
    # Find the object that best matches the selected bounding box
    best_match = None
    best_iou = 0
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
        
        for i, box in enumerate(boxes):
            # Convert to [x, y, width, height] format
            box_x, box_y, box_w, box_h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            
            # Calculate IoU
            intersection_x = max(x, box_x)
            intersection_y = max(y, box_y)
            intersection_w = min(x + w, box_x + box_w) - intersection_x
            intersection_h = min(y + h, box_y + box_h) - intersection_y
            
            if intersection_w > 0 and intersection_h > 0:
                intersection_area = intersection_w * intersection_h
                union_area = w * h + box_w * box_h - intersection_area
                iou = intersection_area / union_area if union_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
    
    # Now track the video with the selected object
    results = model.track(
        source=video_path,
        save=True,
        classes=[0],  # Track only the "person" class
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        project='runs/detect',
        name='track',
        exist_ok=True,
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
                    continuity_issues += 1

    # Calculate FPS
    total_time = end_time - start_time
    fps = total_frames / total_time if total_time > 0 else 0

    # Prepare metrics for display
    output_filename = f"{os.path.splitext(video_filename)[0]}.avi"
    
    metrics = {
        'total_time': f"{total_time:.2f}",
        'total_frames': total_frames,
        'fps': f"{fps:.2f}",
        'continuity_issues': continuity_issues,
        'output_video': output_filename,
        'selected_object_found': best_match is not None,
        'selection_confidence': f"{best_iou:.2f}" if best_match is not None else "0.00"
    }

    return render_template('tracking_results.html', metrics=metrics)

@app.route('/start_live_tracking', methods=['POST'])
def start_live_tracking():
    """Start live tracking with selected detection"""
    global current_video_path, tracking_active, current_target_id, video_ended, video_status_message, current_frame_count, total_video_frames
    
    data = request.get_json()
    video_filename = data.get('video_filename')
    detection_id = data.get('detection_id')
    
    if not video_filename or detection_id is None:
        return jsonify({'error': 'Missing video filename or detection ID'}), 400
    
    # Convert detection_id to integer
    try:
        detection_id = int(detection_id)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid detection ID'}), 400
    
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    # Reset video status variables
    video_ended = False
    video_status_message = ""
    current_frame_count = 0
    total_video_frames = 0
    
    current_video_path = video_path
    current_target_id = detection_id
    tracking_active = True
    
    # Start tracking in a separate thread
    tracking_thread = threading.Thread(target=live_tracking_worker)
    tracking_thread.daemon = True
    tracking_thread.start()
    
    return jsonify({'status': 'success', 'message': 'Live tracking started'})

@app.route('/stop_live_tracking', methods=['POST'])
def stop_live_tracking():
    """Stop live tracking"""
    global tracking_active, video_ended, video_status_message, current_frame_count, total_video_frames
    tracking_active = False
    video_ended = False
    video_status_message = ""
    current_frame_count = 0
    total_video_frames = 0
    return jsonify({'status': 'success', 'message': 'Live tracking stopped'})

@app.route('/get_tracking_status')
def get_tracking_status():
    """Get current tracking status"""
    global tracking_active, video_ended, video_status_message, current_frame_count, total_video_frames
    return jsonify({
        'tracking_active': tracking_active,
        'video_ended': video_ended,
        'status_message': video_status_message,
        'current_frame': current_frame_count,
        'total_frames': total_video_frames
    })

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def live_tracking_worker():
    """Worker function for live tracking in separate thread"""
    global current_video_path, tracking_active, current_target_id, frame_queue, video_ended, video_status_message, current_frame_count, total_video_frames
    
    if not current_video_path:
        return
    
    model = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(current_video_path)
    
    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_video_frames = total_frames  # Update global variable
    
    # Get the first frame to find initial target
    ret, first_frame = cap.read()
    if not ret:
        tracking_active = False
        video_ended = True
        video_status_message = "Error: Could not read video file"
        return
    
    # Get detections on first frame to find the target
    results = model(first_frame, classes=[0], conf=CONFIDENCE_THRESHOLD)
    target_bbox = None
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if current_target_id < len(boxes):
            target_box = boxes[current_target_id]
            target_bbox = [int(target_box[0]), int(target_box[1]), 
                          int(target_box[2] - target_box[0]), int(target_box[3] - target_box[1])]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    frame_count = 0
    target_tracking_id = None  # Store the tracking ID of our target person
    
    while tracking_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Video has ended
            video_ended = True
            video_status_message = f"Video completed! Processed {frame_count} frames out of {total_frames} total frames."
            tracking_active = False
            print(f"Video ended: {video_status_message}")
            break
        
        # Run tracking
        results = model.track(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, persist=True)
        
        # Draw tracking results
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # On first frame, identify the target person's tracking ID
                if frame_count == 0 and target_bbox and target_tracking_id is None:
                    current_bbox = [x1, y1, x2-x1, y2-y1]
                    if bbox_similarity(target_bbox, current_bbox) > 0.5:
                        target_tracking_id = track_id  # Store the target's tracking ID
                        print(f"Target person assigned tracking ID: {target_tracking_id}")
                
                # Highlight the target person in green throughout the entire video
                if target_tracking_id is not None and track_id == target_tracking_id:
                    color = (0, 255, 0)  # Green for target person
                    thickness = 3
                    label = f"TARGET ID:{int(track_id)} {conf:.2f}"
                else:
                    color = (255, 0, 0)  # Blue for other persons
                    thickness = 2
                    label = f"ID:{int(track_id)} {conf:.2f}"
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness-1)
        
        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add target status info
        target_status = "TARGET FOUND" if target_tracking_id is not None else "SEARCHING..."
        cv2.putText(annotated_frame, target_status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if target_tracking_id is not None else (0, 255, 255), 2)
        
        # Put frame in queue for streaming
        if not frame_queue.full():
            try:
                frame_queue.put_nowait(annotated_frame)
            except queue.Full:
                pass
        
        frame_count += 1
        current_frame_count = frame_count  # Update global frame count
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    
    # If video ended naturally (not manually stopped), add a final message frame
    if video_ended and frame_count > 0:
        # Create a final frame with completion message
        final_frame = np.zeros((400, 800, 3), dtype=np.uint8)
        cv2.putText(final_frame, "VIDEO COMPLETED!", (200, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(final_frame, f"Processed {frame_count} frames", (250, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(final_frame, "Thank you for using our tracker!", (180, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Put final frame in queue
        if not frame_queue.full():
            try:
                frame_queue.put_nowait(final_frame)
            except queue.Full:
                pass
    
    tracking_active = False
    print(f"Live tracking worker finished. Video ended: {video_ended}")

def generate_frames():
    """Generate frames for video streaming"""
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get_nowait()
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)
        except queue.Empty:
            time.sleep(0.01)

def bbox_similarity(bbox1, bbox2):
    """Calculate similarity between two bounding boxes using IoU"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

@app.route('/live_tracking')
def live_tracking_page():
    """Live tracking page"""
    video_filename = request.args.get('video_filename')
    detection_id = request.args.get('detection_id')
    
    if not video_filename or detection_id is None:
        return redirect(url_for('index'))
    
    # Convert detection_id to integer
    try:
        detection_id = int(detection_id)
    except (ValueError, TypeError):
        return redirect(url_for('index'))
    
    return render_template('live_tracking.html', 
                         video_filename=video_filename,
                         detection_id=detection_id)

# Route to serve the processed video
@app.route('/download/<filename>')
def download_video(filename):
    # Check if file exists in the expected output folder
    expected_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(expected_path):
        return send_from_directory(OUTPUT_FOLDER, filename)
    
    # If not found, check the predict folder (fallback)
    predict_folder = 'runs/detect/predict'
    predict_path = os.path.join(predict_folder, filename)
    if os.path.exists(predict_path):
        # Move the file to the correct location
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        shutil.move(predict_path, expected_path)
        return send_from_directory(OUTPUT_FOLDER, filename)
    
    # File not found in either location
    return "Video file not found", 404

# Route to serve static frames
@app.route('/static/frames/<filename>')
def serve_frame(filename):
    return send_from_directory(FRAMES_FOLDER, filename)

if __name__ == '__main__':
    # Run the application
    app.run(debug=True)
