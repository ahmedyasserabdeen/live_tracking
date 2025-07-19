# YOLOv8 Live Person Tracking Application

A modern Flask application for **real-time person tracking** using YOLOv8. Upload a video, click on any detected person, and watch them being tracked live with smart bounding box highlighting!

## ✨ Features

### 🎯 **Smart Person Selection**
- **Automatic YOLO Detection**: AI automatically detects all persons in the first frame
- **Click-to-Select**: Simply click on any detected person to select them for tracking
- **Visual Feedback**: Selected person highlighted in green, others in blue
- **Confidence Display**: Shows detection confidence for each person

### 🎥 **Live Video Tracking**
- **Real-time Streaming**: Watch tracking happen live as the video plays
- **Persistent Highlighting**: Selected person stays highlighted in **GREEN** throughout the entire video
- **Multi-person Display**: All detected persons shown with unique tracking IDs
- **Performance Metrics**: Live FPS, frame count, and processing time display

### 📊 **Professional Interface**
- **Dark Theme**: Optimized interface for video viewing
- **Live Controls**: Start/stop tracking with real-time status indicators
- **Comprehensive Metrics**: Detailed performance and tracking analytics
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 **How It Works**

1. **Upload Video** → System extracts first frame
2. **Auto-Detection** → YOLO finds all persons automatically  
3. **Click Selection** → Click the person you want to track
4. **Live Tracking** → Watch real-time tracking with green highlighting
5. **Smart Following** → Selected person stays highlighted throughout video

## 🎬 **Demo Workflow**

```
Upload Video → YOLO Detects Persons → Click Target → Live Stream Tracking
     ↓                ↓                   ↓              ↓
   test.mp4    →   35 persons found  →  Person #5   →   GREEN box follows
```

## 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for better performance)
- Webcam or video files for testing

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-repository-name/yolov8-live-tracking.git
cd yolov8-live-tracking
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```env
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=runs/detect/track
FRAMES_FOLDER=static/frames
ALLOWED_EXTENSIONS=mp4,mov,avi
YOLO_MODEL_PATH=models/yolov8n.pt
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.6
```

### 4. Download YOLO models
```bash
# Models will be downloaded automatically on first run
# Or download manually to models/ folder
```

## 🎮 Usage

### Start the Application
```bash
python app.py
```
Open your browser to `http://localhost:5000`

### Live Tracking Workflow

#### Step 1: Upload Video
- Drag & drop or select video file (MP4, MOV, AVI)
- System automatically extracts first frame

#### Step 2: Person Detection
- YOLO automatically detects all persons in first frame
- Each person shown with blue bounding box
- Confidence percentage displayed for each detection

#### Step 3: Select Target
- Click on the person you want to track
- Selected person turns **GREEN**
- Selection details displayed (position, confidence, size)

#### Step 4: Live Tracking
- Click "Start Live Tracking"
- Watch real-time video stream
- **Your selected person stays GREEN throughout the entire video**
- Other persons remain blue with tracking IDs

#### Step 5: Control & Monitor
- Live performance metrics (FPS, frame count, time)
- Start/stop controls
- Status indicators (LIVE/STOPPED)

## 🏗️ Project Structure
```
yolov8-live-tracking/
├── app.py                      # Main Flask application with live tracking
├── api.py                      # REST API endpoints  
├── requirements.txt            # Python dependencies
├── .env                       # Configuration file
├── uploads/                   # Video uploads
├── static/
│   └── frames/               # Extracted frames for selection
├── runs/
│   └── detect/
│       └── track/            # Processed videos
├── templates/
│   ├── index.html           # Upload interface
│   ├── select_object.html   # Person selection with YOLO detection
│   ├── live_tracking.html   # Real-time tracking interface
│   └── tracking_results.html # Results summary
└── models/
    ├── yolov8n.pt           # YOLO model weights
    └── yolo11n.pt
```

## 🔧 API Endpoints

### Core Tracking Endpoints
```http
GET  /                          # Main upload page
POST /upload                    # Upload video and extract first frame
GET  /get_detections/<filename> # Get YOLO detections for first frame
POST /start_live_tracking       # Start live tracking session
POST /stop_live_tracking        # Stop live tracking session
GET  /video_feed               # Live video stream endpoint
GET  /live_tracking            # Live tracking interface
```

### Legacy Endpoints (Compatibility)
```http
POST /track_with_selection     # Traditional bounding box tracking
GET  /download/<filename>      # Download processed video
```

## 🎨 **Visual Features**

### Color Coding
- 🟢 **Green Box**: Your selected target person
- 🔵 **Blue Box**: Other detected persons  
- 🟡 **Yellow Text**: "TARGET FOUND" status indicator
- 🔴 **Red Text**: Error states or "SEARCHING..." status

### Live Display Elements
- **Frame Counter**: Current frame being processed
- **FPS Indicator**: Real-time processing speed
- **Target Status**: "TARGET FOUND" or "SEARCHING..."
- **Tracking IDs**: Unique identifier for each person
- **Confidence Scores**: Detection confidence percentage

## ⚙️ Configuration Options

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_FOLDER` | `uploads` | Directory for uploaded videos |
| `FRAMES_FOLDER` | `static/frames` | First frame extraction directory |
| `YOLO_MODEL_PATH` | `models/yolov8n.pt` | YOLO model file path |
| `CONFIDENCE_THRESHOLD` | `0.25` | Detection confidence threshold |
| `IOU_THRESHOLD` | `0.6` | Tracking IoU threshold |

### Performance Tuning
- **GPU Acceleration**: Automatically uses CUDA if available
- **Frame Rate**: ~30 FPS live streaming (adjustable)
- **Queue Management**: Smart frame buffering to prevent lag
- **Memory Optimization**: Efficient video processing pipeline

## 🔍 **Technical Implementation**

### YOLO Integration
- **Person Detection**: Uses class 0 (person) from YOLO
- **Real-time Processing**: Frame-by-frame analysis
- **Tracking Persistence**: Maintains IDs across frames
- **Confidence Filtering**: Configurable detection thresholds

### Live Streaming
- **Threading**: Background video processing
- **Frame Queue**: Buffered frame delivery
- **HTTP Streaming**: MJPEG video stream over HTTP
- **Error Handling**: Graceful connection recovery

### Target Tracking
- **IoU Matching**: Intelligent bounding box similarity
- **ID Persistence**: Consistent tracking across frames
- **Visual Highlighting**: Dynamic color coding
- **Status Updates**: Real-time tracking feedback

## 🐛 Troubleshooting

### Common Issues

**Video not loading in browser**
```bash
# Check if video format is supported
# Convert to MP4 if needed: ffmpeg -i input.mov output.mp4
```

**YOLO model not found**
```bash
# Download model manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
```

**Performance issues**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
# Reduce video resolution or adjust confidence threshold
```

**Browser compatibility**
- Use Chrome/Firefox for best performance
- Enable hardware acceleration in browser settings
- Clear browser cache if video doesn't load

## 📦 Dependencies
```
flask>=2.3.0
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
werkzeug>=2.3.0
python-dotenv>=1.0.0
numpy>=1.24.0
pillow>=10.0.0
```

## 🎯 **Advanced Usage**

### Custom Model Training
```python
# Train custom YOLO model for specific use cases
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='custom_dataset.yaml', epochs=100)
```

### Batch Processing
```python
# Process multiple videos
import os
for video in os.listdir('uploads/'):
    # Process video with custom parameters
    pass
```

### Integration with Other Services
- **REST API**: JSON responses for external integration
- **Webhook Support**: Real-time notifications
- **Database Storage**: Track processing history
- **Cloud Deployment**: Docker containerization ready

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 💡 **What's Next?**
- Multi-object tracking (track multiple persons simultaneously)
- Real-time analytics and alerts
- Mobile app integration
- Cloud-based processing
- Advanced tracking algorithms (DeepSORT, ByteTrack)

---

**🎯 Ready to track? Upload your video and start clicking! 🚀**

