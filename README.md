# Construction Helmet Detection System

A production-ready real-time PPE detection application using YOLOv8 and Streamlit for construction site safety monitoring.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

## 🎯 Overview

This application uses a specialized YOLOv8 model to detect construction safety helmets and other PPE in real-time from images, videos, or live webcam feeds. It's designed for construction site safety monitoring and compliance checking.

## ✨ Features

- **Multi-Input Support**: Process images, video files, or live webcam streams
- **Real-time Detection**: Fast inference with adjustable resolution settings
- **Advanced Controls**: Fine-tune confidence thresholds, IOU, and detection limits
- **Free & Open Source**: Uses free models from GitHub, no API keys required
- **Modern UI**: Clean, intuitive interface built with Streamlit
- **Production Ready**: Optimized for deployment and real-world usage
- **Multi-PPE Detection**: Detects helmets, vests, and other safety equipment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (optional, for live detection)

### Installation

1. **Clone or download this repository**

2. **Download the model**:
   - Get `best.pt` from [hafizqaim/Workspace-Safety-Detection releases](https://github.com/hafizqaim/Workspace-Safety-Detection-using-YOLOv8/releases)
   - Place it in the project root directory

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the app**:
   - Open your browser and navigate to `http://localhost:8501`

## 📖 Usage

### Image Upload Mode
1. Select "Image Upload" from the sidebar
2. Adjust detection settings (see Settings Guide below)
3. Upload an image (JPG, JPEG, PNG)
4. View results with detection statistics

### Live Webcam Mode
1. Select "Live Webcam"
2. Click "Start Webcam"
3. Real-time detection will begin
4. Uncheck to stop

### Video File Mode
1. Select "Video File"
2. Upload a video (MP4, AVI, MOV)
3. Watch frame-by-frame detection

## ⚙️ Settings Guide

### Basic Settings

**Confidence Threshold** (Default: 0.15)
- Range: 0.0 - 1.0
- Lower values (0.10-0.15): Detect more objects, may include false positives
- Higher values (0.25-0.40): Only detect very confident detections
- **Recommended**: Start at 0.15, adjust based on your needs

**Inference Size** (Default: 1280)
- Options: 320, 640, 1280, 1920 pixels
- Higher resolution = better accuracy for small/distant objects
- Higher resolution = slower processing
- **Recommended**: 1280 for balanced performance, 1920 for maximum accuracy

### Advanced Settings

**IOU Threshold** (Default: 0.45)
- Range: 0.0 - 1.0
- Controls overlap tolerance for detections
- Lower values (0.30-0.40): Allow more overlapping detections (good for crowded scenes)
- Higher values (0.50-0.70): Stricter filtering of overlapping boxes
- **Recommended**: 0.35-0.45 for construction sites

**Max Detections** (Default: 100)
- Range: 10 - 300
- Maximum number of objects to detect per image
- **Recommended**: Keep at 100 unless processing very large crowds

## 🎛️ Optimal Settings for Different Scenarios

### Crowded Construction Site (10+ people)
```
Confidence Threshold: 0.12
Inference Size: 1280
IOU Threshold: 0.35
Max Detections: 100
```

### Distant Workers (Far from camera)
```
Confidence Threshold: 0.15
Inference Size: 1920
IOU Threshold: 0.45
Max Detections: 100
```

### Real-time Webcam (Performance priority)
```
Confidence Threshold: 0.20
Inference Size: 640
IOU Threshold: 0.45
Max Detections: 50
```

## 🤖 Model Information

- **Model**: hafizqaim/Workspace-Safety-Detection
- **Source**: GitHub (100% FREE, no API keys required)
- **Architecture**: YOLOv8
- **Training Data**: 23,000+ images from PPE Detection v3 dataset (Roboflow)
- **Classes Detected**: 
  - Safety helmets (head_helmet)
  - Safety vests (vest)
  - Other PPE equipment
- **Accuracy**: 
  - 86% for helmets
  - 73.5% mAP50 overall
- **Size**: ~6MB

### Why This Model?
- Trained on large, diverse dataset (23K+ images)
- Higher accuracy than standard models (86% vs 85%)
- Detects multiple PPE items, not just helmets
- Optimized for construction site scenarios
- Handles various lighting conditions and angles
- Better performance on occluded and distant objects

## 📁 Project Structure

```
helmet-detection/
├── .gitignore              # Git ignore rules
├── QUICKSTART.md           # Quick start guide (1-2 min setup)
├── README.md               # Main documentation (this file)
├── DOCUMENTATION.md        # Technical details (architecture, API, tuning)
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit application (main entry point)
├── detector.py             # YOLOv8 detector class
└── best.pt                 # Pre-trained PPE detection model weights (6MB)
```

## 🔧 Technical Requirements

### Python Packages
- `ultralytics`: YOLOv8 implementation
- `streamlit`: Web interface
- `opencv-python-headless`: Image/video processing
- `Pillow`: Image handling

### System Requirements
- **RAM**: Minimum 4GB, 8GB recommended
- **Storage**: 500MB for dependencies + model
- **CPU**: Any modern processor (GPU optional but recommended for video processing)

## 🚀 Deployment

### Local Deployment
Already covered in Quick Start section.

### Cloud Deployment (Streamlit Cloud)
1. Push this repository to GitHub
2. Upload `best.pt` to GitHub (or use Git LFS for large files)
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Deploy from your GitHub repository
5. The app will be publicly accessible

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t helmet-detection .
docker run -p 8501:8501 helmet-detection
```

## 🐛 Troubleshooting

### Model not found
- Ensure `best.pt` is in the project directory
- Download from: https://github.com/hafizqaim/Workspace-Safety-Detection-using-YOLOv8/releases

### Webcam not working
- Ensure webcam permissions are granted
- Check if another application is using the webcam
- Try a different browser

### Low detection accuracy
- Lower the confidence threshold to 0.10-0.15
- Increase inference size to 1280 or 1920
- Ensure good lighting in images/videos
- Check that helmets/PPE are clearly visible (not occluded)

### Slow performance
- Reduce inference size to 640
- Lower max detections
- Use GPU if available (install `torch` with CUDA support)

## 📊 Performance Metrics

- **Inference Speed**: 
  - 640px: ~50-100ms per image
  - 1280px: ~200-400ms per image
  - 1920px: ~500-800ms per image
- **Accuracy**: 
  - Helmets: 86%
  - Overall mAP50: 73.5%
- **False Positives**: <5% with confidence threshold 0.20+

## 🤝 Contributing

This is a production application. For improvements:
1. Test thoroughly before proposing changes
2. Maintain backward compatibility
3. Update documentation accordingly

## 📄 License

This project uses open-source models and libraries. Please check individual component licenses:
- YOLOv8: AGPL-3.0
- Model: Check GitHub repository
- Streamlit: Apache 2.0

## 🙏 Credits

- **Model**: [hafizqaim/Workspace-Safety-Detection](https://github.com/hafizqaim/Workspace-Safety-Detection-using-YOLOv8)
- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **UI**: [Streamlit](https://streamlit.io/)
- **Dataset**: PPE Detection v3 from Roboflow

## 📞 Support

For detailed technical information, see [DOCUMENTATION.md](DOCUMENTATION.md)

---

**Version**: 2.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready
