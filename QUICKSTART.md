# Quick Start Guide

## Installation (3 minutes)

1. **Download the model**:
   - Go to https://github.com/hafizqaim/Workspace-Safety-Detection-using-YOLOv8/releases
   - Download `best.pt` (~6MB)
   - Place it in the project root directory

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**:
   - Navigate to `http://localhost:8501`
   - App loads automatically

## First Detection (1 minute)

1. **Upload an image**:
   - Click "Browse files"
   - Select a construction site image

2. **Adjust settings** (optional):
   - Confidence: 0.15 (default is good)
   - Inference Size: 1280 (default is good)

3. **View results**:
   - See detections on the right
   - Check statistics below
   - Detects helmets, vests, and other PPE

## Tips for Best Results

- **Good lighting**: Ensure PPE is clearly visible
- **Clear view**: Avoid heavy occlusion
- **Adjust confidence**: Lower to 0.10-0.12 if missing detections
- **Increase resolution**: Use 1920 for distant workers

## Model Information

- **Accuracy**: 86% for helmets, 73.5% mAP50 overall
- **Training**: 23,000+ images from PPE Detection v3 dataset
- **Detects**: Helmets, vests, and other safety equipment
- **Source**: 100% FREE, no API keys required

## Need Help?

- See [README.md](README.md) for full setup guide
- See [DOCUMENTATION.md](DOCUMENTATION.md) for technical details

---

**That's it! You're ready to detect PPE.** 🎉
