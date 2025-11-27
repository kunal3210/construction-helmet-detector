# Quick Start Guide

## Installation (2 minutes)

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser**:
   - Navigate to `http://localhost:8501`
   - Model will auto-download on first run (~6MB, takes ~30 seconds)

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

## Tips for Best Results

- **Good lighting**: Ensure helmets are clearly visible
- **Clear view**: Avoid heavy occlusion
- **Adjust confidence**: Lower to 0.12 if missing helmets
- **Increase resolution**: Use 1920 for distant workers

## Need Help?

- See [README.md](README.md) for full setup guide
- See [DOCUMENTATION.md](DOCUMENTATION.md) for technical details

---

**That's it! You're ready to detect helmets.** 🎉
