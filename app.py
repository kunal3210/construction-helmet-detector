import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detector import HelmetDetector
import tempfile

# Page Config
st.set_page_config(
    page_title="Safety Helmet Detection",
    page_icon="👷",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    return HelmetDetector()

def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")
    
    mode = st.sidebar.radio("Select Mode", ["Image Upload", "Live Webcam", "Video File"])
    
    st.sidebar.markdown("### Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15, 0.05, 
                                       help="Lower values detect more objects but may include false positives")
    img_size = st.sidebar.select_slider("Inference Size (Resolution)", options=[320, 640, 1280, 1920], value=1280,
                                        help="Higher resolution = better accuracy for small objects")
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05,
                                  help="Lower values allow overlapping detections")
        max_det = st.slider("Max Detections", 10, 300, 100, 10,
                           help="Maximum number of detections per image")
    
    st.sidebar.markdown("---")
    st.sidebar.success("✓ Using Construction Helmet Detection Model")
    st.sidebar.info("Model: sharathhhhh/safetyHelmet-detection-yolov8 (FREE)")
    st.sidebar.markdown("**Detects:** with_helmet, without_helmet")

    # Main Content
    st.title("👷 Construction Helmet Detection System")
    st.markdown("Real-time detection of construction safety helmets using YOLOv8.")

    detector = load_detector()

    if mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            image = Image.open(uploaded_file)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with st.spinner("Detecting..."):
                # Convert PIL to numpy for OpenCV if needed, but YOLO handles PIL
                annotated_image, detections = detector.detect(image, conf_threshold, imgsz=img_size, iou=iou_threshold, max_det=max_det)
                
                # Convert BGR to RGB for Streamlit display if it came back as BGR
                if isinstance(annotated_image, np.ndarray):
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Detection Result")
                st.image(annotated_image, use_container_width=True)
                
            # Statistics
            st.markdown("### 📊 Detection Statistics")
            helmet_count = sum(1 for d in detections if 'helmet' in d['class'] and 'no_helmet' not in d['class'])
            no_helmet_count = sum(1 for d in detections if 'no_helmet' in d['class'])
            
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Total Detections", len(detections))
            kpi2.metric("With Helmet", helmet_count)
            kpi3.metric("Without Helmet", no_helmet_count, delta_color="inverse")

    elif mode == "Live Webcam":
        st.warning("Ensure your webcam is enabled.")
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        
        camera = cv2.VideoCapture(0)
        
        while run:
            _, frame = camera.read()
            if frame is None:
                st.error("Could not access webcam.")
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            annotated_frame, _ = detector.detect(frame, conf_threshold, imgsz=img_size, iou=iou_threshold, max_det=max_det)
            
            # Display
            FRAME_WINDOW.image(annotated_frame)
        
        camera.release()

    elif mode == "Video File":
        uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            vf = cv2.VideoCapture(tfile.name)
            
            stframe = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_frame, _ = detector.detect(frame, conf_threshold, imgsz=img_size, iou=iou_threshold, max_det=max_det)
                stframe.image(annotated_frame)
            
            vf.release()

if __name__ == "__main__":
    main()
