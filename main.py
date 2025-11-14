import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import random
import gradio as gr
from pathlib import Path
import ffmpy

IMAGE_SIZE = (112, 112)
NUM_FRAMES_PER_VIDEO = 32

CLASS_NAMES = [
    "original",
    "forgery_insertion",
    "forgery_deletion",
    "forgery_duplication",
    "forgery_horizontal_flipping",
    "forgery_vertikal_flipping",
    "forgery_other_rotation",
    "forgery_zooming"
]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_WEIGHTS_PATH = "./model/best_video_resnet_multi_class_model.pth"

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Gradio app using device: {DEVICE}")

model = None
video_transform = None
idx_to_class_mapping = None

def load_model_and_transforms():
    """
    Memuat model dan menyiapkan transformasi. Fungsi ini akan dipanggil sekali.
    """
    global model, video_transform, idx_to_class_mapping

    print(f"Memuat model dari {MODEL_WEIGHTS_PATH}...")
    try:
        # Buat kembali arsitektur model
        model = models.video.r2plus1d_18(weights=None)
        num_ftrs_inf = model.fc.in_features
        model.fc = nn.Linear(num_ftrs_inf, NUM_CLASSES)

        # Muat state_dict yang sudah dilatih
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        print("Model berhasil dimuat.")

        # Define transformations
        video_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Transformasi video diinisialisasi.")

        # Re-create the idx_to_class mapping
        idx_to_class_mapping = {i: name for i, name in enumerate(CLASS_NAMES)}
        print("Class mapping initialized.")

    except Exception as e:
        print(f"ERROR: Could not load model or initialize: {e}")
        model = None

# Panggil fungsi loading sekali saat script dijalankan
load_model_and_transforms()

def video_to_tensor_conversion(video_path: str, transform, num_frames_to_sample: int, device: torch.device):
    """
    Converts a video file to a PyTorch tensor suitable for model inference.
    Supports: MP4, AVI, MOV, FLV, WMV, MKV, etc.
    Returns: (video_tensor, metadata, processed_video_path)
    """
    processed_video_path = video_path
    supported_formats = ['.avi', '.mov', '.flv', '.wmv', '.mkv', '.webm', '.m4v', '.3gp']
    
    # Konversi format non-MP4 ke MP4 dengan H.264
    if Path(video_path).suffix.lower() in supported_formats:
        mp4_path = str(Path(video_path).with_suffix('.mp4'))
        
        try:
            print(f"🔄 Converting {Path(video_path).suffix.upper()} to MP4 for browser compatibility...")
            ff = ffmpy.FFmpeg(
                inputs={video_path: None},
                outputs={mp4_path: [
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-loglevel', 'quiet',
                    '-y'
                ]}
            )
            ff.run()
            print(f"✅ Video converted successfully to: {mp4_path}")
            processed_video_path = mp4_path
        except Exception as e:
            print(f"⚠️  FFmpeg conversion failed: {e}. Attempting to read original file...")
            # Coba baca file asli tanpa konversi
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    cap.release()
                    processed_video_path = video_path
                    print("✅ Original file can be read directly")
                else:
                    return None, {"error": f"Could not read video: {Path(video_path).suffix}"}, None
            except:
                return None, {"error": "Video conversion failed and original file is unreadable"}, None
    
    # Jika sudah MP4 tapi mungkin codec-nya bukan H.264, re-encode
    elif Path(video_path).suffix.lower() == '.mp4':
        # Re-encode untuk memastikan browser-compatible
        temp_path = str(Path(video_path).with_name(Path(video_path).stem + '_converted.mp4'))
        try:
            print("🔄 Re-encoding MP4 to ensure browser compatibility...")
            ff = ffmpy.FFmpeg(
                inputs={video_path: None},
                outputs={temp_path: [
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-loglevel', 'quiet',
                    '-y'
                ]}
            )
            ff.run()
            processed_video_path = temp_path
            print(f"✅ MP4 re-encoded to: {temp_path}")
        except Exception as e:
            print(f"⚠️  Re-encoding warning: {e}. Using original file...")
            processed_video_path = video_path

    cap = cv2.VideoCapture(processed_video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {processed_video_path}")
        return None, {"error": f"Could not open video file: {Path(video_path).name}"}, None

    # Extract metadata
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = frame_count / fps if fps > 0 else 0

    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration_seconds
    }

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"⚠️  Warning: No frames extracted from {processed_video_path}.")
        return None, metadata, processed_video_path

    # Sample frames
    selected_frames = []
    if len(frames) >= num_frames_to_sample:
        indices = np.linspace(0, len(frames) - 1, num_frames_to_sample, dtype=int)
        selected_frames = [frames[i] for i in indices]
    else:
        selected_frames = frames
        while len(selected_frames) < num_frames_to_sample:
            selected_frames.append(random.choice(frames))
        selected_frames = selected_frames[:num_frames_to_sample]

    transformed_frames = [transform(frame) for frame in selected_frames]
    video_tensor = torch.stack(transformed_frames, dim=1)  # (C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)
    
    return video_tensor.to(device), metadata, processed_video_path

def predict_forgery_gradio(video_file_path: str):
    """
    Gradio-compatible function to predict forgery and extract metadata.
    Returns: (prob_dict, metadata_html, processed_video_path)
    """
    if model is None or video_transform is None or idx_to_class_mapping is None:
        error_html = "<div style='color: red;'><b>❌ Error:</b> Model not loaded. Please check server logs.</div>"
        return {"Error": 1.0}, error_html, None

    if not video_file_path:
        error_html = "<div style='color: orange;'><b>⚠️  Warning:</b> Please upload a video file.</div>"
        return {"Input Error": 1.0}, error_html, None

    print(f"📽️  Received video for prediction: {video_file_path}")

    try:
        video_tensor, metadata, processed_video_path = video_to_tensor_conversion(
            video_file_path, video_transform, NUM_FRAMES_PER_VIDEO, DEVICE
        )

        if video_tensor is None:
            error_msg = metadata.get("error", "Could not process video file") if metadata else "Unknown error"
            error_html = f"<div style='color: red;'><b>❌ Error:</b> {error_msg}<br><small>Supported formats: MP4, AVI, MOV, FLV, WMV, MKV, WebM</small></div>"
            return {"Processing Error": 1.0}, error_html, None

        # Lakukan inferensi
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Format probabilitas
        prob_dict = {CLASS_NAMES[i]: round(prob.item(), 4) for i, prob in enumerate(probabilities)}

        # Tentukan kelas prediksi
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class_name = idx_to_class_mapping[predicted_idx]
        confidence = probabilities[predicted_idx].item()

        # Buat HTML output yang lebih menarik
        metadata_html = f"""
                <style>
                    .metadata-container {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        padding: 20px;
                        border-radius: 12px;
                        background-color: #ffffff;
                        color: #2c3e50;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        max-width: 100%;
                    }}
                    
                    .metadata-container h3 {{
                        color: #1a237e;
                        margin-top: 0;
                        font-size: 1.5rem;
                        border-bottom: 2px solid #2196f3;
                        padding-bottom: 10px;
                    }}
                    
                    .metadata-container h4 {{
                        color: #1a237e;
                        margin-top: 20px;
                        margin-bottom: 10px;
                        font-size: 1.2rem;
                    }}
                    
                    .prediction-box {{
                        padding: 15px;
                        border-left: 4px solid #2196f3;
                        background-color: #e3f2fd;
                        margin-bottom: 20px;
                        border-radius: 4px;
                    }}
                    
                    .prediction-box p {{
                        margin: 8px 0;
                        color: #37474f;
                        font-size: 1rem;
                    }}
                    
                    .predicted-type {{
                        color: #1565c0;
                        font-size: 1.3rem;
                        font-weight: bold;
                    }}
                    
                    .metadata-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                        overflow-x: auto;
                        display: block;
                    }}
                    
                    .metadata-table tr {{
                        border-bottom: 1px solid #e0e0e0;
                    }}
                    
                    .metadata-table tr:nth-child(odd) {{
                        background-color: #f5f5f5;
                    }}
                    
                    .metadata-table tr:nth-child(even) {{
                        background-color: #ffffff;
                    }}
                    
                    .metadata-table td {{
                        padding: 12px 8px;
                        color: #37474f;
                        font-size: 0.95rem;
                    }}
                    
                    .metadata-table td:first-child {{
                        font-weight: 600;
                        color: #1a237e;
                        width: 40%;
                    }}
                    
                    /* Dark Mode */
                    @media (prefers-color-scheme: dark) {{
                        .metadata-container {{
                            background-color: #1e1e1e;
                            color: #e0e0e0;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
                        }}
                        
                        .metadata-container h3 {{
                            color: #64b5f6;
                            border-bottom-color: #42a5f5;
                        }}
                        
                        .metadata-container h4 {{
                            color: #64b5f6;
                        }}
                        
                        .prediction-box {{
                            background-color: #263238;
                            border-left-color: #42a5f5;
                        }}
                        
                        .prediction-box p {{
                            color: #b0bec5;
                        }}
                        
                        .predicted-type {{
                            color: #42a5f5;
                        }}
                        
                        .metadata-table tr {{
                            border-bottom-color: #37474f;
                        }}
                        
                        .metadata-table tr:nth-child(odd) {{
                            background-color: #263238;
                        }}
                        
                        .metadata-table tr:nth-child(even) {{
                            background-color: #2c3e50;
                        }}
                        
                        .metadata-table td {{
                            color: #b0bec5;
                        }}
                        
                        .metadata-table td:first-child {{
                            color: #64b5f6;
                        }}
                    }}
                    
                    /* Responsive untuk Mobile */
                    @media (max-width: 768px) {{
                        .metadata-container {{
                            padding: 15px;
                        }}
                        
                        .metadata-container h3 {{
                            font-size: 1.3rem;
                        }}
                        
                        .metadata-container h4 {{
                            font-size: 1.1rem;
                        }}
                        
                        .predicted-type {{
                            font-size: 1.1rem;
                        }}
                        
                        .metadata-table {{
                            font-size: 0.9rem;
                        }}
                        
                        .metadata-table td {{
                            padding: 10px 6px;
                        }}
                        
                        .metadata-table td:first-child {{
                            width: 45%;
                        }}
                    }}
                    
                    /* Responsive untuk Tablet */
                    @media (min-width: 769px) and (max-width: 1024px) {{
                        .metadata-container {{
                            padding: 18px;
                        }}
                    }}
                </style>

                <div class="metadata-container">
                    <h3>🎯 Prediction Result</h3>
                    <div class="prediction-box">
                        <p><b>Predicted Type:</b> <span class="predicted-type">{predicted_class_name}</span></p>
                        <p><b>Confidence:</b> {confidence:.2%}</p>
                    </div>
                    
                    <h4>📹 Video Metadata</h4>
                    <table class="metadata-table">
                        <tbody>
                            <tr>
                                <td>Resolution</td>
                                <td>{metadata['width']}x{metadata['height']}</td>
                            </tr>
                            <tr>
                                <td>FPS</td>
                                <td>{metadata['fps']:.2f}</td>
                            </tr>
                            <tr>
                                <td>Frame Count</td>
                                <td>{metadata['frame_count']}</td>
                            </tr>
                            <tr>
                                <td>Duration</td>
                                <td>{metadata['duration']:.2f} seconds</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            """



        return prob_dict, metadata_html, processed_video_path

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        error_html = f"<div style='color: red;'><b>Unexpected Error:</b> {str(e)}</div>"
        return {"Unexpected Error": 1.0}, error_html, None

# --- Gradio Interface dengan Blocks untuk layout lebih baik ---
with gr.Blocks(title="Video Forgery Detection", theme=gr.themes.Citrus()) as demo:
    gr.Markdown("""
    # 🎬 Video Forgery Detection System
    ### Wulandari Hafel 07352011078
    Upload a video to detect common forgery types including insertion, deletion, duplication, flipping, rotation, and zooming.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="📤 Upload Video",
                sources=["upload", "webcam"],
                format=None,  # Accept semua format video
                include_audio=False
            )
            
            submit_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            prediction_label = gr.Label(
                label="📊 Class Probabilities",
                num_top_classes=8
            )
    
    with gr.Row():
        metadata_display = gr.HTML(label="📋 Analysis Results")
    
    with gr.Row():
        processed_video = gr.Video(
            label="✅ Processed Video (Browser-Compatible)",
            format="mp4",
            autoplay=False
        )
    
    # Event handler
    submit_btn.click(
        fn=predict_forgery_gradio,
        inputs=video_input,
        outputs=[prediction_label, metadata_display, processed_video]
    )
    
    gr.Markdown("""
    ---
    **Note:** 
    - Supported formats: MP4, AVI, MOV, FLV, WMV (auto-converted to browser-compatible MP4)
    - Videos are automatically re-encoded for optimal browser playback
    - Model uses R(2+1)D architecture with 32 frames per video
    """)

# Jalankan aplikasi
if __name__ == "__main__":
    import os
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    
    demo.launch(
        share=True,
        debug=True,
        server_port=port,
        server_name="0.0.0.0"  # Untuk akses dari luar container
    )
