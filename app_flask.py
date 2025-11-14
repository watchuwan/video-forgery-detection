import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import random
from pathlib import Path
import ffmpy
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import io
import re


# === Configuration ===
IMAGE_SIZE = (112, 112)
NUM_FRAMES_PER_VIDEO = 32
# Get the directory of the current file and use uploads folder relative to it
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'flv', 'wmv', 'mkv', 'webm', 'm4v', '3gp'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB


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
MODEL_WEIGHTS_PATH = os.path.join(APP_ROOT, "model", "best_video_resnet_multi_class_model.pth")


# === Flask App Setup ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


print(f"📁 Upload folder: {UPLOAD_FOLDER}")
print(f"📁 Model path: {MODEL_WEIGHTS_PATH}")


# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Flask app using device: {DEVICE}")


# Global variables
model = None
video_transform = None
idx_to_class_mapping = None


# === Model Loading ===
def load_model_and_transforms():
    """Load model and initialize transforms"""
    global model, video_transform, idx_to_class_mapping

    print(f"Loading model from {MODEL_WEIGHTS_PATH}...")
    try:
        # Create model architecture
        model = models.video.r2plus1d_18(weights=None)
        num_ftrs_inf = model.fc.in_features
        model.fc = nn.Linear(num_ftrs_inf, NUM_CLASSES)

        # Load trained weights
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        print("✅ Model loaded successfully")

        # Define transformations
        video_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("✅ Video transformations initialized")

        # Create idx_to_class mapping
        idx_to_class_mapping = {i: name for i, name in enumerate(CLASS_NAMES)}
        print("✅ Class mapping initialized")

    except Exception as e:
        print(f"❌ ERROR: Could not load model: {e}")
        model = None


# Load model on startup
load_model_and_transforms()


# === Video Processing ===
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_video_to_mp4(video_path):
    """Convert video to MP4 format compatible with browsers"""
    file_ext = Path(video_path).suffix.lower()
    
    if file_ext == '.mp4':
        return video_path
    
    mp4_path = str(Path(video_path).with_suffix('.mp4'))
    
    try:
        print(f"🔄 Converting {file_ext.upper()} to MP4...")
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
        print(f"✅ Video converted to: {mp4_path}")
        return mp4_path
    except Exception as e:
        print(f"⚠️  Conversion failed: {e}. Trying original file...")
        return video_path


def video_to_tensor_conversion(video_path):
    """Convert video to tensor for model inference"""
    try:
        # Convert to MP4 if needed
        processed_path = convert_video_to_mp4(video_path)
        
        cap = cv2.VideoCapture(processed_path)
        if not cap.isOpened():
            return None, {"error": f"Could not open video file"}, None
        
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
            "duration": round(duration_seconds, 2)
        }
        
        # Extract frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if not frames:
            return None, {"error": "No frames extracted from video"}, None
        
        # Sample frames
        selected_frames = []
        if len(frames) >= NUM_FRAMES_PER_VIDEO:
            indices = np.linspace(0, len(frames) - 1, NUM_FRAMES_PER_VIDEO, dtype=int)
            selected_frames = [frames[i] for i in indices]
        else:
            selected_frames = frames
            while len(selected_frames) < NUM_FRAMES_PER_VIDEO:
                selected_frames.append(random.choice(frames))
            selected_frames = selected_frames[:NUM_FRAMES_PER_VIDEO]
        
        # Transform frames
        transformed_frames = [video_transform(frame) for frame in selected_frames]
        video_tensor = torch.stack(transformed_frames, dim=1)  # (C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)
        
        return video_tensor.to(DEVICE), metadata, processed_path
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None, {"error": str(e)}, None


# === Range Request Support untuk Video Streaming ===
@app.after_request
def after_request(response):
    """Tambahkan Accept-Ranges header untuk semua response"""
    response.headers.add('Accept-Ranges', 'bytes')
    return response


def send_file_partial(path):
    """
    Wrapper untuk send_file yang mendukung HTTP 206 Partial Content (byte ranges)
    untuk video streaming di browser
    """
    range_header = request.headers.get('Range', None)
    
    # Jika tidak ada Range header, kirim file biasa
    if not range_header:
        return send_file(path, mimetype='video/mp4')
    
    # Dapatkan ukuran file
    size = os.path.getsize(path)
    byte1, byte2 = 0, None
    
    # Parse Range header (format: bytes=start-end)
    m = re.search(r'(\d+)-(\d*)', range_header)
    g = m.groups()
    
    if g[0]:
        byte1 = int(g[0])
    if g[1]:
        byte2 = int(g[1])
    
    # Hitung panjang data yang akan dikirim
    if byte2 is not None:
        length = byte2 - byte1 + 1
    else:
        length = size - byte1
    
    # Baca data dari file
    with open(path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)
    
    # Buat response 206 Partial Content
    rv = Response(
        data,
        206,
        mimetype='video/mp4',
        direct_passthrough=True
    )
    
    # Tambahkan Content-Range header
    rv.headers.add(
        'Content-Range',
        f'bytes {byte1}-{byte1 + length - 1}/{size}'
    )
    
    return rv


# === Flask Routes ===
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for video prediction"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Check if file is in request
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"📽️  Processing video: {filename}")
        
        # Convert to tensor and get the processed video path
        video_tensor, metadata, processed_path = video_to_tensor_conversion(filepath)
        
        if video_tensor is None:
            return jsonify({"error": metadata.get("error", "Could not process video")}), 400
        
        # Get the filename of the processed MP4 video
        processed_filename = os.path.basename(processed_path)
        
        # Run inference
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Format results
        prob_dict = {CLASS_NAMES[i]: round(prob.item(), 4) for i, prob in enumerate(probabilities)}
        
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = idx_to_class_mapping[predicted_idx]
        confidence = round(probabilities[predicted_idx].item(), 4)
        
        result = {
            "success": True,
            "filename": processed_filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "metadata": metadata,
            "video_url": f"/api/video/{processed_filename}",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Prediction complete: {predicted_class} ({confidence})")
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "supported_formats": list(ALLOWED_EXTENSIONS)
    }), 200


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get available forgery classes"""
    return jsonify({
        "classes": CLASS_NAMES,
        "num_classes": NUM_CLASSES
    }), 200


@app.route('/api/video/<filename>', methods=['GET'])
def serve_video(filename):
    """Serve processed video file dengan range request support"""
    try:
        # Security: validate filename
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({"error": "Video file not found"}), 404
        
        # Gunakan send_file_partial untuk mendukung range requests
        return send_file_partial(filepath)
    
    except Exception as e:
        print(f"❌ Error serving video: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_video(filename):
    """Download processed video file"""
    try:
        # Security: validate filename
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            return jsonify({"error": "Invalid filename"}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({"error": "Video file not found"}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    except Exception as e:
        print(f"❌ Error downloading video: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({"error": f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024):.0f}MB"}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 error"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 error"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print(f"🚀 Starting Flask app on http://0.0.0.0:5000")
    print(f"📋 Supported video formats: {', '.join(ALLOWED_EXTENSIONS)}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
