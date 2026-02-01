"""
Sugarcane Disease & Insect Detection - Flask Web Application
Professional web interface with REST API
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
from pathlib import Path
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load models
MODEL_DIR = Path("../models")
DETECTION_MODEL_PATH = MODEL_DIR / "yolov8.pt"
SEGMENTATION_MODEL_PATH = MODEL_DIR / "yolov8_seg.pt"

print("Loading models...")
detection_model = None
segmentation_model = None
models_available = []

# Try loading detection model
try:
    detection_model = YOLO(str(DETECTION_MODEL_PATH))
    models_available.append("detection")
    print("✓ Detection model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load detection model: {e}")

# Try loading segmentation model
try:
    segmentation_model = YOLO(str(SEGMENTATION_MODEL_PATH))
    models_available.append("segmentation")
    print("✓ Segmentation model loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Segmentation model failed to load: {e}")
    print("  This is likely due to version compatibility issues.")
    print("  The app will run in detection-only mode.")

# Check if at least one model is available
if not models_available:
    print("\n" + "="*60)
    print("ERROR: No models could be loaded!")
    print("="*60)
    print("\nPlease check:")
    print("1. Model files exist in ../models/")
    print("2. Model files are valid YOLOv8 .pt files")
    print("3. Ultralytics version is compatible")
    print("\nTry reinstalling ultralytics:")
    print("  pip install --upgrade ultralytics")
    print("="*60 + "\n")
    exit(1)

print(f"\n✓ Available models: {', '.join(models_available)}")
if 'segmentation' not in models_available:
    print("ℹ Running in detection-only mode")

# Class information
CLASS_INFO = {
    "healthy": {
        "color": "#10b981",
        "icon": "✓",
        "description": "Healthy sugarcane tissue",
        "severity": "low",
        "recommendation": "No action needed. Continue regular monitoring and maintenance."
    },
    "disease": {
        "color": "#ef4444",
        "icon": "⚠",
        "description": "Disease detected",
        "severity": "high",
        "recommendation": "Apply appropriate fungicide immediately. Isolate affected plants if severe. Consult agronomist for treatment plan."
    },
    "insect": {
        "color": "#f59e0b",
        "icon": "⚡",
        "description": "Insect pest detected",
        "severity": "medium",
        "recommendation": "Apply targeted insecticide. Monitor surrounding plants. Consider biological pest control methods."
    }
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def pil_to_base64(img):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def process_image(image_path, model_type='detection', conf_threshold=0.25):
    """Process image with selected model"""
    try:
        # Select model with fallback
        if model_type == 'segmentation' and segmentation_model is not None:
            model = segmentation_model
        elif model_type == 'segmentation' and segmentation_model is None:
            # Fallback to detection if segmentation not available
            model = detection_model
            model_type = 'detection'
            print("⚠ Segmentation model not available, using detection model")
        else:
            model = detection_model
        
        if model is None:
            return {
                'success': False,
                'error': 'No models available. Please check model files.'
            }
        
        # Run inference
        results = model(image_path, conf=conf_threshold)
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_img)
        
        # Generate analysis
        detections = results[0].boxes
        analysis = generate_analysis(detections, results[0].names, model_type)
        
        return {
            'success': True,
            'image': pil_to_base64(annotated_pil),
            'analysis': analysis
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def generate_analysis(detections, class_names, model_type):
    """Generate detailed analysis from detections"""
    if len(detections) == 0:
        return {
            'total_detections': 0,
            'status': 'healthy',
            'message': 'No issues detected! Your sugarcane appears healthy.',
            'detections': [],
            'recommendations': [
                'Continue regular monitoring',
                'Maintain current care practices',
                'Check again in 1-2 weeks'
            ]
        }
    
    # Count detections by class
    class_counts = {}
    class_confidences = {}
    
    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        conf = float(box.conf[0])
        
        if cls_name not in class_counts:
            class_counts[cls_name] = 0
            class_confidences[cls_name] = []
        
        class_counts[cls_name] += 1
        class_confidences[cls_name].append(conf)
    
    # Build detection details
    detection_details = []
    overall_severity = 'low'
    
    for cls_name, count in class_counts.items():
        avg_conf = sum(class_confidences[cls_name]) / count
        info = CLASS_INFO.get(cls_name, {})
        
        detection_details.append({
            'class': cls_name,
            'count': count,
            'confidence': round(avg_conf * 100, 1),
            'color': info.get('color', '#6b7280'),
            'icon': info.get('icon', '•'),
            'description': info.get('description', 'Unknown'),
            'severity': info.get('severity', 'low'),
            'recommendation': info.get('recommendation', 'Monitor closely')
        })
        
        # Update overall severity
        if info.get('severity') == 'high':
            overall_severity = 'high'
        elif info.get('severity') == 'medium' and overall_severity != 'high':
            overall_severity = 'medium'
    
    # Generate recommendations
    recommendations = []
    if 'disease' in class_counts:
        recommendations.extend([
            'Immediate treatment required for detected diseases',
            'Apply appropriate fungicide or bactericide',
            'Isolate severely affected plants',
            'Monitor daily for spread'
        ])
    if 'insect' in class_counts:
        recommendations.extend([
            'Apply targeted pest control measures',
            'Consider integrated pest management (IPM)',
            'Check surrounding plants for infestation'
        ])
    if not recommendations:
        recommendations = [
            'Continue current practices',
            'Regular monitoring recommended'
        ]
    
    # Determine status
    status = 'critical' if overall_severity == 'high' else ('warning' if overall_severity == 'medium' else 'healthy')
    
    return {
        'total_detections': len(detections),
        'status': status,
        'model_type': model_type,
        'message': f'Detected {len(detections)} issue(s) in the image',
        'detections': detection_details,
        'recommendations': recommendations
    }


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint for image analysis"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Get parameters
        model_type = request.form.get('model_type', 'detection')
        conf_threshold = float(request.form.get('conf_threshold', 0.25))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        result = process_image(filepath, model_type, conf_threshold)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_available': models_available,
        'detection_model': str(DETECTION_MODEL_PATH) if detection_model else 'Not loaded',
        'segmentation_model': str(SEGMENTATION_MODEL_PATH) if segmentation_model else 'Not loaded'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌾 Sugarcane Disease & Insect Detection System")
    print("="*60)
    print("\n✓ Models loaded successfully")
    print("✓ Server starting...")
    print("\n📍 Open your browser to: http://localhost:5000")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
