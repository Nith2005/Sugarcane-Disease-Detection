# 🌾 Sugarcane Disease & Insect Detection

An AI-powered web application for early detection and classification of diseases and insect pests in sugarcane crops using YOLOv8 deep learning models.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-black)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Overview

This project provides a complete solution for detecting and analyzing sugarcane diseases and insect damage using state-of-the-art computer vision. It includes:

- **Dual AI Models**: Object detection and instance segmentation
- **Professional Web Interface**: Modern, responsive UI built with Flask
- **Real-time Analysis**: Instant detection with detailed reports
- **Actionable Insights**: Treatment recommendations for detected issues

## ✨ Features

### 🤖 AI Models
- **YOLOv8 Object Detection**: Fast bounding box detection (~30-50 FPS)
- **YOLOv8 Instance Segmentation**: Precise pixel-level masks (~20-30 FPS)
- **Multi-class Detection**: Healthy, Disease, and Insect categories

### 🌐 Web Interface
- Drag-and-drop image upload
- Adjustable confidence threshold
- Side-by-side model comparison
- Detailed analysis reports
- Treatment recommendations
- Mobile-responsive design

### 📊 Analysis Features
- Detection count and confidence scores
- Severity assessment
- Color-coded visualizations
- Comprehensive recommendations
- Export-ready reports

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Sugarcane-Disease-Insect-Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the web application**
```bash
cd interface
python app.py
```

4. **Open in browser**
```
http://localhost:5000
```

## 📁 Project Structure

```
Sugarcane-Disease-Insect-Detection/
├── interface/                  # Web application
│   ├── app.py                 # Flask backend
│   ├── templates/             # HTML templates
│   ├── static/                # CSS, JavaScript
│   ├── uploads/               # Temporary uploads
│   ├── results/               # Temporary results
│   ├── run.bat                # Windows launcher
│   └── README.md              # Interface docs
├── models/                     # Trained AI models
│   ├── yolov8.pt              # Detection model (22.5 MB)
│   └── yolov8_seg.pt          # Segmentation model (71.2 MB)
├── inference/                  # Command-line inference
│   ├── predict.py             # Inference script
│   ├── input/                 # Input images
│   └── output/                # Results
├── training/                   # Model training
│   ├── train_seg.py           # Training script
│   ├── setup_dataset.py       # Dataset setup
│   └── README.md              # Training guide
├── data/                       # Dataset configuration
│   └── dataset.yaml           # YOLO config
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 💻 Usage

### Web Interface (Recommended)

1. **Start the server**
   ```bash
   cd interface
   python app.py
   ```

2. **Upload an image**
   - Click the upload area or drag & drop
   - Supports: PNG, JPG, JPEG, BMP, TIFF (max 16MB)

3. **Configure analysis**
   - Choose model type (Detection or Segmentation)
   - Adjust confidence threshold (0.1 - 0.9)

4. **Analyze**
   - Click "Analyze Image"
   - View annotated results and detailed report

### Command-Line Inference

For batch processing:
```bash
cd inference
python predict.py
```

Edit `predict.py` to configure paths and parameters.

## 🎯 Model Information

### Detection Model (`yolov8.pt`)
- **Type**: YOLOv8n Object Detection
- **Size**: 22.5 MB
- **Speed**: ~30-50 FPS (GPU)
- **Output**: Bounding boxes
- **Use Case**: Quick screening, real-time monitoring

### Segmentation Model (`yolov8_seg.pt`)
- **Type**: YOLOv8n Instance Segmentation
- **Size**: 71.2 MB
- **Speed**: ~20-30 FPS (GPU)
- **Output**: Pixel-level masks
- **Use Case**: Detailed analysis, area calculation

### Classes
1. **Healthy** 🟢 - Normal, healthy sugarcane tissue
2. **Disease** 🔴 - Disease-affected areas (fungal, bacterial, viral)
3. **Insect** 🟠 - Insect pest damage or presence

## 📊 Performance

### Accuracy (Validation Set)
- **mAP50**: ~0.85-0.90
- **mAP50-95**: ~0.70-0.80

### Inference Speed (NVIDIA RTX 3060)
- **Detection**: 20-30ms per image
- **Segmentation**: 40-60ms per image

*Results may vary based on hardware and image resolution*

## 🎓 Training Your Own Models

See detailed training guide: [`training/README.md`](training/README.md)

**Quick steps:**
1. Prepare your dataset in YOLO format
2. Run `python training/setup_dataset.py`
3. Place images and labels in `data/` folders
4. Configure `training/train_seg.py`
5. Run `python training/train_seg.py`

## 🔧 Configuration

### Update Class Names
Edit `interface/app.py` (lines 37-57):
```python
CLASS_INFO = {
    "healthy": {...},
    "disease": {...},
    "insect": {...}
}
```

### Change Model Paths
Edit `interface/app.py` (lines 29-30):
```python
DETECTION_MODEL_PATH = MODEL_DIR / "yolov8.pt"
SEGMENTATION_MODEL_PATH = MODEL_DIR / "yolov8_seg.pt"
```

### Adjust Server Settings
Edit `interface/app.py` (line 229):
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🌐 API Documentation

### POST `/api/analyze`
Analyze an uploaded image

**Request:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@image.jpg" \
  -F "model_type=detection" \
  -F "conf_threshold=0.25"
```

**Response:**
```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "analysis": {
    "total_detections": 3,
    "status": "warning",
    "detections": [...],
    "recommendations": [...]
  }
}
```

### GET `/api/health`
Check server health
```bash
curl http://localhost:5000/api/health
```

## 🛠️ Troubleshooting

### Common Issues

**Models not loading**
- Verify model files exist in `models/` directory
- Check file paths in `app.py`
- Ensure models are valid YOLOv8 `.pt` files

**CUDA out of memory**
- Use detection model instead of segmentation
- Reduce image resolution
- Close other GPU applications

**Port already in use**
```python
# Change port in app.py
app.run(port=5001)
```

**Slow inference**
- Enable GPU acceleration
- Use smaller images
- Use detection model for speed

## 🚀 Deployment

### Local Network
```python
# In app.py
app.run(host='0.0.0.0', port=5000)
# Access: http://YOUR_IP:5000
```

### Production Deployment
- **Docker**: Containerize the application
- **Heroku**: Add `Procfile` and `runtime.txt`
- **AWS EC2**: Use Gunicorn + Nginx
- **Google Cloud Run**: Deploy with Docker
- **Azure**: Use Azure App Service

**Security checklist for production:**
- [ ] Add authentication
- [ ] Implement rate limiting
- [ ] Use HTTPS
- [ ] Validate file uploads
- [ ] Set up CORS
- [ ] Use environment variables
- [ ] Enable logging

## 📚 Technologies Used

- **Deep Learning**: PyTorch, Ultralytics YOLOv8
- **Computer Vision**: OpenCV, Pillow
- **Web Framework**: Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: NumPy

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

Developed by [Nith2005](https://github.com/Nith2005)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep learning platform

## 📞 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [https://github.com/Nith2005/Sugarcane-Disease-Detection/issues](https://github.com/Nith2005/Sugarcane-Disease-Detection/issues)
- **Pull Requests**: Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Documentation**: See README files in each directory

## 📈 Roadmap

- [ ] Add more disease classes
- [ ] Implement batch processing
- [ ] Add data augmentation options
- [ ] Create mobile app
- [ ] Add multi-language support
- [ ] Integrate with IoT sensors
- [ ] Add historical tracking
- [ ] Export to PDF reports

## ⚠️ Disclaimer

This is an AI-assisted agricultural tool designed to help with early detection. For critical decisions regarding crop management and treatment, always consult with qualified agricultural experts and agronomists.

---

**Made with ❤️ for sustainable agriculture**
