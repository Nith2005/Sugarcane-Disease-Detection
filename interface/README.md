# 🌾 Sugarcane Disease & Insect Detection - Web Interface

Professional web application for detecting diseases and insects in sugarcane crops using AI.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ..
pip install -r requirements.txt
```

### 2. Run the Application

**Windows:**
```bash
# Double-click run.bat or:
python app.py
```

**Linux/Mac:**
```bash
python app.py
```

### 3. Open in Browser
Navigate to: **http://localhost:5000**

## ✨ Features

- 🖼️ **Drag & Drop Upload** - Easy image uploading
- 🤖 **Dual Models** - Object Detection & Instance Segmentation
- ⚡ **Real-time Analysis** - Instant results
- 📊 **Detailed Reports** - Comprehensive analysis with recommendations
- 🎨 **Modern UI** - Beautiful, responsive design
- 📱 **Mobile Friendly** - Works on all devices

## 🎯 How to Use

1. **Upload Image**
   - Click the upload area or drag & drop an image
   - Supports: PNG, JPG, JPEG, BMP, TIFF (max 16MB)

2. **Choose Model**
   - **Object Detection**: Fast, bounding boxes
   - **Instance Segmentation**: Precise, pixel-level masks

3. **Adjust Confidence**
   - Lower threshold = more detections
   - Higher threshold = more confident detections

4. **Analyze**
   - Click "Analyze Image" button
   - View annotated results and detailed report

## 📁 Project Structure

```
interface/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Main HTML page
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend logic
├── uploads/              # Temporary upload folder
├── results/              # Temporary results folder
├── run.bat               # Windows launcher
└── README.md             # This file
```

## 🔧 Configuration

### Model Paths
Edit in `app.py` (lines 29-30):
```python
DETECTION_MODEL_PATH = MODEL_DIR / "yolov8.pt"
SEGMENTATION_MODEL_PATH = MODEL_DIR / "yolov8_seg.pt"
```

### Class Information
Update class names and recommendations in `app.py` (lines 37-57):
```python
CLASS_INFO = {
    "healthy": {...},
    "disease": {...},
    "insect": {...}
}
```

### Server Settings
Modify in `app.py` (line 229):
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🌐 API Endpoints

### POST `/api/analyze`
Analyze an uploaded image

**Request:**
- `file`: Image file (multipart/form-data)
- `model_type`: "detection" or "segmentation"
- `conf_threshold`: Float (0.1 - 0.9)

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

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "detection_model": "path/to/model",
  "segmentation_model": "path/to/model"
}
```

## 🎨 Customization

### Change Colors
Edit CSS variables in `static/style.css` (lines 10-20):
```css
:root {
    --primary: #10b981;
    --danger: #ef4444;
    --warning: #f59e0b;
    ...
}
```

### Modify Layout
Edit `templates/index.html` to change structure

### Add Features
Extend `app.py` with new endpoints and functionality

## 🛠️ Troubleshooting

### Port Already in Use
```bash
# Change port in app.py
app.run(port=5001)
```

### Models Not Loading
- Verify model files exist in `../models/`
- Check file paths in `app.py`
- Ensure models are valid YOLOv8 .pt files

### Upload Fails
- Check file size (max 16MB)
- Verify file format (PNG, JPG, etc.)
- Ensure `uploads/` folder exists

### Slow Performance
- Use detection model instead of segmentation
- Reduce image resolution
- Enable GPU acceleration

## 📊 Performance

**Typical Response Times:**
- Detection: ~100-300ms
- Segmentation: ~200-500ms

*Times vary based on hardware and image size*

## 🔐 Security Notes

For production deployment:
- [ ] Add authentication
- [ ] Implement rate limiting
- [ ] Use HTTPS
- [ ] Validate file uploads
- [ ] Set up CORS properly
- [ ] Use environment variables for config

## 🚀 Deployment

### Local Network
```python
# In app.py
app.run(host='0.0.0.0', port=5000)
# Access from other devices: http://YOUR_IP:5000
```

### Cloud Deployment
- **Heroku**: Add `Procfile` and `runtime.txt`
- **AWS EC2**: Use Gunicorn + Nginx
- **Google Cloud Run**: Containerize with Docker
- **Azure**: Use Azure App Service

## 📚 Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [Deployment Guide](https://flask.palletsprojects.com/en/latest/deploying/)

## 🆘 Support

For issues:
1. Check console output for errors
2. Review this README
3. Check main project README
4. Verify all dependencies are installed

---

**Built with ❤️ using Flask, YOLOv8, and modern web technologies**
