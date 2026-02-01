// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const modelType = document.getElementById('modelType');
const confThreshold = document.getElementById('confThreshold');
const confValue = document.getElementById('confValue');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const analysisReport = document.getElementById('analysisReport');

let selectedFile = null;

// Upload Area Click
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File Input Change
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

// Confidence Threshold Slider
confThreshold.addEventListener('input', (e) => {
    confValue.textContent = e.target.value;
});

// Analyze Button
analyzeBtn.addEventListener('click', analyzeImage);

// Handle File
function handleFile(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, JPEG, BMP, TIFF)');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Enable analyze button
    analyzeBtn.disabled = false;

    // Hide previous results
    resultsSection.style.display = 'none';
}

// Analyze Image
async function analyzeImage() {
    if (!selectedFile) return;

    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('.btn-text').style.display = 'none';
    analyzeBtn.querySelector('.btn-loader').style.display = 'flex';

    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_type', modelType.value);
    formData.append('conf_threshold', confThreshold.value);

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error analyzing image: ' + error.message);
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-text').style.display = 'block';
        analyzeBtn.querySelector('.btn-loader').style.display = 'none';
    }
}

// Display Results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';

    // Display annotated image
    resultImage.src = data.image;

    // Display analysis report
    const analysis = data.analysis;

    let reportHTML = '';

    // Status badge
    const statusClass = `status-${analysis.status}`;
    const statusText = analysis.status === 'healthy' ? '✓ Healthy' :
        analysis.status === 'warning' ? '⚠ Warning' :
            '⚠ Critical';

    reportHTML += `
        <div class="status-badge ${statusClass}">
            ${statusText}
        </div>
        <p style="margin-bottom: 1.5rem; font-size: 1.125rem;">
            <strong>${analysis.message}</strong>
        </p>
    `;

    // Detection details
    if (analysis.detections && analysis.detections.length > 0) {
        reportHTML += '<h4 style="margin-bottom: 1rem; color: var(--gray-900);">Detection Details</h4>';

        analysis.detections.forEach(detection => {
            reportHTML += `
                <div class="detection-item" style="border-left-color: ${detection.color}">
                    <div class="detection-header">
                        <span class="detection-class" style="color: ${detection.color}">
                            ${detection.icon} ${detection.class}
                        </span>
                        <span class="detection-confidence">
                            ${detection.confidence}% confident
                        </span>
                    </div>
                    <p class="detection-description">
                        <strong>Count:</strong> ${detection.count} | 
                        <strong>Severity:</strong> ${detection.severity}
                    </p>
                    <p class="detection-description">${detection.description}</p>
                    <div class="detection-recommendation">
                        <strong>Recommendation:</strong> ${detection.recommendation}
                    </div>
                </div>
            `;
        });
    }

    // Overall recommendations
    if (analysis.recommendations && analysis.recommendations.length > 0) {
        reportHTML += `
            <h4 style="margin: 1.5rem 0 1rem; color: var(--gray-900);">
                Recommended Actions
            </h4>
            <ul class="recommendations-list">
        `;

        analysis.recommendations.forEach(rec => {
            reportHTML += `<li>${rec}</li>`;
        });

        reportHTML += '</ul>';
    }

    // Model info
    reportHTML += `
        <p style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid var(--gray-200); color: var(--gray-600); font-size: 0.875rem;">
            <strong>Analysis Type:</strong> ${analysis.model_type === 'detection' ? 'Object Detection' : 'Instance Segmentation'}
        </p>
    `;

    analysisReport.innerHTML = reportHTML;

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
