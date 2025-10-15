# Open Crust Mining Detection System

An AI-powered web application for detecting open crust mining activities from satellite imagery with boundary compliance checking.

## Features

- 🛰️ **Satellite Image Analysis**: Upload EO/SAR satellite imagery for automated mining detection
- 🗺️ **Boundary Compliance**: Define authorized mining boundaries and detect illegal mining activities
- 📊 **Interactive Visualization**: 2D visualization of detected mining areas with overlay comparisons
- 📄 **Report Generation**: Automated compliance report generation for authorities
- 🎯 **Area Calculation**: Precise calculation of total and illegal mining areas
- 🌐 **Web Interface**: User-friendly browser-based interface with interactive mapping


## Project Structure

```
opencrustmine/
├── app.py                  # Flask backend with ML model
├── templates/
│   └── index.html         # Frontend interface
├── requirements.txt        # Python dependencies
├── uploads/               # Uploaded images (auto-created)
├── results/               # Analysis results (auto-created)
└── README.md              # This file
```

## Technical Details

### Model
- **Architecture**: UperNet with Swin Transformer backbone
- **Task**: Semantic segmentation for mining area detection
- **Model**: `ericyu/minenetcd-upernet-Swin-Diff-S-Pretrained-ChannelMixing-Dropout`

### Backend (Flask)
- Image upload and processing
- Mining area detection using deep learning
- Boundary violation detection
- Area calculations
- Report generation

### Frontend
- Responsive HTML/CSS/JS interface
- Leaflet.js for interactive mapping
- Drag-and-drop file upload
- Real-time visualization
- Statistical dashboard

## Problem Statement Addressed

This tool addresses the following requirements from the problem statement:

✅ Automated detection of open crust mining activity from satellite imagery (EO/SAR)
✅ Detection of mining extent and area calculation
✅ Boundary layer support (Shapefile/KML concept via interactive drawing)
✅ Identification of mining outside authorized boundaries
✅ Calculation of illegal mining area
✅ Interactive mapping platform for visualization
✅ 2D visualization of mining areas
✅ Report generation module for authorities

### Future Enhancements (Not in Current Scope)
- 3D visualization using DEM data
- Mining depth and volume calculation (Simpson's method)
- Shapefile/KML file upload support
- PDF report generation with maps
- Multi-temporal analysis
- Batch processing

## Installation

1. **Clone or navigate to the repository**
   ```bash
   cd c:\Users\neilf\Documents\opencrustmine
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:5000`

3. **Analyze satellite imagery**
   - Upload a satellite image (PNG, JPG, JPEG)
   - Optionally draw boundary polygon on the map for authorized mining area
   - Click "Analyze Mining Activity"
   - Review results and generate compliance report


## System Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended for GPU)
- CUDA-compatible GPU (optional, for faster processing)
- Modern web browser (Chrome, Firefox, Edge)

## Notes

- First run will download the ML model (~500MB)
- Processing time depends on image size and available hardware
- GPU acceleration significantly improves performance
- Boundary coordinates are normalized for flexibility

## License

This is an educational/research project for mining activity detection and compliance monitoring.
