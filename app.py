"""
Open Crust Mining Detection System - Flask Backend
Detects mining areas from satellite imagery and identifies illegal mining outside boundaries
"""

import base64
import io
import json
import os
from datetime import datetime
import dotenv

import cv2
import geopandas as gpd
import matplotlib
import numpy as np
import torch
import yagmail
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
from PIL import Image
from shapely.geometry import Point, Polygon
from torchvision import transforms
from transformers import UperNetForSemanticSegmentation

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


# Custom JSON encoder to handle numpy types
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


app = Flask(__name__)
app.json = NumpyJSONProvider(app)  # Use custom JSON provider
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FOLDER"] = "results"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max file size

# Create necessary folders
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
os.makedirs("static", exist_ok=True)

# Global model variables
model = None
device = None
preprocess = None


def initialize_model():
    """Initialize the mining detection model"""
    global model, device, preprocess

    MODEL_NAME = "ericyu/minenetcd-upernet-Swin-Diff-S-Pretrained-ChannelMixing-Dropout"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {MODEL_NAME} on {device}...")
    model = UperNetForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Model loaded successfully.")


def predict_mining_area(img_path):
    """Predict mining areas from satellite image"""
    img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)

    logits = outputs.logits
    logits = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )

    pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    return img, pred_mask


def calculate_mining_area(mask, pixel_resolution=10):
    """Calculate mining area in square meters"""
    mining_pixels = int(np.sum(mask > 0))
    area_sqm = mining_pixels * (pixel_resolution**2)
    return float(area_sqm)


def check_boundary_violation(mask, boundary_coords=None):
    """Check if mining occurs outside authorized boundary"""
    if boundary_coords is None:
        return None, 0

    # Create binary mask for mining areas
    mining_mask = (mask > 0).astype(np.uint8)

    # Create boundary mask (simplified for demonstration)
    h, w = mining_mask.shape
    boundary_mask = np.zeros((h, w), dtype=np.uint8)

    # Convert normalized coordinates to pixel coordinates
    if boundary_coords and len(boundary_coords) > 0:
        pts = []
        for coord in boundary_coords:
            x = int(coord["x"] * w)
            y = int(coord["y"] * h)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(boundary_mask, [pts], 1)

    # Calculate illegal mining (mining outside boundary)
    illegal_mining = mining_mask * (1 - boundary_mask)
    illegal_area = int(np.sum(illegal_mining > 0))

    return illegal_mining, illegal_area


def create_visualization(img, pred_mask, boundary_coords=None, illegal_mask=None):
    """Create visualization images and return as base64"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Original Image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Satellite Image", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    # Predicted Mining Mask
    axes[0, 1].imshow(pred_mask, cmap="YlOrRd")
    axes[0, 1].set_title("Detected Mining Areas", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    # Overlay with boundary
    overlay = np.array(img).copy()
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.4 + np.array([255, 0, 0]) * 0.6

    # Draw boundary if provided
    if boundary_coords and len(boundary_coords) > 0:
        h, w = pred_mask.shape
        pts = []
        for coord in boundary_coords:
            x = int(coord["x"] * w)
            y = int(coord["y"] * h)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(
        "Mining Area Overlay (Red) with Boundary (Green)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1, 0].axis("off")

    # Illegal mining areas
    if illegal_mask is not None:
        illegal_overlay = np.array(img).copy()
        illegal_overlay[illegal_mask > 0] = (
            illegal_overlay[illegal_mask > 0] * 0.3 + np.array([255, 165, 0]) * 0.7
        )
        axes[1, 1].imshow(illegal_overlay)
        axes[1, 1].set_title(
            "Illegal Mining (Outside Boundary)", fontsize=14, fontweight="bold"
        )
    else:
        axes[1, 1].imshow(img)
        axes[1, 1].set_title("No Boundary Defined", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return img_base64


@app.route("/")
def index():
    """Serve main page"""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Analyze uploaded satellite image for mining activity"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Get boundary coordinates if provided
        boundary_coords = None
        if "boundary" in request.form:
            boundary_data = request.form["boundary"]
            if boundary_data:
                boundary_coords = json.loads(boundary_data)

        # Perform prediction
        img, pred_mask = predict_mining_area(filepath)

        # Calculate statistics
        total_area = calculate_mining_area(pred_mask)
        illegal_mask, illegal_area_pixels = check_boundary_violation(
            pred_mask, boundary_coords
        )
        illegal_area = float(
            illegal_area_pixels * 100
        )  # Approximate area, ensure float

        # Create visualization
        viz_base64 = create_visualization(img, pred_mask, boundary_coords, illegal_mask)

        # Prepare response (convert numpy types to Python native types)
        result = {
            "success": True,
            "visualization": viz_base64,
            "statistics": {
                "total_mining_area": float(round(total_area, 2)),
                "illegal_mining_area": float(round(illegal_area, 2)),
                "authorized_mining_area": float(round(total_area - illegal_area, 2)),
                "unit": "sq meters",
                "has_boundary": bool(boundary_coords is not None),
                "violation_detected": bool(illegal_area > 0),
            },
            "timestamp": str(timestamp),
        }

        if illegal_area >= 1:
            from os import environ

            # create reusable transporter (like nodemailer.createTransport)
            yag = yagmail.SMTP(environ["SMTP_USER"], environ["SMTP_PASS"])
            html = """
            <h2>Hey there 👋</h2>
            <p>This email was sent using <b>Python yagmail</b>.</p>
            """

            yag.send(
                to="faberneil69@gmail.com",
                subject="HTML + Attachments Example",
                contents=[html, "path/to/image.png", "path/to/file.pdf"],
            )
            print("Success")

        return jsonify(result)

    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    """Generate comprehensive PDF report of mining analysis"""
    try:
        data = request.json
        stats = data["statistics"]
        timestamp = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Get report metadata
        district = data.get("district", "Not Specified")
        state = data.get("state", "Not Specified")
        location_name = data.get("location_name", "Mining Site")
        latitude = data.get("latitude", "N/A")
        longitude = data.get("longitude", "N/A")

        # Determine severity level
        violation_severity = "CRITICAL" if stats["violation_detected"] else "NORMAL"

        # Create comprehensive report
        report_content = f"""
{'='*80}
                    OPEN CRUST MINING DETECTION REPORT
                          AUTOMATED SURVEILLANCE SYSTEM
{'='*80}

REPORT METADATA
{'='*80}
Report ID:              MINING-{timestamp}
Generated On:           {datetime.now().strftime('%d %B %Y, %I:%M:%S %p')}
Analysis Date:          {datetime.now().strftime('%d/%m/%Y')}
Report Type:            Automated Mining Activity Detection
Severity Level:         {violation_severity}

{'='*80}
LOCATION INFORMATION
{'='*80}
Site Name:              {location_name}
State:                  {state}
District:               {district}
GPS Coordinates:        Lat: {latitude}, Lon: {longitude}
Boundary Defined:       {'Yes' if stats['has_boundary'] else 'No'}

{'='*80}
MINING ACTIVITY ANALYSIS
{'='*80}
Total Mining Area Detected:     {stats['total_mining_area']:,.2f} sq meters
                               ({stats['total_mining_area']/10000:.2f} hectares)

Authorized Mining Area:         {stats['authorized_mining_area']:,.2f} sq meters
                               ({stats['authorized_mining_area']/10000:.2f} hectares)

Illegal Mining Area:            {stats['illegal_mining_area']:,.2f} sq meters
                               ({stats['illegal_mining_area']/10000:.2f} hectares)

Compliance Percentage:          {(stats['authorized_mining_area']/stats['total_mining_area']*100) if stats['total_mining_area'] > 0 else 100:.1f}%

{'='*80}
COMPLIANCE STATUS
{'='*80}
Violation Detected:     {'YES - IMMEDIATE ACTION REQUIRED' if stats['violation_detected'] else 'NO'}
Mining Status:          {'ILLEGAL ACTIVITY DETECTED' if stats['violation_detected'] else 'WITHIN AUTHORIZED LIMITS'}

"""

        if stats["violation_detected"]:
            report_content += f"""
{'='*80}
⚠️  VIOLATION DETAILS - URGENT ATTENTION REQUIRED  ⚠️
{'='*80}
Illegal Mining Area:    {stats['illegal_mining_area']:,.2f} sq meters
Violation Percentage:   {(stats['illegal_mining_area']/stats['total_mining_area']*100) if stats['total_mining_area'] > 0 else 0:.1f}% of total mining area

SEVERITY ASSESSMENT:
"""
            if stats["illegal_mining_area"] > 10000:  # > 1 hectare
                report_content += (
                    "- CRITICAL: Large-scale illegal mining detected (>1 hectare)\n"
                )
            elif stats["illegal_mining_area"] > 5000:
                report_content += (
                    "- HIGH: Significant illegal mining detected (>0.5 hectare)\n"
                )
            elif stats["illegal_mining_area"] > 1000:
                report_content += (
                    "- MEDIUM: Moderate illegal mining detected (>0.1 hectare)\n"
                )
            else:
                report_content += "- LOW: Minor boundary violation detected\n"

        report_content += f"""
{'='*80}
RECOMMENDATIONS & ACTION ITEMS
{'='*80}
"""

        if stats["violation_detected"]:
            report_content += """
IMMEDIATE ACTIONS REQUIRED:
1. Deploy field inspection team to verify illegal mining activity
2. Issue stop-work notice to mining organization
3. Notify State Mining Department and District Collector
4. Initiate penalty proceedings under relevant mining regulations
5. Conduct environmental impact assessment
6. Suspend mining lease if violations continue

AUTHORITIES TO BE NOTIFIED:
1. District Magistrate / District Collector
2. State Directorate of Mines & Geology
3. Regional Office - Ministry of Mines
4. State Pollution Control Board
5. Indian Bureau of Mines (IBM) Regional Office
6. Local Police Station (for enforcement)

LEGAL PROVISIONS:
- Mines and Minerals (Development and Regulation) Act, 1957
- Environment (Protection) Act, 1986
- Relevant State Mining Rules
"""
        else:
            report_content += """
ROUTINE MONITORING:
1. Continue regular satellite monitoring
2. Schedule quarterly compliance inspection
3. Maintain mining activity logs
4. Update boundary verification records

AUTHORITIES TO BE INFORMED:
1. State Directorate of Mines & Geology (Compliance Report)
2. District Mining Officer (Routine Update)
"""

        report_content += f"""
{'='*80}
TECHNICAL DETAILS
{'='*80}
Detection Method:       AI-powered Semantic Segmentation (UperNet)
Image Source:           Satellite Imagery (EO/SAR)
Model Accuracy:         Deep Learning Based Detection
Analysis Type:          2D Spatial Analysis
Boundary Method:        GPS Coordinate Polygon Mapping

{'='*80}
REPORT CERTIFICATION
{'='*80}
This report has been automatically generated by the Open Crust Mining Detection
System using AI-powered satellite image analysis. The findings are based on
remote sensing data and should be verified through field inspection.

System Version:         1.0.0
Report Format:          Standard Compliance Report
Authorized By:          Automated Mining Surveillance System

{'='*80}
CONTACT INFORMATION
{'='*80}
For queries regarding this report, please contact:
- District Mining Office: {district}, {state}
- State Mining Department: {state}
- Email: mining.surveillance@gov.in (placeholder)
- Helpline: 1800-XXX-XXXX (placeholder)

{'='*80}
                            END OF REPORT
{'='*80}

Report generated by Open Crust Mining Detection System
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report ID: MINING-{timestamp}
"""

        # Save report
        report_filename = f"mining_compliance_report_{state}_{district}_{timestamp}.txt"
        report_path = os.path.join(app.config["RESULTS_FOLDER"], report_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return jsonify(
            {
                "success": True,
                "report": report_content,
                "filename": report_filename,
                "report_id": f"MINING-{timestamp}",
                "severity": violation_severity,
            }
        )

    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Check if model is loaded and ready"""
    return jsonify(
        {
            "status": "ready" if model is not None else "not_ready",
            "device": str(device) if device else "unknown",
        }
    )


if __name__ == "__main__":
    print("Initializing Open Crust Mining Detection System...")
    initialize_model()
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
