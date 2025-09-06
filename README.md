# Gesture Prediction API

A Flask-based REST API for gesture prediction that accepts base64 encoded images and returns predicted gesture characters. Features a modular design with extensible gesture recognition capabilities.

## Features

- **🌐 Web Interface**: Complete webcam-based gesture recognition interface at `/`
- **📹 Live Webcam**: Real-time video streaming with frame capture
- **🤖 AI Prediction**: MediaPipe-powered hand landmark detection with gesture classification
- **📊 Detailed Results**: Shows gesture key, confidence, processing time, and hand landmarks
- **📱 Responsive Design**: Works on desktop and mobile devices
- **👋 Real-time Hand Tracking**: 21-point hand landmark detection using MediaPipe
- **🔌 REST API**: Full API access for integration
  - **POST /predict**: Accepts base64 encoded images and returns predicted gesture characters
  - **GET /gestures**: Returns all supported gestures and their mappings
  - **GET /health**: Health check endpoint
  - **GET /api**: API information and usage guide
- **🧩 Modular Design**: Separate model.py module with MediaPipe integration
- **🔧 Extensible**: Ready for advanced machine learning model integration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### POST /predict
Accepts base64 encoded image and returns gesture prediction.

**Request Format:**
```json
{
    "image": "base64_encoded_image_data"
}
```

**Response Format:**
```json
{
    "success": true,
    "gesture": "YES",
    "details": {
        "gesture_key": "gesture_yes",
        "confidence": 0.75,
        "features": {...}
    },
    "image_shape": [480, 640, 3],
    "processing_time": 234
}
```

### GET /gestures
Returns all available gesture mappings organized by category.

**Query Parameters:**
- `category` (optional): Filter gestures by category (letters, numbers, words, common, etc.)

### POST /gestures/register
Register a new custom gesture in the system.

**Request Format:**
```json
{
    "gesture_key": "my_custom_gesture",
    "gesture_value": "CUSTOM_WORD",
    "category": "custom"
}
```

### DELETE /gestures/unregister
Remove a gesture from the system.

**Request Format:**
```json
{
    "gesture_key": "gesture_to_remove"
}
```

### GET /gestures/categories
Returns all gesture categories with counts and gesture lists.

### GET /health
Health check endpoint.

## API Usage

### Predict Gesture

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
    "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
    "success": true,
    "gesture": "A",
    "message": "Successfully predicted gesture: A",
    "image_shape": [480, 640, 3],
    "details": {
        "gesture_key": "gesture_A",
        "confidence": 0.85,
        "features": {
            "mean_intensity": 127.5,
            "std_intensity": 45.2,
            "frame_shape": [480, 640, 3],
            "non_zero_pixels": 307200
        }
    }
}
```

### Get Supported Gestures

**Endpoint:** `GET /gestures`

**Response:**
```json
{
    "gesture_map": {
        "gesture_A": "A",
        "one_finger": "1",
        "thumbs_up": "THUMBS_UP",
        "..."
    },
    "supported_gestures": ["gesture_A", "gesture_B", "..."],
    "total_gestures": 51
}
```

### Example Usage with curl

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "your_base64_encoded_image_here"}'
```

### Example Usage with Python

```python
import requests
import base64

# Read and encode image
with open("gesture_image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Make API request
response = requests.post(
    "http://localhost:5000/predict",
    json={"image": encoded_string}
)

result = response.json()
print(f"Predicted gesture: {result['gesture']}")
```

## Project Structure

```
gesture-prediction-api/
├── app.py              # Main Flask application with API endpoints
├── model.py            # Gesture recognition model module
├── index.html          # Web interface for webcam gesture recognition
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## MediaPipe Hand Tracking Features

The `model.py` module now includes:

- **MediaPipe Hands Integration**: Real-time hand landmark detection with 21 key points
- **51 Supported Gestures**: Including sign language letters (A-Z), numbers (0-9), and common gestures
- **Advanced Feature Extraction**: 
  - Hand landmark coordinates (x, y, z)
  - Finger extension states
  - Hand angles and geometric relationships
  - Finger-to-finger distances
  - Hand orientation analysis
- **Rule-based Classification**: Intelligent gesture recognition based on:
  - Number of extended fingers
  - Finger positions and angles
  - Thumb-index distance for specific gestures
  - Hand shape analysis
- **High Accuracy Detection**: MediaPipe provides robust hand tracking with confidence scoring
- **Multi-hand Support**: Can detect and process up to 2 hands simultaneously

### Web Interface Usage

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Click "Start Camera"** to access your webcam
3. **Position your hand** in front of the camera
4. **Click "Capture Gesture"** to take a photo and get AI prediction
5. **View results** including the predicted gesture, confidence score, and processing details

## API Usage (for developers)

The API is running at `http://localhost:5000` with the new model fully integrated. The design is ready for future integration with actual machine learning models - simply replace the placeholder prediction logic in the `GesturePredictor` class methods. Update the `load_model()` method to load your trained model
3. Modify `preprocess_frame()` and `extract_features()` as needed
4. The API endpoints will automatically use the new model

## Technical Implementation

### MediaPipe Integration Details

- **Hand Landmark Detection**: 21 3D landmarks per hand including fingertips, joints, and wrist
- **Real-time Processing**: Optimized for live webcam input with minimal latency
- **Gesture Classification**: Rule-based system analyzing finger states and hand geometry
- **Confidence Scoring**: MediaPipe detection confidence combined with gesture classification confidence
- **Multi-format Support**: Handles various image formats (JPEG, PNG, etc.) with automatic preprocessing

### Advanced Gesture Classification

**Numeric Gestures (0-9):**
- **0**: Closed fist or O-shape with thumb and index
- **1**: Index finger extended
- **2**: Index and middle fingers in V-shape
- **3**: Index, middle, and ring fingers extended
- **4**: Four fingers extended (thumb folded)
- **5**: All five fingers extended
- **6**: Thumb and pinky extended (others folded)
- **7**: Thumb, index, and middle extended
- **8**: Middle, ring, and pinky extended (thumb and index folded)
- **9**: All fingers except pinky extended

**Alphabetic Gestures (A-Z ASL):**
- **A**: Closed fist with thumb alongside
- **B**: Four fingers straight up, thumb across palm
- **C**: Curved hand shape (C-curve)
- **D**: Index finger up, others form O with thumb
- **E**: Fingers folded, thumb across fingertips
- **F**: Index and thumb form circle, others extended
- **G**: Index finger pointing sideways
- **H**: Index and middle fingers horizontal
- **I**: Pinky extended, others folded
- **L**: Thumb and index form 90-degree angle
- **V**: Index and middle in V-shape
- **Y**: Thumb and pinky extended
- *And many more...*

**Common Gestures:**
- Thumbs up, peace sign, OK sign, fist, open palm
- Special cases: No gesture detection, unknown gesture handling

### Classification Algorithm Features

- **Multi-tier Classification**: Numeric → Alphabetic → Common gestures priority system
- **Geometric Analysis**: 
  - Finger extension state detection
  - Inter-finger distance calculations
  - Hand curvature analysis for C-shapes
  - Circular formation detection for O-shapes
  - Finger alignment scoring
  - Angular measurements for L-shapes
- **Confidence Thresholds**: 
  - Numeric gestures: 85% base confidence
  - Alphabetic gestures: 75% base confidence
  - Fallback gestures: 70% confidence
- **Advanced Features**:
  - Thumb position analysis relative to other fingers
  - Hand orientation detection
  - Finger crossing detection for complex letters
  - V-shape vs U-shape differentiation

### Performance Features

- **Efficient Processing**: Frame resizing for optimal performance
- **Robust Detection**: Works in various lighting conditions
- **Error Handling**: Graceful fallback when hands are not detected
- **Real-time Feedback**: Instant gesture recognition results
- **High Accuracy**: Rule-based system with geometric validation
