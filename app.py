from flask import Flask, request, jsonify, render_template_string, send_from_directory
import base64
import numpy as np
from PIL import Image
import io
from io import BytesIO
import time
import cv2
from model import GesturePredictor
from functools import lru_cache
import threading
import queue
from model import predict_gesture, get_detailed_prediction, get_gesture_map, get_supported_gestures, register_gesture, unregister_gesture, get_gestures_by_category, get_all_categories, GESTURE_CATEGORIES
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST endpoint that accepts base64 encoded image and returns predicted gesture.
    
    Expected JSON format:
    {
        "image": "base64_encoded_image_string"
    }
    
    Returns JSON format:
    {
        "gesture": "predicted_character",
        "success": true/false,
        "message": "status_message"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'No image data provided. Expected JSON with "image" field containing base64 encoded image.',
                'gesture': None
            }), 400
        
        # Get base64 encoded image
        base64_image = data['image']
        
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_image)
        
        # Convert bytes to PIL Image
        pil_image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL Image to NumPy array
        frame = np.array(pil_image)
        
        # Convert to RGB if image has alpha channel
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            # Image is already RGB, but PIL uses RGB while OpenCV uses BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Call the gesture prediction function
        predicted_gesture = predict_gesture(frame)
        
        # Get detailed prediction for additional information
        detailed_prediction = get_detailed_prediction(frame)
        
        return jsonify({
            'success': True,
            'gesture': predicted_gesture,
            'message': f'Successfully predicted gesture: {predicted_gesture}',
            'image_shape': frame.shape,
            'details': {
                'gesture_key': detailed_prediction.get('gesture_key'),
                'confidence': detailed_prediction.get('confidence'),
                'features': detailed_prediction.get('features', {})
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}',
            'gesture': None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'message': 'Gesture Prediction API is running'
    }), 200

@app.route('/gestures', methods=['GET'])
def get_gestures():
    """
    GET endpoint that returns all available gesture mappings organized by category.
    
    Returns:
        JSON: Dictionary of all gesture mappings with categories
    """
    category = request.args.get('category')
    
    if category:
        gestures = get_gestures_by_category(category)
        return jsonify({
            'success': True,
            'category': category,
            'gestures': gestures,
            'total_gestures': len(gestures)
        })
    
    return jsonify({
        'success': True,
        'gestures': get_gesture_map(),
        'categories': GESTURE_CATEGORIES,
        'available_categories': get_all_categories(),
        'total_gestures': len(get_gesture_map())
    })

@app.route('/gestures/register', methods=['POST'])
def register_new_gesture():
    """
    POST endpoint to register a new gesture in the system.
    
    Expected JSON format:
    {
        "gesture_key": "my_custom_gesture",
        "gesture_value": "CUSTOM",
        "category": "custom"
    }
    
    Returns:
        JSON: Success status and message
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No JSON data provided'
            }), 400
        
        gesture_key = data.get('gesture_key')
        gesture_value = data.get('gesture_value')
        category = data.get('category', 'custom')
        
        if not gesture_key or not gesture_value:
            return jsonify({
                'success': False,
                'message': 'gesture_key and gesture_value are required'
            }), 400
        
        success = register_gesture(gesture_key, gesture_value, category)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Gesture "{gesture_key}" registered successfully',
                'gesture_key': gesture_key,
                'gesture_value': gesture_value,
                'category': category
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Gesture key "{gesture_key}" already exists'
            }), 409
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error registering gesture: {str(e)}'
        }), 500

@app.route('/gestures/unregister', methods=['DELETE'])
def unregister_existing_gesture():
    """
    DELETE endpoint to remove a gesture from the system.
    
    Expected JSON format:
    {
        "gesture_key": "gesture_to_remove"
    }
    
    Returns:
        JSON: Success status and message
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No JSON data provided'
            }), 400
        
        gesture_key = data.get('gesture_key')
        
        if not gesture_key:
            return jsonify({
                'success': False,
                'message': 'gesture_key is required'
            }), 400
        
        success = unregister_gesture(gesture_key)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Gesture "{gesture_key}" removed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Gesture key "{gesture_key}" not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error removing gesture: {str(e)}'
        }), 500

@app.route('/gestures/categories', methods=['GET'])
def get_gesture_categories():
    """
    GET endpoint that returns all gesture categories and their contents.
    
    Returns:
        JSON: Dictionary of categories with gesture counts
    """
    categories_info = {}
    for category in get_all_categories():
        gestures = get_gestures_by_category(category)
        categories_info[category] = {
            'count': len(gestures),
            'gestures': list(gestures.keys())
        }
    
    return jsonify({
        'success': True,
        'categories': categories_info,
        'total_categories': len(categories_info)
    }), 200

@app.route('/')
def index():
    """Serve the main HTML interface."""
    return send_from_directory('.', 'index.html')

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint."""
    return jsonify({
        'message': 'Gesture Prediction API',
        'version': '2.0.0',
        'endpoints': {
            'GET /': 'Main web interface',
            'POST /predict': 'Predict gesture from base64 encoded image',
            'GET /gestures': 'Get all supported gestures and mappings',
            'GET /health': 'Health check',
            'GET /api': 'API information'
        },
        'usage': {
            'predict_endpoint': {
                'method': 'POST',
                'content_type': 'application/json',
                'body': {
                    'image': 'base64_encoded_image_string'
                },
                'response': {
                    'success': 'boolean',
                    'gesture': 'predicted_character',
                    'message': 'status_message',
                    'details': {
                        'gesture_key': 'internal_gesture_name',
                        'confidence': 'prediction_confidence_score',
                        'features': 'extracted_image_features'
                    }
                }
            }
        }
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
