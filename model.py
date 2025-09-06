"""
Gesture Recognition Model Module

This module contains the gesture mapping and prediction functionality.
Integrates MediaPipe Hands for real-time hand landmark detection and
gesture classification based on hand pose analysis.
"""

import numpy as np
import random
import cv2
import mediapipe as mp
from typing import Dict, Any, Optional, List, Tuple

# Gesture mapping dictionary - maps gesture keys to output characters/numbers
gesture_map = {
    # Sign Language Letters
    'gesture_A': 'A',
    'gesture_B': 'B',
    'gesture_C': 'C',
    'gesture_D': 'D',
    'gesture_E': 'E',
    'gesture_F': 'F',
    'gesture_G': 'G',
    'gesture_H': 'H',
    'gesture_I': 'I',
    'gesture_J': 'J',
    'gesture_K': 'K',
    'gesture_L': 'L',
    'gesture_M': 'M',
    'gesture_N': 'N',
    'gesture_O': 'O',
    'gesture_P': 'P',
    'gesture_Q': 'Q',
    'gesture_R': 'R',
    'gesture_S': 'S',
    'gesture_T': 'T',
    'gesture_U': 'U',
    'gesture_V': 'V',
    'gesture_W': 'W',
    'gesture_X': 'X',
    'gesture_Y': 'Y',
    'gesture_Z': 'Z',
    
    # Numbers
    'zero_fingers': '0',
    'one_finger': '1',
    'two_fingers': '2',
    'three_fingers': '3',
    'four_fingers': '4',
    'five_fingers': '5',
    'six_fingers': '6',
    'seven_fingers': '7',
    'eight_fingers': '8',
    'nine_fingers': '9',
    
    # Common Gestures
    'thumbs_up': 'THUMBS_UP',
    'thumbs_down': 'THUMBS_DOWN',
    'peace_sign': 'PEACE',
    'ok_sign': 'OK',
    'pointing_up': 'POINT_UP',
    'pointing_left': 'POINT_LEFT',
    'pointing_right': 'POINT_RIGHT',
    'pointing_down': 'POINT_DOWN',
    'fist': 'FIST',
    'open_palm': 'PALM',
    'wave': 'WAVE',
    'rock_on': 'ROCK_ON',
    
    # Word Gestures
    'gesture_yes': 'YES',
    'gesture_no': 'NO',
    'gesture_mute': 'MUTE',
    'gesture_hello': 'HELLO',
    'gesture_goodbye': 'GOODBYE',
    'gesture_please': 'PLEASE',
    'gesture_thank_you': 'THANK_YOU',
    'gesture_stop': 'STOP',
    'gesture_go': 'GO',
    'gesture_help': 'HELP',
    
    # Special Cases
    'no_gesture': '',
    'unknown_gesture': '?',
    'multiple_gestures': '*'
}

# Extensible gesture categories for modular management
GESTURE_CATEGORIES = {
    'letters': {key: val for key, val in gesture_map.items() if key.startswith('gesture_') and len(val) == 1 and val.isalpha()},
    'numbers': {key: val for key, val in gesture_map.items() if key.endswith('_fingers') or key.startswith('zero_')},
    'words': {key: val for key, val in gesture_map.items() if key.startswith('gesture_') and len(val) > 1 and val.isupper() and '_' not in val},
    'common': {key: val for key, val in gesture_map.items() if key in ['thumbs_up', 'thumbs_down', 'peace_sign', 'ok_sign', 'fist', 'open_palm', 'wave', 'rock_on']},
    'pointing': {key: val for key, val in gesture_map.items() if key.startswith('pointing_')},
    'special': {key: val for key, val in gesture_map.items() if key in ['no_gesture', 'unknown_gesture', 'multiple_gestures']}
}

def register_gesture(gesture_key: str, gesture_value: str, category: str = 'custom') -> bool:
    """
    Register a new gesture in the mapping system.
    
    Args:
        gesture_key: Unique identifier for the gesture
        gesture_value: Output value/word for the gesture
        category: Category to organize the gesture
    
    Returns:
        bool: True if successfully registered, False if key already exists
    """
    if gesture_key in gesture_map:
        return False
    
    gesture_map[gesture_key] = gesture_value
    
    if category not in GESTURE_CATEGORIES:
        GESTURE_CATEGORIES[category] = {}
    
    GESTURE_CATEGORIES[category][gesture_key] = gesture_value
    return True

def unregister_gesture(gesture_key: str) -> bool:
    """
    Remove a gesture from the mapping system.
    
    Args:
        gesture_key: Gesture identifier to remove
    
    Returns:
        bool: True if successfully removed, False if key not found
    """
    if gesture_key not in gesture_map:
        return False
    
    # Remove from main map
    del gesture_map[gesture_key]
    
    # Remove from categories
    for category in GESTURE_CATEGORIES.values():
        if gesture_key in category:
            del category[gesture_key]
            break
    
    return True

def get_gestures_by_category(category: str) -> Dict[str, str]:
    """Get all gestures in a specific category."""
    return GESTURE_CATEGORIES.get(category, {})

def get_all_categories() -> List[str]:
    """Get list of all gesture categories."""
    return list(GESTURE_CATEGORIES.keys())

class GesturePredictor:
    """
    Gesture prediction class with MediaPipe Hands integration for real-time
    hand landmark detection and gesture classification.
    """
    
    def __init__(self):
        """Initialize the gesture predictor with MediaPipe Hands."""
        self.model_loaded = False
        self.confidence_threshold = 0.7
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Hand landmark indices for key points
        self.landmark_indices = {
            'thumb_tip': 4,
            'thumb_ip': 3,
            'thumb_mcp': 2,
            'index_tip': 8,
            'index_pip': 6,
            'index_mcp': 5,
            'middle_tip': 12,
            'middle_pip': 10,
            'middle_mcp': 9,
            'ring_tip': 16,
            'ring_pip': 14,
            'ring_mcp': 13,
            'pinky_tip': 20,
            'pinky_pip': 18,
            'pinky_mcp': 17,
            'wrist': 0
        }
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained gesture recognition model.
        
        Args:
            model_path (str, optional): Path to the trained model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Placeholder for future model loading
        # In a real implementation, this would load a trained ML model
        # e.g., TensorFlow, PyTorch, scikit-learn model
        
        if model_path:
            print(f"Model loading from {model_path} not implemented yet")
        
        # For now, just set the flag to indicate "model" is ready
        self.model_loaded = True
        return True
    
    def detect_hand_landmarks(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect hand landmarks using MediaPipe Hands.
        
        Args:
            frame (np.ndarray): Input image frame (BGR format)
            
        Returns:
            Dict[str, Any]: Detection results with landmarks and metadata
        """
        if frame is None or frame.size == 0:
            return {'landmarks': [], 'handedness': [], 'success': False}
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        detection_data = {
            'landmarks': [],
            'handedness': [],
            'success': False,
            'num_hands': 0
        }
        
        if results.multi_hand_landmarks:
            detection_data['success'] = True
            detection_data['num_hands'] = len(results.multi_hand_landmarks)
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
                
                detection_data['landmarks'].append(landmarks)
                
                # Get handedness (left/right hand)
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                    detection_data['handedness'].append(handedness)
                else:
                    detection_data['handedness'].append('Unknown')
        
        return detection_data

    def extract_landmark_coordinates(self, landmarks: List[Dict[str, float]]) -> np.ndarray:
        """
        Extract landmark coordinates as a numpy array.
        
        Args:
            landmarks (List[Dict[str, float]]): List of landmark dictionaries
            
        Returns:
            np.ndarray: Flattened array of landmark coordinates (x, y, z)
        """
        if not landmarks:
            return np.zeros(63)  # 21 landmarks * 3 coordinates
        
        coords = []
        for landmark in landmarks:
            coords.extend([landmark['x'], landmark['y'], landmark['z']])
        
        return np.array(coords)

    def calculate_finger_angles(self, landmarks: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate angles between finger joints for gesture analysis.
        
        Args:
            landmarks (List[Dict[str, float]]): Hand landmarks
            
        Returns:
            Dict[str, float]: Finger angles and ratios
        """
        if len(landmarks) < 21:
            return {}
        
        def calculate_angle(p1, p2, p3):
            """Calculate angle between three points."""
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        
        angles = {}
        
        # Finger tip to wrist distances (normalized)
        wrist = landmarks[0]
        for finger, tip_idx in [('thumb', 4), ('index', 8), ('middle', 12), ('ring', 16), ('pinky', 20)]:
            tip = landmarks[tip_idx]
            distance = np.sqrt((tip['x'] - wrist['x'])**2 + (tip['y'] - wrist['y'])**2)
            angles[f'{finger}_distance'] = distance
        
        # Finger bend angles
        finger_joints = [
            ('thumb', [2, 3, 4]),
            ('index', [5, 6, 8]),
            ('middle', [9, 10, 12]),
            ('ring', [13, 14, 16]),
            ('pinky', [17, 18, 20])
        ]
        
        for finger_name, joint_indices in finger_joints:
            if len(joint_indices) >= 3:
                angle = calculate_angle(
                    landmarks[joint_indices[0]], 
                    landmarks[joint_indices[1]], 
                    landmarks[joint_indices[2]]
                )
                angles[f'{finger_name}_angle'] = angle
        
        return angles

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the input frame for MediaPipe processing.
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            np.ndarray: Preprocessed frame in BGR format
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Ensure frame is in BGR format for MediaPipe
        if len(frame.shape) == 3:
            # If frame has 3 channels, assume it's already in correct format
            processed_frame = frame.copy()
        else:
            # Convert grayscale to BGR
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Resize frame if too large (for performance)
        height, width = processed_frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            processed_frame = cv2.resize(processed_frame, (new_width, new_height))
        
        return processed_frame
    
    def extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from the preprocessed frame using MediaPipe hand landmarks.
        
        Args:
            frame (np.ndarray): Preprocessed image frame
            
        Returns:
            Dict[str, Any]: Extracted features including hand landmarks and geometric data
        """
        features = {
            'frame_shape': frame.shape,
            'has_hands': False,
            'num_hands': 0,
            'landmarks': [],
            'hand_angles': [],
            'gesture_features': {}
        }
        
        if frame.size == 0:
            return features
        
        # Detect hand landmarks
        detection_results = self.detect_hand_landmarks(frame)
        
        if detection_results['success']:
            features['has_hands'] = True
            features['num_hands'] = detection_results['num_hands']
            features['handedness'] = detection_results['handedness']
            
            # Process each detected hand
            for i, hand_landmarks in enumerate(detection_results['landmarks']):
                # Extract coordinate array
                landmark_coords = self.extract_landmark_coordinates(hand_landmarks)
                features['landmarks'].append(landmark_coords.tolist())
                
                # Calculate finger angles and geometric features
                angles = self.calculate_finger_angles(hand_landmarks)
                features['hand_angles'].append(angles)
                
                # Extract gesture-specific features
                gesture_features = self.extract_gesture_features(hand_landmarks)
                features['gesture_features'][f'hand_{i}'] = gesture_features
        
        # Fallback to basic image features if no hands detected
        if not features['has_hands']:
            features.update({
                'mean_intensity': float(np.mean(frame)) if frame.size > 0 else 0.0,
                'std_intensity': float(np.std(frame)) if frame.size > 0 else 0.0,
                'non_zero_pixels': int(np.count_nonzero(frame)) if frame.size > 0 else 0
            })
        
        return features

    def extract_gesture_features(self, landmarks: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Extract specific gesture features for classification.
        
        Args:
            landmarks (List[Dict[str, float]]): Hand landmarks
            
        Returns:
            Dict[str, Any]: Gesture-specific features
        """
        if len(landmarks) < 21:
            return {}
        
        features = {}
        
        # Finger extension states (tip higher than pip joint)
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_pips = [3, 6, 10, 14, 18]
        
        extended_fingers = []
        for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips)):
            tip_y = landmarks[tip_idx]['y']
            pip_y = landmarks[pip_idx]['y']
            
            # For thumb, check x-axis extension instead
            if i == 0:  # thumb
                tip_x = landmarks[tip_idx]['x']
                mcp_x = landmarks[2]['x']  # thumb MCP
                extended = abs(tip_x - mcp_x) > 0.05
            else:
                extended = tip_y < pip_y  # tip above pip (extended)
            
            extended_fingers.append(extended)
        
        features['extended_fingers'] = extended_fingers
        features['num_extended'] = sum(extended_fingers)
        
        # Hand openness (distance between fingertips)
        if features['num_extended'] >= 2:
            distances = []
            for i in range(len(finger_tips)):
                for j in range(i + 1, len(finger_tips)):
                    if extended_fingers[i] and extended_fingers[j]:
                        tip1 = landmarks[finger_tips[i]]
                        tip2 = landmarks[finger_tips[j]]
                        dist = np.sqrt((tip1['x'] - tip2['x'])**2 + (tip1['y'] - tip2['y'])**2)
                        distances.append(dist)
            
            features['avg_fingertip_distance'] = np.mean(distances) if distances else 0
            features['max_fingertip_distance'] = np.max(distances) if distances else 0
        
        # Thumb position relative to other fingers
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        features['thumb_index_distance'] = np.sqrt(
            (thumb_tip['x'] - index_tip['x'])**2 + (thumb_tip['y'] - index_tip['y'])**2
        )
        
        # Hand orientation (wrist to middle finger direction)
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        features['hand_angle'] = np.arctan2(
            middle_mcp['y'] - wrist['y'], 
            middle_mcp['x'] - wrist['x']
        )
        
        return features
    
    def classify_numeric_gesture(self, landmarks: List[Dict[str, float]], gesture_features: Dict[str, Any]) -> tuple:
        """
        Classify numeric gestures (0-9) based on hand landmarks and finger positions.
        
        Args:
            landmarks (List[Dict[str, float]]): Hand landmarks
            gesture_features (Dict[str, Any]): Extracted gesture features
            
        Returns:
            tuple: (gesture_key, confidence_score)
        """
        extended_fingers = gesture_features.get('extended_fingers', [False] * 5)
        num_extended = gesture_features.get('num_extended', 0)
        thumb_index_distance = gesture_features.get('thumb_index_distance', 0)
        
        # Advanced finger position analysis
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate relative positions
        wrist = landmarks[0]
        palm_center = landmarks[9]  # Middle finger MCP
        
        confidence = 0.85
        
        # Number 0 - Closed fist or O-shape
        if num_extended == 0:
            return 'zero_fingers', confidence
        elif thumb_index_distance < 0.03 and extended_fingers[0] and extended_fingers[1]:
            # O-shape with thumb and index forming circle
            return 'zero_fingers', confidence + 0.1
        
        # Number 1 - Index finger extended
        elif num_extended == 1 and extended_fingers[1]:
            return 'one_finger', confidence
        
        # Number 2 - Index and middle fingers (peace sign)
        elif num_extended == 2:
            if extended_fingers[1] and extended_fingers[2]:
                # Check if fingers are spread (V-shape)
                index_middle_distance = np.sqrt(
                    (index_tip['x'] - middle_tip['x'])**2 + 
                    (index_tip['y'] - middle_tip['y'])**2
                )
                if index_middle_distance > 0.05:
                    return 'two_fingers', confidence
            elif extended_fingers[0] and extended_fingers[1]:
                return 'two_fingers', confidence - 0.1
        
        # Number 3 - Index, middle, and ring fingers
        elif num_extended == 3:
            if extended_fingers[1] and extended_fingers[2] and extended_fingers[3]:
                return 'three_fingers', confidence
            elif extended_fingers[0] and extended_fingers[1] and extended_fingers[2]:
                return 'three_fingers', confidence - 0.1
        
        # Number 4 - Four fingers (thumb folded)
        elif num_extended == 4:
            if not extended_fingers[0] and all(extended_fingers[1:]):
                return 'four_fingers', confidence
        
        # Number 5 - All fingers extended
        elif num_extended == 5:
            return 'five_fingers', confidence
        
        # Numbers 6-9 require more complex hand shapes
        # Number 6 - Thumb and pinky extended, others folded
        if extended_fingers[0] and extended_fingers[4] and not any(extended_fingers[1:4]):
            return 'six_fingers', confidence - 0.1
        
        # Number 7 - Thumb, index, middle extended
        elif extended_fingers[0] and extended_fingers[1] and extended_fingers[2] and not extended_fingers[3] and not extended_fingers[4]:
            return 'seven_fingers', confidence - 0.1
        
        # Number 8 - Thumb and index folded, others extended
        elif not extended_fingers[0] and not extended_fingers[1] and all(extended_fingers[2:]):
            return 'eight_fingers', confidence - 0.1
        
        # Number 9 - Only pinky folded
        elif all(extended_fingers[:4]) and not extended_fingers[4]:
            return 'nine_fingers', confidence - 0.1
        
        return None, 0

    def classify_alphabetic_gesture(self, landmarks: List[Dict[str, float]], gesture_features: Dict[str, Any]) -> tuple:
        """
        Classify alphabetic gestures (A-Z) based on ASL hand shapes.
        
        Args:
            landmarks (List[Dict[str, float]]): Hand landmarks
            gesture_features (Dict[str, Any]): Extracted gesture features
            
        Returns:
            tuple: (gesture_key, confidence_score)
        """
        extended_fingers = gesture_features.get('extended_fingers', [False] * 5)
        num_extended = gesture_features.get('num_extended', 0)
        thumb_index_distance = gesture_features.get('thumb_index_distance', 0)
        
        # Key landmark positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        confidence = 0.75  # Slightly lower for alphabet as they're more complex
        
        # Letter A - Closed fist with thumb alongside
        if not any(extended_fingers[1:]) and extended_fingers[0]:
            # Check if thumb is positioned alongside the fist
            thumb_side_position = abs(thumb_tip['x'] - landmarks[5]['x']) < 0.05
            if thumb_side_position:
                return 'gesture_A', confidence
        
        # Letter B - Four fingers extended, thumb folded across palm
        elif all(extended_fingers[1:]) and not extended_fingers[0]:
            # Check if fingers are straight and together
            finger_alignment = self._check_finger_alignment(landmarks, [8, 12, 16, 20])
            if finger_alignment > 0.7:
                return 'gesture_B', confidence
        
        # Letter C - Curved hand shape (C-shape)
        elif num_extended >= 3:
            # Check for C-curve by analyzing finger curvature
            curve_score = self._analyze_hand_curvature(landmarks)
            if curve_score > 0.6:
                return 'gesture_C', confidence - 0.1
        
        # Letter D - Index finger extended, others form O with thumb
        elif extended_fingers[1] and not any(extended_fingers[2:]):
            # Check if thumb and other fingers form circular shape
            circle_score = self._check_circular_formation(landmarks, [4, 12, 16, 20])
            if circle_score > 0.5:
                return 'gesture_D', confidence
        
        # Letter E - All fingers folded, thumb across fingertips
        elif not any(extended_fingers):
            # Check thumb position relative to fingertips
            thumb_over_fingers = thumb_tip['y'] < index_tip['y']
            if thumb_over_fingers:
                return 'gesture_E', confidence
        
        # Letter F - Index and thumb form circle, others extended
        elif extended_fingers[2] and extended_fingers[3] and extended_fingers[4]:
            if thumb_index_distance < 0.04:  # Close circle formation
                return 'gesture_F', confidence
        
        # Letter G - Index finger pointing sideways
        elif extended_fingers[1] and not any(extended_fingers[2:]):
            # Check if index finger is horizontal
            index_horizontal = abs(index_tip['y'] - landmarks[5]['y']) < 0.03
            if index_horizontal:
                return 'gesture_G', confidence - 0.1
        
        # Letter H - Index and middle fingers extended sideways
        elif extended_fingers[1] and extended_fingers[2] and not any(extended_fingers[3:]):
            # Check if fingers are horizontal and parallel
            fingers_horizontal = (abs(index_tip['y'] - middle_tip['y']) < 0.02 and
                                abs(index_tip['y'] - landmarks[5]['y']) < 0.03)
            if fingers_horizontal:
                return 'gesture_H', confidence - 0.1
        
        # Letter I - Pinky extended, others folded
        elif extended_fingers[4] and not any(extended_fingers[:4]):
            return 'gesture_I', confidence
        
        # Letter J - Pinky extended with motion (static detection)
        elif extended_fingers[4] and not any(extended_fingers[:4]):
            # J is similar to I but with motion - for static detection, treat as I
            return 'gesture_J', confidence - 0.2
        
        # Letter K - Index and middle in V, thumb between them
        elif extended_fingers[1] and extended_fingers[2]:
            # Check thumb position between index and middle
            thumb_between = (thumb_tip['x'] > min(index_tip['x'], middle_tip['x']) and
                           thumb_tip['x'] < max(index_tip['x'], middle_tip['x']))
            if thumb_between:
                return 'gesture_K', confidence - 0.1
        
        # Letter L - Thumb and index form L-shape
        elif extended_fingers[0] and extended_fingers[1] and not any(extended_fingers[2:]):
            # Check for 90-degree angle between thumb and index
            angle = self._calculate_finger_angle(landmarks[4], landmarks[5], landmarks[8])
            if 80 <= angle <= 100:  # Approximately 90 degrees
                return 'gesture_L', confidence
        
        # Letter M - Three fingers over thumb
        elif not extended_fingers[0] and extended_fingers[1] and extended_fingers[2] and extended_fingers[3]:
            return 'gesture_M', confidence - 0.1
        
        # Letter N - Two fingers over thumb
        elif not extended_fingers[0] and extended_fingers[1] and extended_fingers[2]:
            return 'gesture_N', confidence - 0.1
        
        # Letter O - All fingers form circle
        elif num_extended >= 4:
            circle_score = self._check_circular_formation(landmarks, [4, 8, 12, 16, 20])
            if circle_score > 0.6:
                return 'gesture_O', confidence - 0.1
        
        # Letter P - Similar to K but pointing down
        elif extended_fingers[1] and extended_fingers[2]:
            # Check if hand is tilted downward
            hand_tilt = middle_tip['y'] > landmarks[9]['y']
            if hand_tilt:
                return 'gesture_P', confidence - 0.2
        
        # Letter Q - Similar to G but with thumb and index
        elif extended_fingers[0] and extended_fingers[1]:
            return 'gesture_Q', confidence - 0.2
        
        # Letter R - Index and middle crossed
        elif extended_fingers[1] and extended_fingers[2]:
            # Check if fingers are crossed
            fingers_crossed = abs(index_tip['x'] - middle_tip['x']) < 0.02
            if fingers_crossed:
                return 'gesture_R', confidence - 0.1
        
        # Letter S - Closed fist with thumb over fingers
        elif not any(extended_fingers):
            # Check thumb position over fist
            thumb_over_fist = thumb_tip['y'] < landmarks[9]['y']
            if thumb_over_fist:
                return 'gesture_S', confidence
        
        # Letter T - Thumb between index and middle
        elif not extended_fingers[0] and extended_fingers[1]:
            return 'gesture_T', confidence - 0.2
        
        # Letter U - Index and middle together, pointing up
        elif extended_fingers[1] and extended_fingers[2] and not any(extended_fingers[3:]):
            fingers_together = abs(index_tip['x'] - middle_tip['x']) < 0.02
            if fingers_together:
                return 'gesture_U', confidence
        
        # Letter V - Index and middle in V-shape
        elif extended_fingers[1] and extended_fingers[2] and not any(extended_fingers[3:]):
            # Check for V separation
            v_separation = abs(index_tip['x'] - middle_tip['x']) > 0.04
            if v_separation:
                return 'gesture_V', confidence
        
        # Letter W - Three fingers in W-shape
        elif extended_fingers[1] and extended_fingers[2] and extended_fingers[3]:
            return 'gesture_W', confidence - 0.1
        
        # Letter X - Index finger bent (hook shape)
        elif not any(extended_fingers[1:]):
            # Check for bent index finger
            index_bent = index_tip['y'] > index_pip['y']
            if index_bent:
                return 'gesture_X', confidence - 0.2
        
        # Letter Y - Thumb and pinky extended
        elif extended_fingers[0] and extended_fingers[4] and not any(extended_fingers[1:4]):
            return 'gesture_Y', confidence
        
        # Letter Z - Index finger tracing Z (static detection difficult)
        elif extended_fingers[1] and not any(extended_fingers[2:]):
            return 'gesture_Z', confidence - 0.3
        
        return None, 0

    def _check_finger_alignment(self, landmarks: List[Dict[str, float]], finger_indices: List[int]) -> float:
        """Check how well aligned the specified fingers are."""
        if len(finger_indices) < 2:
            return 0.0
        
        y_positions = [landmarks[idx]['y'] for idx in finger_indices]
        y_variance = np.var(y_positions)
        
        # Lower variance means better alignment
        alignment_score = max(0, 1.0 - y_variance * 10)
        return alignment_score

    def _analyze_hand_curvature(self, landmarks: List[Dict[str, float]]) -> float:
        """Analyze the curvature of the hand for C-like shapes."""
        # Check curvature by analyzing the arc formed by fingertips
        fingertips = [landmarks[i] for i in [8, 12, 16, 20]]
        
        # Calculate if fingertips form an arc
        x_coords = [tip['x'] for tip in fingertips]
        y_coords = [tip['y'] for tip in fingertips]
        
        # Simple curvature check - fingertips should form a curved line
        if len(set(x_coords)) > 2:  # Need variation in x
            curvature_score = 1.0 - (max(y_coords) - min(y_coords)) / 2
            return max(0, min(1, curvature_score))
        
        return 0.0

    def _check_circular_formation(self, landmarks: List[Dict[str, float]], finger_indices: List[int]) -> float:
        """Check if specified fingers form a circular shape."""
        if len(finger_indices) < 3:
            return 0.0
        
        # Calculate center point
        center_x = np.mean([landmarks[idx]['x'] for idx in finger_indices])
        center_y = np.mean([landmarks[idx]['y'] for idx in finger_indices])
        
        # Calculate distances from center
        distances = []
        for idx in finger_indices:
            dist = np.sqrt((landmarks[idx]['x'] - center_x)**2 + 
                          (landmarks[idx]['y'] - center_y)**2)
            distances.append(dist)
        
        # Check if distances are similar (circular)
        distance_variance = np.var(distances)
        circle_score = max(0, 1.0 - distance_variance * 20)
        
        return circle_score

    def _calculate_finger_angle(self, point1: Dict[str, float], point2: Dict[str, float], point3: Dict[str, float]) -> float:
        """Calculate angle between three points in degrees."""
        v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def predict_from_features(self, features: Dict[str, Any]) -> tuple:
        """
        Enhanced gesture prediction using advanced classification logic.
        
        Args:
            features (Dict[str, Any]): Extracted features including hand landmarks
            
        Returns:
            tuple: (gesture_key, confidence_score)
        """
        # Check if hands are detected
        if not features.get('has_hands', False):
            return 'no_gesture', 0.1
        
        # Get landmarks and gesture features for the first detected hand
        landmarks = features.get('landmarks', [])
        if not landmarks or len(landmarks[0]) < 63:  # 21 landmarks * 3 coordinates
            return 'unknown_gesture', 0.2
        
        # Convert flat coordinate array back to landmark format
        hand_landmarks = []
        coords = landmarks[0]
        for i in range(0, len(coords), 3):
            if i + 2 < len(coords):
                hand_landmarks.append({
                    'x': coords[i],
                    'y': coords[i + 1],
                    'z': coords[i + 2]
                })
        
        gesture_features = features.get('gesture_features', {}).get('hand_0', {})
        if not gesture_features:
            return 'unknown_gesture', 0.2
        
        # Try numeric classification first
        numeric_result, numeric_confidence = self.classify_numeric_gesture(hand_landmarks, gesture_features)
        if numeric_result and numeric_confidence > 0.6:
            return numeric_result, numeric_confidence
        
        # Try alphabetic classification
        alpha_result, alpha_confidence = self.classify_alphabetic_gesture(hand_landmarks, gesture_features)
        if alpha_result and alpha_confidence > 0.5:
            return alpha_result, alpha_confidence
        
        # Try word gesture classification
        word_result, word_confidence = self.classify_word_gesture(hand_landmarks, gesture_features)
        if word_result and word_confidence > 0.5:
            return word_result, word_confidence
        
        # Fallback to basic gesture classification
        extended_fingers = gesture_features.get('extended_fingers', [False] * 5)
        num_extended = gesture_features.get('num_extended', 0)
        thumb_index_distance = gesture_features.get('thumb_index_distance', 0)
        
        # Common gestures
        if all(extended_fingers):
            return 'open_palm', 0.7
        elif not any(extended_fingers):
            return 'fist', 0.7
        elif extended_fingers[0] and not any(extended_fingers[1:]):
            return 'thumbs_up', 0.8
        elif thumb_index_distance < 0.04 and extended_fingers[0] and extended_fingers[1]:
            return 'ok_sign', 0.75
        
        return 'unknown_gesture', 0.3
    
    def classify_word_gesture(self, hand_landmarks: List[Dict], gesture_features: Dict) -> Tuple[str, float]:
        """
        Classify word gestures like YES, NO, MUTE, etc.
        
        Args:
            hand_landmarks: List of hand landmark dictionaries
            gesture_features: Extracted gesture features
            
        Returns:
            Tuple[str, float]: (gesture_key, confidence)
        """
        if not hand_landmarks:
            return None, 0.0
        
        landmarks = hand_landmarks
        extended_fingers = gesture_features.get('extended_fingers', [False] * 5)
        num_extended = gesture_features.get('num_extended', 0)
        
        # Get key landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        confidence = 0.75
        
        # YES - Thumbs up gesture
        if extended_fingers[0] and not any(extended_fingers[1:]):
            # Check if thumb is pointing up
            thumb_up = thumb_tip['y'] < wrist['y']
            if thumb_up:
                return 'gesture_yes', confidence
        
        # NO - Index finger wagging or flat hand side to side
        elif extended_fingers[1] and not any([extended_fingers[0], extended_fingers[2], extended_fingers[3], extended_fingers[4]]):
            # Single index finger extended (wagging motion)
            return 'gesture_no', confidence - 0.1
        elif all(extended_fingers):
            # Open palm can also mean NO (stop gesture)
            return 'gesture_no', confidence - 0.2
        
        # MUTE - Index finger over lips (pointing to mouth area)
        elif extended_fingers[1] and not any([extended_fingers[2], extended_fingers[3], extended_fingers[4]]):
            # Check if index is pointing toward center (mouth area)
            pointing_center = abs(index_tip['x'] - 0.5) < 0.1
            if pointing_center:
                return 'gesture_mute', confidence - 0.1
        
        # HELLO - Open palm facing forward or wave
        elif all(extended_fingers):
            # Open palm greeting
            return 'gesture_hello', confidence - 0.1
        
        # GOODBYE - Wave gesture (open palm)
        elif all(extended_fingers):
            return 'gesture_goodbye', confidence - 0.2
        
        # PLEASE - Open palm with slight cupping
        elif all(extended_fingers):
            # Check for slight hand curvature
            hand_curvature = self._analyze_hand_curvature(landmarks)
            if hand_curvature > 0.3:
                return 'gesture_please', confidence - 0.1
        
        # THANK_YOU - Open palm touching chest area or flat palm forward
        elif all(extended_fingers):
            # Palm forward gesture
            return 'gesture_thank_you', confidence - 0.2
        
        # STOP - Open palm facing forward (stop sign)
        elif all(extended_fingers):
            # Check if palm is facing forward (all fingers extended)
            return 'gesture_stop', confidence
        
        # GO - Pointing gesture or thumbs up
        elif extended_fingers[1] and not any([extended_fingers[0], extended_fingers[2], extended_fingers[3], extended_fingers[4]]):
            # Index finger pointing forward
            return 'gesture_go', confidence - 0.1
        elif extended_fingers[0] and not any(extended_fingers[1:]):
            # Thumbs up can also mean GO
            return 'gesture_go', confidence - 0.2
        
        # HELP - Open palm or specific help gesture
        elif all(extended_fingers):
            return 'gesture_help', confidence - 0.3
        
        return None, 0.0
    
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main prediction method that processes frame and returns gesture prediction.
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            Dict[str, Any]: Prediction results including gesture, character, and confidence
        """
        try:
            # Preprocess the frame
            processed_frame = self.preprocess_frame(frame)
            
            # Extract features
            features = self.extract_features(processed_frame)
            
            # Get prediction
            gesture_key, confidence = self.predict_from_features(features)
            
            # Map gesture key to output character
            output_character = gesture_map.get(gesture_key, gesture_map['unknown_gesture'])
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                gesture_key = 'unknown_gesture'
                output_character = gesture_map['unknown_gesture']
            
            return {
                'gesture_key': gesture_key,
                'character': output_character,
                'confidence': confidence,
                'features': features,
                'success': True
            }
            
        except Exception as e:
            return {
                'gesture_key': 'unknown_gesture',
                'character': gesture_map['unknown_gesture'],
                'confidence': 0.0,
                'error': str(e),
                'success': False
            }

# Global predictor instance
_predictor = GesturePredictor()

def predict_gesture(frame: np.ndarray) -> str:
    """
    Simplified prediction function that returns just the character.
    This maintains compatibility with the existing API while using the new model structure.
    
    Args:
        frame (np.ndarray): Input image frame as NumPy array
        
    Returns:
        str: Predicted gesture character
    """
    global _predictor
    
    # Initialize predictor if not already done
    if not _predictor.model_loaded:
        _predictor.load_model()
    
    # Get prediction
    result = _predictor.predict(frame)
    
    return result.get('character', gesture_map['unknown_gesture'])

def get_detailed_prediction(frame: np.ndarray) -> Dict[str, Any]:
    """
    Get detailed prediction results including confidence and features.
    
    Args:
        frame (np.ndarray): Input image frame as NumPy array
        
    Returns:
        Dict[str, Any]: Detailed prediction results
    """
    global _predictor
    
    # Initialize predictor if not already done
    if not _predictor.model_loaded:
        _predictor.load_model()
    
    return _predictor.predict(frame)

def get_gesture_map() -> Dict[str, str]:
    """
    Get the complete gesture mapping dictionary.
    
    Returns:
        Dict[str, str]: Gesture map
    """
    return gesture_map.copy()

def get_supported_gestures() -> list:
    """
    Get list of supported gesture keys.
    
    Returns:
        list: List of gesture keys
    """
    return list(gesture_map.keys())

# Example usage and testing functions
if __name__ == "__main__":
    # Test the model with a dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("Testing Gesture Model...")
    print(f"Supported gestures: {len(get_supported_gestures())}")
    
    # Test simple prediction
    prediction = predict_gesture(test_frame)
    print(f"Simple prediction: {prediction}")
    
    # Test detailed prediction
    detailed = get_detailed_prediction(test_frame)
    print(f"Detailed prediction: {detailed}")
    
    print("\nGesture Map Sample:")
    for i, (key, value) in enumerate(gesture_map.items()):
        if i < 10:  # Show first 10 mappings
            print(f"  {key}: {value}")
        elif i == 10:
            print(f"  ... and {len(gesture_map) - 10} more")
            break
