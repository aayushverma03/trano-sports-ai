"""
Model factory for dynamic model selection
Provides unified interface for detector and tracker selection
"""

from typing import Optional, Literal
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import detectors
from models.footandball_detector import FootAndBallDetector
from ball_detector import BallDetector
from pose_estimator import PoseEstimator

# Import trackers
from tracking.bytetrack_tracker import ByteTrackTracker
from player_tracker import PlayerTracker


def get_ball_detector(
    model_type: Literal['footandball', 'yolo', 'auto'] = 'auto',
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.3
):
    """
    Get ball detector instance
    
    Args:
        model_type: Type of detector ('footandball', 'yolo', 'auto')
                   'auto' tries FootAndBall first, falls back to YOLO
        model_path: Path to model weights
        confidence_threshold: Minimum detection confidence
        
    Returns:
        Detector instance (FootAndBallDetector or BallDetector)
    """
    if model_type == 'footandball':
        # Force FootAndBall (will error if not available)
        return FootAndBallDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            use_footandball=True
        )
    
    elif model_type == 'yolo':
        # Force YOLO
        return BallDetector(
            model_path=model_path or 'yolo11x.pt',
            confidence_threshold=confidence_threshold
        )
    
    else:  # 'auto'
        # Try FootAndBall, fall back to YOLO
        try:
            detector = FootAndBallDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                use_footandball=True
            )
            if detector.using_footandball:
                return detector
            else:
                # FootAndBall fell back to YOLO internally
                return detector
        except:
            # Fallback to simple YOLO
            return BallDetector(
                model_path=model_path or 'yolo11x.pt',
                confidence_threshold=confidence_threshold
            )


def get_pose_detector(
    model_path: str = 'yolo11x-pose.pt',
    confidence_threshold: float = 0.25
) -> PoseEstimator:
    """
    Get pose detector instance
    
    Args:
        model_path: Path to YOLO pose model
        confidence_threshold: Minimum detection confidence
        
    Returns:
        PoseEstimator instance
    """
    return PoseEstimator(
        model_path=model_path,
        confidence_threshold=confidence_threshold
    )


def get_tracker(
    tracker_type: Literal['bytetrack', 'kalman', 'auto'] = 'auto',
    max_distance: float = 150.0,
    track_thresh: float = 0.5,
    track_buffer: int = 30,
    match_thresh: float = 0.8
):
    """
    Get tracker instance
    
    Args:
        tracker_type: Type of tracker ('bytetrack', 'kalman', 'auto')
                     'auto' tries ByteTrack first, falls back to Kalman
        max_distance: Maximum distance for association
        track_thresh: Detection confidence threshold (ByteTrack)
        track_buffer: Frames to keep lost tracks (ByteTrack)
        match_thresh: Matching threshold (ByteTrack)
        
    Returns:
        Tracker instance (ByteTrackTracker or PlayerTracker)
    """
    if tracker_type == 'bytetrack':
        # Force ByteTrack (will error if not available)
        return ByteTrackTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            max_distance=max_distance,
            use_bytetrack=True
        )
    
    elif tracker_type == 'kalman':
        # Force Kalman-based simple tracker
        return PlayerTracker(
            max_distance=max_distance
        )
    
    else:  # 'auto'
        # Try ByteTrack, fall back to Kalman
        try:
            tracker = ByteTrackTracker(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                max_distance=max_distance,
                use_bytetrack=True
            )
            if tracker.using_bytetrack:
                return tracker
            else:
                # ByteTrack fell back to simple tracker internally
                return tracker
        except:
            # Fallback to simple Kalman tracker
            return PlayerTracker(max_distance=max_distance)


def get_unified_detector(
    detector_type: Literal['footandball', 'yolo', 'auto'] = 'auto',
    pose_model_path: str = 'yolo11x-pose.pt',
    ball_model_path: Optional[str] = None,
    pose_confidence: float = 0.25,
    ball_confidence: float = 0.3
) -> dict:
    """
    Get unified detector setup with both pose and ball detection
    
    Args:
        detector_type: Type of detector system
        pose_model_path: Path to pose model
        ball_model_path: Path to ball detection model
        pose_confidence: Pose detection confidence threshold
        ball_confidence: Ball detection confidence threshold
        
    Returns:
        Dictionary with 'pose_detector', 'ball_detector', and 'unified'
        If unified detector (FootAndBall), it handles both
    """
    result = {
        'pose_detector': None,
        'ball_detector': None,
        'unified': False,
        'detector_type': detector_type
    }
    
    if detector_type == 'footandball':
        # FootAndBall handles both pose and ball
        detector = get_ball_detector('footandball', ball_model_path, ball_confidence)
        
        if detector.using_footandball:
            result['unified'] = True
            result['ball_detector'] = detector
            result['pose_detector'] = detector  # Same instance for both
        else:
            # Fell back to YOLO, need separate models
            result['pose_detector'] = get_pose_detector(pose_model_path, pose_confidence)
            result['ball_detector'] = detector
    
    else:
        # Use separate YOLO models
        result['pose_detector'] = get_pose_detector(pose_model_path, pose_confidence)
        result['ball_detector'] = get_ball_detector('yolo', ball_model_path, ball_confidence)
    
    return result


def print_model_info(detector=None, tracker=None):
    """
    Print information about active models
    
    Args:
        detector: Detector instance
        tracker: Tracker instance
    """
    print("\n" + "="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    
    if detector is not None:
        if hasattr(detector, 'get_model_info'):
            info = detector.get_model_info()
            print(f"Detector: {info.get('model_type', 'Unknown')}")
            print(f"  Using specialized model: {info.get('using_footandball', False)}")
            print(f"  Confidence threshold: {info.get('confidence_threshold', 'N/A')}")
        else:
            print(f"Detector: {detector.__class__.__name__}")
    
    if tracker is not None:
        if hasattr(tracker, 'get_tracker_info'):
            info = tracker.get_tracker_info()
            print(f"Tracker: {info.get('tracker_type', 'Unknown')}")
            print(f"  Using specialized tracker: {info.get('using_bytetrack', False)}")
        else:
            print(f"Tracker: {tracker.__class__.__name__}")
    
    print("="*60 + "\n")


# Presets for common configurations
MODEL_PRESETS = {
    'fast': {
        'pose_model': 'yolo11m-pose.pt',
        'ball_model': 'yolo11m.pt',
        'detector_type': 'yolo',
        'tracker_type': 'kalman',
        'description': 'Fast processing, good accuracy'
    },
    'balanced': {
        'pose_model': 'yolo11x-pose.pt',
        'ball_model': 'yolo11x.pt',
        'detector_type': 'auto',
        'tracker_type': 'auto',
        'description': 'Balanced speed and accuracy'
    },
    'accurate': {
        'pose_model': 'yolo11x-pose.pt',
        'ball_model': None,
        'detector_type': 'footandball',
        'tracker_type': 'bytetrack',
        'description': 'Maximum accuracy, slower'
    }
}


def get_preset_config(preset: Literal['fast', 'balanced', 'accurate'] = 'balanced') -> dict:
    """
    Get preset model configuration
    
    Args:
        preset: Preset name ('fast', 'balanced', 'accurate')
        
    Returns:
        Configuration dictionary
    """
    return MODEL_PRESETS.get(preset, MODEL_PRESETS['balanced']).copy()

