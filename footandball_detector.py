"""
FootAndBall detector integration for soccer-specific detection
Falls back to YOLO if FootAndBall model is not available
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings

# Try to import FootAndBall model
FOOTANDBALL_AVAILABLE = False
try:
    # Placeholder for actual FootAndBall import
    # from footandball import FootAndBallModel
    FOOTANDBALL_AVAILABLE = False  # Set to True when model is available
except ImportError:
    FOOTANDBALL_AVAILABLE = False

# Fallback to YOLO
from ultralytics import YOLO


class FootAndBallDetector:
    """
    Soccer-specific detector for players and ball
    Uses FootAndBall model if available, falls back to YOLO
    
    FootAndBall achieves 90-95% accuracy vs 80-87% for standard YOLO
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.3,
                 use_footandball: bool = True):
        """
        Initialize FootAndBall detector
        
        Args:
            model_path: Path to model weights (FootAndBall or YOLO)
            confidence_threshold: Minimum confidence for detection
            use_footandball: Try to use FootAndBall if available
        """
        self.confidence_threshold = confidence_threshold
        self.using_footandball = False
        
        if use_footandball and FOOTANDBALL_AVAILABLE:
            try:
                # Initialize FootAndBall model
                self._init_footandball(model_path)
                self.using_footandball = True
                print("✓ Using FootAndBall model for enhanced detection")
            except Exception as e:
                warnings.warn(f"Failed to load FootAndBall model: {e}. Falling back to YOLO.")
                self._init_yolo_fallback(model_path)
        else:
            if use_footandball:
                print("ℹ FootAndBall model not available. Using YOLO fallback.")
            self._init_yolo_fallback(model_path)
    
    def _init_footandball(self, model_path: Optional[str]):
        """Initialize FootAndBall model"""
        # Placeholder for actual FootAndBall initialization
        # self.model = FootAndBallModel(model_path or 'footandball_default.pth')
        # self.model.eval()
        raise NotImplementedError("FootAndBall model integration not yet implemented")
    
    def _init_yolo_fallback(self, model_path: Optional[str]):
        """Initialize YOLO as fallback"""
        # Use YOLO11x for both pose (players) and object detection (ball)
        self.pose_model = YOLO(model_path or 'yolo11x-pose.pt')
        self.ball_model = YOLO('yolo11x.pt')
        self.ball_class_id = 32  # COCO dataset: sports ball
        print("✓ Using YOLO11x as detection fallback")
    
    def detect_players_and_ball(self, frame: np.ndarray) -> Dict:
        """
        Detect players and ball in single pass
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            Dictionary with 'players' and 'ball' detections
        """
        if self.using_footandball:
            return self._detect_footandball(frame)
        else:
            return self._detect_yolo(frame)
    
    def _detect_footandball(self, frame: np.ndarray) -> Dict:
        """Detect using FootAndBall model"""
        # Placeholder for actual FootAndBall detection
        # results = self.model.detect(frame)
        # return self._parse_footandball_results(results)
        raise NotImplementedError("FootAndBall detection not yet implemented")
    
    def _detect_yolo(self, frame: np.ndarray) -> Dict:
        """Detect using YOLO fallback"""
        result = {
            'players': [],
            'ball': None
        }
        
        # Detect players using pose model
        pose_results = self.pose_model(frame, verbose=False)
        
        if pose_results and len(pose_results) > 0:
            pose_result = pose_results[0]
            
            if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                for i, box in enumerate(pose_result.boxes):
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        player = {
                            'id': i,
                            'bbox': (float(x1), float(y1), float(x2), float(y2)),
                            'center': (float(center_x), float(center_y)),
                            'confidence': confidence,
                            'keypoints': None
                        }
                        
                        # Get keypoints if available
                        if pose_result.keypoints is not None and i < len(pose_result.keypoints):
                            kpts = pose_result.keypoints[i].xy.cpu().numpy()
                            player['keypoints'] = kpts
                        
                        result['players'].append(player)
        
        # Detect ball using object detection model
        ball_results = self.ball_model(frame, verbose=False)
        
        if ball_results and len(ball_results) > 0:
            ball_result = ball_results[0]
            
            if ball_result.boxes is not None and len(ball_result.boxes) > 0:
                ball_detections = []
                
                for box in ball_result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == self.ball_class_id and confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        ball_detections.append({
                            'position': (float(center_x), float(center_y)),
                            'bbox': (float(x1), float(y1), float(x2), float(y2)),
                            'confidence': confidence
                        })
                
                if ball_detections:
                    # Return highest confidence detection
                    result['ball'] = max(ball_detections, key=lambda x: x['confidence'])
        
        return result
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect only ball position
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            Ball center position (x, y) or None
        """
        detections = self.detect_players_and_ball(frame)
        
        if detections['ball'] is not None:
            return detections['ball']['position']
        
        return None
    
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect only players
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            List of player detections
        """
        detections = self.detect_players_and_ball(frame)
        return detections['players']
    
    def get_model_info(self) -> Dict:
        """Get information about the active model"""
        return {
            'using_footandball': self.using_footandball,
            'model_type': 'FootAndBall' if self.using_footandball else 'YOLO11x',
            'confidence_threshold': self.confidence_threshold
        }


# Convenience function for backward compatibility
def create_detector(model_path: Optional[str] = None,
                   confidence_threshold: float = 0.3,
                   prefer_footandball: bool = True) -> FootAndBallDetector:
    """
    Create a FootAndBall detector instance
    
    Args:
        model_path: Path to model weights
        confidence_threshold: Minimum detection confidence
        prefer_footandball: Try to use FootAndBall if available
        
    Returns:
        FootAndBallDetector instance
    """
    return FootAndBallDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        use_footandball=prefer_footandball
    )

