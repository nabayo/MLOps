"""
Face blur preprocessing using deface library.

Simple and effective face anonymization for privacy protection.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .base import PreprocessingStep


class FaceBlurStep(PreprocessingStep):
    """Face blur preprocessing using deface library."""
    
    def __init__(self, threshold: float = 0.2):
        """
        Initialize face blur step.
        
        Args:
            threshold: Detection threshold for face detection (0-1)
        """
        self.threshold = threshold
        
        # Import deface here to avoid import at module level
        try:
            import deface
            self.deface = deface
        except ImportError:
            raise ImportError(
                "deface library not installed. "
                "Install with: pip install deface"
            )
    
    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Apply face blurring to image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Image with blurred faces
        """
        # Convert BGR to RGB for deface
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save to temporary file (deface works with file paths)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_in:
            temp_in_path = temp_in.name
            cv2.imwrite(temp_in_path, image)
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_out:
                temp_out_path = temp_out.name
            
            # Run deface
            from deface.deface import main as deface_main
            import sys
            
            # Suppress deface output
            old_argv = sys.argv
            sys.argv = [
                'deface',
                '--thresh', str(self.threshold),
                '--output', temp_out_path,
                temp_in_path
            ]
            
            try:
                deface_main()
            except SystemExit:
                pass  # deface calls sys.exit()
            finally:
                sys.argv = old_argv
            
            # Read result
            if os.path.exists(temp_out_path):
                result = cv2.imread(temp_out_path)
            else:
                # If deface failed, return original image
                result = image.copy()
            
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_in_path):
                os.remove(temp_in_path)
            if os.path.exists(temp_out_path):
                os.remove(temp_out_path)
        
        return result
    
    def get_name(self) -> str:
        """Get the name of this preprocessing step."""
        return "FaceBlur"


class FastFaceBlurStep(PreprocessingStep):
    """
    Fast face blur using OpenCV Haar Cascades.
    
    Faster alternative to deface for real-time processing.
    """
    
    def __init__(self, blur_kernel_size: int = 99):
        """
        Initialize fast face blur step.
        
        Args:
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
        """
        self.blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Apply fast face blurring using Haar Cascades.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Image with blurred faces
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Blur each face
        result = image.copy()
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = result[y:y+h, x:x+w]
            
            # Apply Gaussian blur
            blurred_face = cv2.GaussianBlur(
                face_roi,
                (self.blur_kernel_size, self.blur_kernel_size),
                0
            )
            
            # Replace face region with blurred version
            result[y:y+h, x:x+w] = blurred_face
        
        return result
    
    def get_name(self) -> str:
        """Get the name of this preprocessing step."""
        return "FastFaceBlur"
