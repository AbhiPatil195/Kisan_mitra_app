"""
KisanMitra Quality Assessment Module
This module handles the image processing and quality assessment of agricultural products.
"""

import tensorflow as tf
import cv2
import numpy as np

class CropQualityAnalyzer:
    def __init__(self):
        self.model = tf.keras.models.load_model('quality_model.h5')
        
    def analyze_image(self, image_path):
        """
        Analyzes the quality of a crop from its image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Quality assessment results
        """
        image = cv2.imread(image_path)
        processed = self.preprocess_image(image)
        quality_score = self.model.predict(processed)
        return self.generate_quality_report(quality_score)
        
    def preprocess_image(self, image):
        """
        Preprocesses the image for quality assessment.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Processed image
        """
        # Resize image
        resized = cv2.resize(image, (224, 224))
        # Normalize
        normalized = resized / 255.0
        # Expand dimensions
        processed = np.expand_dims(normalized, axis=0)
        return processed
        
    def generate_quality_report(self, score):
        """
        Generates a detailed quality report.
        
        Args:
            score (float): Quality score from the model
            
        Returns:
            dict: Detailed quality report
        """
        return {
            'quality_grade': self.get_grade(score),
            'confidence': float(score),
            'recommendations': self.get_recommendations(score)
        }
        
    def get_grade(self, score):
        """
        Converts numerical score to grade.
        
        Args:
            score (float): Quality score
            
        Returns:
            str: Quality grade (A+, A, B, C, D)
        """
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        else:
            return 'D'
            
    def get_recommendations(self, score):
        """
        Provides recommendations based on quality score.
        
        Args:
            score (float): Quality score
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        if score >= 0.8:
            recommendations.append("Premium quality - suitable for export market")
        elif score >= 0.7:
            recommendations.append("Good quality - ideal for domestic market")
        else:
            recommendations.append("Needs improvement - consider processing market")
        return recommendations 