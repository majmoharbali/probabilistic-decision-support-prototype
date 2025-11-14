"""
sensor_fusion_logic.py

Module implementing a "Confidence-Weighted Fusion" algorithm.

This file is a tangible prototype of the logic required to solve a
portfolio-level risk management problem[cite: 73, 74]. Its purpose is to
aggregate uncertain, probabilistic data from multiple, heterogeneous
sources (e.g., different sensors on one bridge, or the fused risk data
from multiple bridges in a portfolio) into a single, robust, and
defensible decision[cite: 75, 81].

This updated (V3) version enhances the core logic by replacing hardcoded
"magic numbers" with statistically-derived thresholds loaded from the
model training pipeline[cite: 179]. This demonstrates a rigorous,
data-driven approach to uncertainty management, aligning with the
TUM ERA group's focus on "stochastic methods" and "reliability analysis"[cite: 13, 15].
"""

from collections import deque
import numpy as np
from typing import Dict, Any, Optional, List

# Define class indices for clarity
CLASS_HEALTHY = 0
CLASS_INCIPIENT = 1
CLASS_SEVERE = 2

class SensorFusion:
    """
    Implements a robust, confidence-weighted fusion logic.
    
    This class is data-agnostic. It can be scaled from
    fusing 3 sensors on one bridge to fusing the uncertain risk data
    from 100 bridges in a portfolio[cite: 119].
    """

    def __init__(self, num_sensors: int = 3, confidence_stats: Optional[Dict[str, Any]] = None):
        """
        Initializes the fusion engine.

        Args:
            num_sensors (int): The number of data sources to fuse.
            confidence_stats (Optional[Dict]): A dictionary from the model
                training package containing statistically-derived thresholds
                and analysis data[cite: 179].
        """
        self.num_sensors = num_sensors
        
        # State stores both the last predicted class and the full probability vector
        # for each sensor[cite: 76].
        self.sensor_states: Dict[int, Dict[str, Any]] = {
            i: {'class': CLASS_HEALTHY, 'confidence': np.array([1.0, 0.0, 0.0])}
            for i in range(num_sensors)
        }
        
        # --- Strategic Improvement (V3) ---
        # Initialize decision thresholds using statistical data from training,
        # moving beyond arbitrary values to a defensible, probabilistic method.
        self._initialize_thresholds(confidence_stats)
        
        print(f"Sensor Fusion logic initialized for {num_sensors} sensors.")
        print(f"Thresholds (Method: {self.threshold_method}) - "
              f"Incipient: {self.incipient_threshold:.3f}, Severe: {self.severe_threshold:.3f}")

    def _initialize_thresholds(self, confidence_stats: Optional[Dict[str, Any]]):
        """
        Initializes confidence thresholds using statistical data from training.
        
        This method is key to a robust "probabilistic decision support system"[cite: 33].
        It sets decision boundaries based on the model's actual performance
        during validation, rather than guesswork.
        """
        if confidence_stats and 'incipient_threshold' in confidence_stats:
            # Use statistically-derived thresholds from the model package
            self.incipient_threshold: float = confidence_stats['incipient_threshold']
            self.severe_threshold: float = confidence_stats['severe_threshold']
            
            # Store metadata for diagnostics and dashboard reporting
            self.threshold_method: str = confidence_stats.get('calculation_method', 'statistical')
            self.training_sample_counts: Dict[str, int] = {
                'healthy': len(confidence_stats.get('healthy_confidences', [])),
                'incipient': len(confidence_stats.get('incipient_confidences', [])),
                'severe': len(confidence_stats.get('severe_confidences', []))
            }
            
            print("Successfully loaded statistical thresholds from model training.")
            print(f"  Training samples (Healthy/Incipient/Severe): "
                  f"{self.training_sample_counts['healthy']}/"
                  f"{self.training_sample_counts['incipient']}/"
                  f"{self.training_sample_counts['severe']}")
                  
        else:
            # Fallback to conservative defaults if no statistical data is available
            self.incipient_threshold = 0.55
            self.severe_threshold = 0.65
            self.threshold_method = 'conservative_default'
            self.training_sample_counts = {'healthy': 0, 'incipient': 0, 'severe': 0}
            
            print("WARNING: No statistical threshold data found. "
                  "Using conservative defaults.")

    def update_sensor_state(self, sensor_id: int, new_state_idx: int, confidence_array: List[float]):
        """Updates a single sensor's state with its latest prediction and confidence vector."""
        if sensor_id not in self.sensor_states:
            print(f"Warning: Unknown sensor ID {sensor_id}. Ignoring update.")
            return
            
        self.sensor_states[sensor_id] = {
            'class': new_state_idx,
            'confidence': np.array(confidence_array)
        }

    def get_fused_decision(self) -> int:
        """
        Calculates a final, fused decision for the entire system (or portfolio).
        
        This algorithm performs a confidence-weighted aggregation of all
        available evidence [cite: 75, 77] and then applies the robust,
        statistically-derived thresholds to manage uncertainty and make a
        final classification.
        """
        # 1. Aggregate Evidence: Sum the probability vectors from all active sensors.
        total_confidence = np.zeros(3)  # [Healthy, Incipient, Severe]
        active_sensors = 0
        
        for sensor_id in self.sensor_states:
            if self.sensor_states[sensor_id]['confidence'] is not None:
                total_confidence += self.sensor_states[sensor_id]['confidence']
                active_sensors += 1

        if active_sensors == 0:
            return CLASS_HEALTHY  # Default to healthy if no sensors are active

        # 2. Normalize: Create a single, fused probability distribution[cite: 77].
        fused_probabilities = total_confidence / np.sum(total_confidence)

        # 3. Classify: The fused decision is the class with the highest aggregated confidence
        fused_decision_idx = np.argmax(fused_probabilities)
        max_confidence = fused_probabilities[fused_decision_idx]

        # 4. Apply Statistical Thresholds:
        # This is the core of the robust decision logic. A "severe" or
        # "incipient" classification is only made if the aggregated
        # confidence exceeds the pre-defined statistical threshold,
        # reducing the chance of false positives from noisy data.
        
        # Rule: To declare severe damage, confidence must exceed the severe threshold
        if fused_decision_idx == CLASS_SEVERE and max_confidence > self.severe_threshold:
            return CLASS_SEVERE  # Confirmed Severe

        # Rule: To declare incipient damage, confidence must exceed the incipient threshold
        if fused_decision_idx == CLASS_INCIPIENT and max_confidence > self.incipient_threshold:
            return CLASS_INCIPIENT  # Confirmed Incipient

        # Default to healthy if confidence doesn't meet damage thresholds
        return CLASS_HEALTHY

    def get_confidence_margins(self) -> Dict[str, float]:
        """
        Calculates how close the current fused confidence is to triggering an alert.
        
        This is a valuable metric for a "decision support system"[cite: 33], as it
        quantifies the system's stability (i.e., "how close are we to an alert?").
        """
        total_confidence = np.zeros(3)
        active_sensors = 0
        
        for sensor_id in self.sensor_states:
            if self.sensor_states[sensor_id]['confidence'] is not None:
                total_confidence += self.sensor_states[sensor_id]['confidence']
                active_sensors += 1

        if active_sensors == 0:
            return {'incipient_margin': 1.0, 'severe_margin': 1.0, 
                    'incipient_confidence': 0.0, 'severe_confidence': 0.0}

        fused_probabilities = total_confidence / np.sum(total_confidence)
        
        incipient_confidence = fused_probabilities[CLASS_INCIPIENT]
        severe_confidence = fused_probabilities[CLASS_SEVERE]
        
        # Margin = distance from threshold (positive = safe, negative = triggered)
        incipient_margin = self.incipient_threshold - incipient_confidence
        severe_margin = self.severe_threshold - severe_confidence
        
        return {
            'incipient_margin': float(incipient_margin),
            'severe_margin': float(severe_margin),
            'incipient_confidence': float(incipient_confidence),
            'severe_confidence': float(severe_confidence)
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a full, JSON-serializable snapshot of the fusion system's state
        for the dashboard and network transmission.
        """
        # Calculate aggregated confidence for dashboard visualization
        total_confidence = np.zeros(3)
        active_sensors = 0
        
        for sensor_id, state in self.sensor_states.items():
            if state['confidence'] is not None:
                total_confidence += np.array(state['confidence'])
                active_sensors += 1

        # Normalize to get average confidence distribution
        if active_sensors > 0:
            normalized_confidence = total_confidence / np.sum(total_confidence)
        else:
            normalized_confidence = np.array([1.0, 0.0, 0.0])  # Default to healthy

        # Create serializable version of individual sensor states
        serializable_states = {
            k: {
                'class': int(v['class']),
                'confidence': v['confidence'].tolist() if v['confidence'] is not None else [1.0, 0.0, 0.0]
            }
            for k, v in self.sensor_states.items()
        }

        # Get confidence margins for monitoring
        margins = self.get_confidence_margins()
        fused_decision = self.get_fused_decision()

        return {
            'sensor_states': serializable_states,
            'fused_decision': int(fused_decision),
            'aggregated_confidence': normalized_confidence.tolist(),
            'active_sensors': active_sensors,
            'fusion_method': 'confidence_weighted_statistical',
            
            # Include threshold information in state for monitoring and transparency
            'thresholds': {
                'incipient_threshold': self.incipient_threshold,
                'severe_threshold': self.severe_threshold,
                'method': self.threshold_method
            },
            'confidence_margins': margins,
            'training_sample_counts': self.training_sample_counts
        }

    def update_thresholds(self, new_incipient_threshold: float, new_severe_threshold: float):
        """
        (Advanced) Allows dynamic threshold updates during operation.
        
        This could be used for an adaptive system or for manual tuning
        based on field experience, demonstrating a flexible framework.
        """
        if new_incipient_threshold is not None:
            old_threshold = self.incipient_threshold
            # Clamp to reasonable range
            self.incipient_threshold = max(0.1, min(0.9, new_incipient_threshold)) 
            print(f"Updated incipient threshold: {old_threshold:.3f} -> {self.incipient_threshold:.3f}")
            
        if new_severe_threshold is not None:
            old_threshold = self.severe_threshold
            # Clamp to reasonable range
            self.severe_threshold = max(0.1, min(0.9, new_severe_threshold))
            print(f"Updated severe threshold: {old_threshold:.3f} -> {self.severe_threshold:.3f}")
            
        # Ensure hierarchy is maintained (severe must be >= incipient)
        if self.severe_threshold < self.incipient_threshold:
            self.severe_threshold = self.incipient_threshold + 0.05
            print(f"Adjusted severe threshold to maintain hierarchy: {self.severe_threshold:.3f}")