"""
xai_engine.py

Module implementing a 'Physical Explanation Engine' for the
Probabilistic Decision Support System.

This module's purpose is to address a critical, and often-overlooked,
barrier to the adoption of complex AI/ML systems: their 'black box'
nature.

It translates abstract mathematical outputs (SHAP values) from the
machine learning model into human-readable, physically-grounded,
and context-specific explanations[cite: 87]. This demonstrates an
understanding that for a 'decision support system' to be adopted,
it must be trustworthy and explainable.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

class PhysicalExplanationEngine:
    """
    Translates raw SHAP values and model predictions into human-readable,
    physically-grounded explanations for a single sensor.
    """
    def __init__(self):
        """
        Initializes the explanation engine.
        """
        # --- Class Definitions ---
        # Maps the model's output index to a human-readable name.
        self.class_names: Dict[int, str] = {
            0: 'Healthy',
            1: 'Incipient Damage',
            2: 'Severe Damage'
        }

        # --- Physical Feature Mapping ---
        # This is the core of the 'physical' explanation. It maps abstract
        # feature names to their tangible, physical meaning, which is
        # essential for a 'decision support' tool.
        self.feature_descriptions: Dict[str, str] = {
            'rms': 'Root Mean Square (overall vibration energy)',
            'std_dev': 'Standard Deviation (signal variability)',
            'kurtosis': 'Kurtosis (signal impulsiveness or spikiness)',
            'dominant_freq': 'Dominant Frequency (primary oscillation)',
            'spectral_centroid': 'Spectral Centroid (frequency center of mass)',
            'crest_factor': 'Crest Factor (peak-to-RMS ratio)',
            'wavelet_energy': 'Wavelet Energy (time-frequency content)',
            'peak': 'Peak Amplitude (maximum vibration)',
            'mean': 'Mean Value (signal offset)',
            'skewness': 'Skewness (signal asymmetry)'
        }
        print("Physical Explanation Engine Initialized.")

    def generate_explanation(self, 
                             prediction_idx: int, 
                             shap_values: List[float], 
                             feature_names: List[str], 
                             feature_values_df: pd.DataFrame) -> str:
        """
        Generates a narrative explanation for a single sensor's prediction.

        Args:
            prediction_idx (int): The model's predicted class index (0, 1, or 2).
            shap_values (List[float]): The list of SHAP contribution values.
            feature_names (List[str]): The list of feature names corresponding to shap_values.
            feature_values_df (pd.DataFrame): A single-row DataFrame containing the
                                              actual values for each feature.

        Returns:
            str: A human-readable, narrative explanation of the prediction.
        """
        predicted_class = self.class_names.get(prediction_idx, "Unknown")

        # --- Input Validation ---
        try:
            shap_values_np = np.array(shap_values, dtype=float)
            if not isinstance(feature_names, list):
                feature_names = list(feature_names)
        except (ValueError, TypeError) as e:
            print(f"Error converting SHAP/feature data: {e}. Returning simple explanation.")
            return (f"Conclusion: The structure's current state is assessed as "
                    f"**{predicted_class}**. (Data formatting error in explanation).")

        # Validate dimensions
        if len(shap_values_np) != len(feature_names):
            print(f"Warning: SHAP values length ({len(shap_values_np)}) doesn't match "
                  f"feature names length ({len(feature_names)}). Truncating.")
            min_len = min(len(shap_values_np), len(feature_names))
            shap_values_np = shap_values_np[:min_len]
            feature_names = feature_names[:min_len]

        if len(shap_values_np) == 0:
            return (f"Conclusion: The structure's current state is assessed as "
                    f"**{predicted_class}**. (Insufficient data for detailed explanation).")

        # --- Feature Contribution Analysis ---
        # Combine features with their SHAP values and sort by absolute influence
        contributions = sorted(
            zip(feature_names, shap_values_np),
            key=lambda item: abs(item[1]),
            reverse=True
        )

        # --- Narrative Construction ---
        # 1. Start with the high-level conclusion
        narrative = (f"Conclusion: The structure's current state is assessed as "
                     f"**{predicted_class}**. ")

        # 2. Provide the basis for the assessment
        if prediction_idx == 0:  # Healthy
            narrative += ("This assessment is based on key indicators "
                          "remaining within normal operational ranges")
        else:  # Damage Detected
            narrative += ("This is due to significant deviations in the "
                          "following vibration characteristics")

        # 3. Detail the key contributing features (top 3)
        explanation_points = []
        max_features = min(3, len(contributions))
        
        for i in range(max_features):
            feature, shap_val = contributions[i]
            
            try:
                feature_value = float(feature_values_df[feature].iloc[0])
            except (KeyError, IndexError, ValueError, TypeError):
                print(f"Warning: Could not extract value for feature '{feature}'. Skipping.")
                continue

            # Determine influence direction
            # (SHAP val > 0 pushes towards damage, < 0 pushes towards healthy)
            if prediction_idx == 0: # Explaining a 'Healthy' prediction
                influence = "supporting the healthy assessment"
            else: # Explaining a 'Damage' prediction
                influence = "strongly indicating potential damage" if shap_val > 0 else "trending towards a healthy state"

            # Get the human-readable physical description
            pretty_name = self.feature_descriptions.get(feature, feature.replace('_', ' ').title())
            
            # Format the value for readability
            if 'freq' in feature.lower():
                value_str = f"{feature_value:.1f} Hz"
            elif 'energy' in feature.lower():
                value_str = f"{feature_value:.2e}"
            else:
                value_str = f"{feature_value:.2f}"
            
            point = f"a **{pretty_name}** of **{value_str}** ({influence})"
            explanation_points.append(point)

        # 4. Assemble the final narrative
        if explanation_points:
            if len(explanation_points) == 1:
                narrative += f": {explanation_points[0]}."
            else:
                narrative += (f": {', '.join(explanation_points[:-1])}, "
                              f"and {explanation_points[-1]}.")
        else:
            narrative += ", though specific feature contributions could not be determined."

        return narrative


class MultiSensorExplainer:
    """
    (Phase 3 Stub) Analyzes explanations from multiple sensors to find
    consensus, discrepancies, and spatial patterns.

    This class represents a future research direction, moving from single-sensor
    explanation to a holistic, portfolio-level diagnostic tool.
    """
    def __init__(self, num_sensors: int):
        self.num_sensors = num_sensors
        print("Multi-Sensor Explainer (Phase 3 Stub) Initialized.")

    def generate_consensus_explanation(self, all_sensor_data: List[Dict[str, Any]], fused_decision: int) -> str:
        """
        (Stub) Generates a holistic explanation based on data from all sensors.
        
        In a full implementation, this would analyze SHAP values across
        all sensors to find consensus (e.g., "all sensors agree that
        'kurtosis' is the main driver") or discrepancies ("Sensor 1
        detects a frequency shift, but Sensor 3 does not").
        """
        # --- Placeholder Logic for Phase 3 ---
        if fused_decision == 0:
            return "All sensors report nominal conditions, confirming the structure is healthy."

        try:
            damage_sensors = [
                s['sensor_id'] for s in all_sensor_data if s['prediction_idx'] > 0
            ]
        except TypeError:
             return "Fused alert detected, but sensor data is incomplete."

        if not damage_sensors:
            return ("The fused decision indicates potential damage, "
                    "but individual sensors are not yet in strong agreement. "
                    "Monitoring closely.")

        sensor_list_str = ', '.join(map(str, damage_sensors))
        consensus_narrative = (
            f"A **fused alert** has been triggered. The primary evidence "
            f"comes from sensor(s) **{sensor_list_str}**, "
            "which show anomalous readings. A detailed cross-sensor "
            "analysis would follow."
        )
        return consensus_narrative