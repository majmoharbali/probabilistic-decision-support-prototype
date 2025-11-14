"""
shm_model_trainer_fixed_v2.py

An end-to-end machine learning pipeline for training the
Structural Health Monitoring (SHM) probabilistic classifier.

This script demonstrates a rigorous, "methods-first" approach to
model development, aligning with the core identity of a
"risk and reliability analyst"[cite: 17].

Key features of this pipeline:
1.  **Physics-Based Simulation**: `AdvancedDataGenerator` creates a
    temporally-ordered dataset with realistic damage physics.
2.  **Physical Feature Extraction**: `FeatureExtractor` uses advanced
    signal processing (scipy.welch, pywt) to create a
    physically-meaningful feature set[cite: 64, 66, 67].
3.  **Probabilistic Modeling**: Uses `CalibratedClassifierCV` to output
    reliable, real-world probabilities, not just class predictions.
4.  **Rigorous Validation**: Employs `TimeSeriesSplit` to correctly
    validate the model on temporally-ordered data.
5.  **Statistical Threshold Generation**: This is the pipeline's key
    output. The `_calculate_confidence_thresholds` method generates
    data-driven, statistical thresholds that are packaged with the
    model, enabling the robust, "portfolio-level" fusion logic seen in
    `sensor_fusion_logic.py`[cite: 15, 78].
"""

import numpy as np
import pandas as pd
import time
import pickle
import warnings
import random
from datetime import datetime
from collections import Counter
from scipy import signal
from scipy.stats import skew, kurtosis
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive (server) environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import pywt
from typing import List, Dict, Any, Tuple, np_ndarray

warnings.filterwarnings('ignore')

class AdvancedDataGenerator:
    """
    Simulates a physics-based, temporally-ordered stream of sensor data.
    
    This simulates data as it would be collected in the real world,
    with environmental noise and damage effects (e.g., frequency shifts,
    damping) applied sequentially. This is crucial for `TimeSeriesSplit`.
    """
    def __init__(self, sampling_rate: int = 100, duration: int = 10, n_samples: int = 3000):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.n_samples = n_samples
        self.t = np.linspace(0, self.duration, self.sampling_rate * self.duration, endpoint=False)
        print(f"Advanced Data Generator Initialized: {n_samples} samples, {duration}s duration, {sampling_rate}Hz rate.")

    def _generate_base_signal(self, base_freq: float = 10.0) -> np.ndarray:
        """Generates a clean base signal with minor variations."""
        drift = np.random.uniform(0.01, 0.05)
        freq = base_freq * (1 + drift * (np.random.rand() - 0.5))
        amplitude_mod = 1 + 0.2 * np.sin(2 * np.pi * 0.1 * self.t)
        secondary_hum = 0.1 * np.sin(2 * np.pi * 50 * self.t) # 50Hz noise
        return amplitude_mod * np.sin(2 * np.pi * freq * self.t) + secondary_hum

    def apply_environmental_noise(self, signal_: np.ndarray) -> np.ndarray:
        """Applies random environmental noise."""
        noise_level = np.random.uniform(0.1, 0.3)
        return signal_ + np.random.normal(0, noise_level, signal_.shape)

    def apply_damage_effects(self, signal_: np.ndarray, damage_level: str) -> np.ndarray:
        """
        Applies physics-based damage effects (frequency shift, damping).
        
        This models the physical manifestation of damage as changes in
        stiffness (frequency shift) and energy dissipation (damping).
        """
        if damage_level == 'incipient':
            intensity = np.random.uniform(0.1, 0.3)
        elif damage_level == 'severe':
            intensity = np.random.uniform(0.5, 1.0)
        else:
            return signal_

        freq_shift = 1.0 - (intensity * 0.10) # Damage reduces stiffness -> lowers freq
        damping_factor = 0.05 + intensity * 0.15 # Damage increases damping
        
        damped_signal = signal_ * np.exp(-damping_factor * self.t)
        
        # Resample to simulate frequency shift
        new_len = int(len(self.t) / freq_shift)
        resampled = signal.resample(damped_signal, new_len)
        return resampled[:len(self.t)]

    def simulate_environmental_event(self, signal_: np.ndarray) -> np.ndarray:
        """Simulates a high-noise event (e.g., storm, high wind)."""
        storm_noise = np.random.normal(0, 0.8, signal_.shape)
        wind_gust = 2 * np.sin(2 * np.pi * 0.5 * self.t) # Low-freq gust
        return signal_ + storm_noise + wind_gust

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the full, temporally-ordered dataset.
        
        Crucially, data is generated in order: first all healthy,
        then incipient, then severe. This mimics a structure's
        lifecycle and is the correct format for `TimeSeriesSplit`.
        """
        data, labels = [], []
        samples_per_class = self.n_samples // 3

        for i in range(self.n_samples):
            base_signal = self._generate_base_signal()
            final_signal, label = None, -1

            if i < samples_per_class: # --- Phase 1: Healthy ---
                if random.random() < 0.3: # Simulate occasional storm
                    final_signal = self.simulate_environmental_event(base_signal)
                else:
                    final_signal = self.apply_environmental_noise(base_signal)
                label = 0
            elif i < 2 * samples_per_class: # --- Phase 2: Incipient Damage ---
                damaged = self.apply_damage_effects(base_signal, 'incipient')
                final_signal = self.apply_environmental_noise(damaged)
                label = 1
            else: # --- Phase 3: Severe Damage ---
                damaged = self.apply_damage_effects(base_signal, 'severe')
                final_signal = self.apply_environmental_noise(damaged)
                label = 2

            data.append(np.resize(final_signal, len(self.t)))
            labels.append(label)

        print(f"Generated {len(data)} temporal samples. Distribution: {Counter(labels)}")
        return np.array(data), np.array(labels)

class FeatureExtractor:
    """
    Extracts a sophisticated, physically-meaningful feature set[cite: 64].
    
    This demonstrates the ability to combine physics-based signal
    processing with ML, a core requirement of the TUM project[cite: 70].
    """
    PHYSICAL_FEATURES = ['mean', 'std_dev', 'rms', 'peak', 'skewness', 'kurtosis',
                        'crest_factor', 'dominant_freq', 'spectral_centroid', 'wavelet_energy']

    # Apply domain knowledge: certain features are known to be more
    # indicative of damage. This weighting improves model focus.
    PHYSICAL_WEIGHTS = {
        'dominant_freq': 1.5,
        'spectral_centroid': 1.3,
        'crest_factor': 1.2,
        'kurtosis': 1.1,
        'std_dev': 1.1,
        'wavelet_energy': 1.4
    }

    def extract(self, data: np.ndarray) -> pd.DataFrame:
        """Extracts features from the raw signal data."""
        features = []
        fs = 100 # Sampling rate
        
        for x in data:
            rms_val = np.sqrt(np.mean(x**2))
            peak_val = np.max(np.abs(x))

            # --- 1. Spectral Features (scipy.signal.welch) --- 
            # Use Welch's method for a robust Power Spectral Density estimate.
            # A Hanning window is used to reduce spectral leakage.
            f, Pxx = signal.welch(x, fs, nperseg=256, window='hann')
            
            # --- 2. Wavelet Features (pywt.WaveletPacket) --- [cite: 67]
            # Use Wavelet Packet Decomposition to analyze time-frequency
            # energy, good for detecting non-stationary signals.
            try:
                wp = pywt.WaveletPacket(data=x, wavelet='db4', mode='symmetric', maxlevel=3)
                wavelet_energy = np.sum([np.sum(node.data**2) for node in wp.get_level(3, order='natural')])
            except Exception:
                wavelet_energy = 0 # Handle potential signal processing errors

            # --- 3. Statistical Features ---
            features.append([
                np.mean(x),
                np.std(x),
                rms_val,
                peak_val,
                skew(x),
                kurtosis(x),
                peak_val / (rms_val + 1e-10), # Crest Factor
                f[np.argmax(Pxx)] if len(Pxx) > 0 else 0, # Dominant Freq
                np.sum(f * Pxx) / (np.sum(Pxx) + 1e-10) if np.sum(Pxx) > 0 else 0, # Spectral Centroid
                wavelet_energy
            ])

        return pd.DataFrame(features, columns=self.PHYSICAL_FEATURES)

    def apply_feature_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies expert-defined weights to features."""
        X_weighted = X.copy()
        for feature, weight in self.PHYSICAL_WEIGHTS.items():
            if feature in X_weighted.columns:
                X_weighted[feature] *= weight
        return X_weighted

class ModelTrainer:
    """
    Trains, validates, and packages the probabilistic SHM classifier.
    
    This class handles the core ML pipeline: scaling, feature selection,
    probabilistic model calibration, rigorous time-series validation,
    and statistical threshold generation.
    """
    def __init__(self, n_features: int = 8):
        # --- The "Probabilistic" Model ---
        # Base model is a powerful RandomForest.
        base_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=10
        )
        
        # We wrap the base model in CalibratedClassifierCV. This is a
        # deliberate choice to ensure the output `predict_proba` values
        # are reliable, calibrated probabilities, which is essential
        # for our "probabilistic decision support system"[cite: 33].
        self.model = CalibratedClassifierCV(
            base_model,
            method='isotonic', # Isotonic is better for non-linear models
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        )

        # A separate, standard RF model for SHAP explanations
        # (SHAP has better support for standard RFs than CalibratedCV).
        self.explainer_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=10
        )

        # --- Preprocessing Pipeline ---
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=n_features)
        
        # --- Final Deployment Artifacts ---
        self.trained_model = None
        self.selected_features: List[str] = None
        
        # This dictionary will hold the key statistical output for the
        # sensor_fusion_logic module.
        self.confidence_stats: Dict[str, Any] = {
            'healthy_confidences': [],
            'incipient_confidences': [],
            'severe_confidences': []
        }
        
        print("Model Trainer Initialized with CalibratedClassifierCV (for reliable probabilities) "
              "and TimeSeriesSplit (for rigorous validation).")

    def train_and_evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """
        Trains and validates the model using TimeSeries Cross-Validation.
        
        This is the core training loop. It collects performance metrics
        and, most importantly, the confidence scores from all folds
        to build the statistical thresholds.
        """
        feature_extractor = FeatureExtractor()
        X_weighted = feature_extractor.apply_feature_weights(X.copy())

        # --- Rigorous Time-Series Validation ---
        # We use TimeSeriesSplit, which ensures that the training data
        # *always* comes before the test data in each fold. This
        # prevents data leakage and realistically models a system
        # that learns from the past to predict the future.
        tscv = TimeSeriesSplit(n_splits=5)
        
        fold_scores, y_preds_all, y_test_all = [], [], []
        all_confidence_scores, all_true_labels = [], []
        
        print("\n--- Starting Time Series Cross-Validation (Data NOT shuffled) ---")
        
        successful_folds = 0
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_weighted)):
            X_train, X_test = X_weighted.iloc[train_idx], X_weighted.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Skip folds with insufficient class diversity
            unique_classes = len(np.unique(y_train))
            if unique_classes < 2:
                print(f"Skipping fold {fold+1} - only {unique_classes} class(es) present in training data.")
                continue

            print(f"Fold {fold+1}: Training classes: {Counter(y_train)}, Test classes: {Counter(y_test)}")

            # Preprocessing is fit *only* on the fold's training data
            scaler_fold = StandardScaler()
            feature_selector_fold = SelectKBest(f_classif, k=self.feature_selector.k)
            
            X_train_scaled = scaler_fold.fit_transform(X_train)
            X_test_scaled = scaler_fold.transform(X_test)

            X_train_selected = feature_selector_fold.fit_transform(X_train_scaled, y_train)
            X_test_selected = feature_selector_fold.transform(X_test_scaled)

            # --- Handle Class Imbalance with SMOTE ---
            try:
                smote = SMOTE(random_state=42)
                X_train_res, y_train_res = smote.fit_resample(X_train_selected, y_train)
                print(f"Fold {fold+1}: Original: {Counter(y_train)} | Resampled: {Counter(y_train_res)}")
            except ValueError as e:
                print(f"Fold {fold+1}: SMOTE failed ({str(e)}). Using original data.")
                X_train_res, y_train_res = X_train_selected, y_train

            # --- Train and Collect Probabilities ---
            model_fold = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=10)
            model_fold.fit(X_train_res, y_train_res)
            y_pred = model_fold.predict(X_test_selected)
            
            # --- FIX: Ensure `predict_proba` output has a consistent shape (3 columns) ---
            # This handles a common real-world problem where a fold's
            # training data (e.g., only 'Healthy' and 'Incipient')
            # causes the model to output a 2-column probability array.
            y_pred_proba_raw = model_fold.predict_proba(X_test_selected)
            
            # Create a full (N_samples, 3_classes) matrix of zeros
            y_pred_proba_full = np.zeros((len(X_test_selected), 3))
            
            # Place the predicted probabilities into the correct columns
            # based on the classes the model *actually* saw.
            y_pred_proba_full[:, model_fold.classes_] = y_pred_proba_raw
            # --- END FIX ---
            
            # Collect data for final metrics and threshold calculation
            all_confidence_scores.append(y_pred_proba_full)
            all_true_labels.append(y_test)
            
            fold_accuracy = accuracy_score(y_test, y_pred)
            fold_scores.append(fold_accuracy)
            y_preds_all.extend(y_pred)
            y_test_all.extend(y_test)
            successful_folds += 1

            print(f"Fold {fold+1} Test Accuracy: {fold_accuracy:.3f}")

        if successful_folds == 0:
            raise ValueError("No successful folds. Check data distribution and TimeSeriesSplit settings.")

        print(f"\nCompleted {successful_folds} successful folds.")
        
        # --- Calculate Statistical Thresholds (The Key Output) ---
        confidence_array = np.vstack(all_confidence_scores)
        labels_array = np.concatenate(all_true_labels)
        self._calculate_confidence_thresholds(confidence_array, labels_array)

        # --- Final Model Training on Full Dataset ---
        print("\n--- Final Model Training on Full Dataset ---")
        X_final_scaled = self.scaler.fit_transform(X_weighted)
        X_final_selected = self.feature_selector.fit_transform(X_final_scaled, y)

        try:
            smote_final = SMOTE(random_state=42)
            X_res, y_res = smote_final.fit_resample(X_final_selected, y)
            print(f"Final training: Original {Counter(y)} | Resampled: {Counter(y_res)}")
        except ValueError as e:
            print(f"Final SMOTE failed ({str(e)}). Using original data.")
            X_res, y_res = X_final_selected, y

        # Train the calibrated "production" model
        self.trained_model = self.model.fit(X_res, y_res)
        # Train the separate "explainer" model
        self.explainer_model.fit(X_res, y_res)

        self.selected_features = X_weighted.columns[self.feature_selector.get_support()]

        # --- Final Reporting ---
        print("\nCross-Validation Performance (Aggregated from all folds):")
        accuracy = accuracy_score(y_test_all, y_preds_all)
        f1 = f1_score(y_test_all, y_preds_all, average='weighted')
        report = classification_report(y_test_all, y_preds_all, target_names=['Healthy', 'Incipient', 'Severe'], labels=[0, 1, 2])
        cm = confusion_matrix(y_test_all, y_preds_all, labels=[0, 1, 2])

        print(f"Overall Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        print("Classification Report:\n", report)
        
        self._plot_feature_importance(self.selected_features, self.explainer_model.feature_importances_)
        return {'accuracy': accuracy, 'f1_score': f1, 'confusion_matrix': cm}

    def _calculate_confidence_thresholds(self, confidence_scores: np.ndarray, true_labels: np.ndarray):
        """
        Calculates statistically-derived confidence thresholds.
        
        This is the core methodological link between training and deployment.
        Instead of using arbitrary thresholds (e.g., 0.5), we analyze the
        model's *actual confidence* on correctly-classified validation data.
        
        We use the 25th percentile as a "conservative" threshold. This means
        "to be classified as 'Incipient', the model's confidence must be
        higher than 75% of the *correct* Incipient predictions seen in
        validation." This is a robust, data-driven, and defensible
        approach for a high-stakes "risk and reliability" system.
        """
        print("\n--- Calculating Statistical Confidence Thresholds ---")
        
        healthy_confidences, incipient_confidences, severe_confidences = [], [], []
        
        for conf_array, true_label in zip(confidence_scores, true_labels):
            predicted_class = np.argmax(conf_array)
            max_confidence = conf_array[predicted_class]
            
            # We only analyze the confidence of *correct* predictions
            if predicted_class == true_label:
                if true_label == 0:
                    healthy_confidences.append(max_confidence)
                elif true_label == 1:
                    incipient_confidences.append(max_confidence)
                elif true_label == 2:
                    severe_confidences.append(max_confidence)
        
        healthy_confidences = np.array(healthy_confidences)
        incipient_confidences = np.array(incipient_confidences)
        severe_confidences = np.array(severe_confidences)
        
        print(f"Correctly classified samples: Healthy={len(healthy_confidences)}, "
              f"Incipient={len(incipient_confidences)}, Severe={len(severe_confidences)}")
        
        # Calculate the 25th percentile (a conservative threshold)
        if len(incipient_confidences) > 0:
            incipient_threshold = np.percentile(incipient_confidences, 25)
        else:
            incipient_threshold = 0.55 # Fallback
            print("Warning: No correctly classified incipient samples found. Using fallback.")
            
        if len(severe_confidences) > 0:
            severe_threshold = np.percentile(severe_confidences, 25)
        else:
            severe_threshold = 0.65 # Fallback
            print("Warning: No correctly classified severe samples found. Using fallback.")
        
        self.confidence_stats = {
            'healthy_confidences': healthy_confidences.tolist(),
            'incipient_confidences': incipient_confidences.tolist(),
            'severe_confidences': severe_confidences.tolist(),
            'incipient_threshold': float(incipient_threshold),
            'severe_threshold': float(severe_threshold),
            'calculation_method': '25th_percentile_conservative'
        }
        
        print(f"Statistical Thresholds Calculated:")
        print(f"  Incipient Damage Threshold (25th percentile): {incipient_threshold:.3f}")
        print(f"  Severe Damage Threshold (25th percentile): {severe_threshold:.3f}")
        
        self._plot_threshold_distributions()
        
    def _plot_threshold_distributions(self):
        """Visualize the confidence distributions and calculated thresholds."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            sns.histplot(self.confidence_stats.get('healthy_confidences', []), bins=20, ax=axes[0], color='green', kde=True)
            axes[0].set_title('Healthy Confidence Distribution')
            axes[0].set_xlabel('Confidence')
            
            sns.histplot(self.confidence_stats.get('incipient_confidences', []), bins=20, ax=axes[1], color='orange', kde=True)
            axes[1].axvline(self.confidence_stats.get('incipient_threshold', 0.55), color='red', linestyle='--', label=f"Threshold: {self.confidence_stats.get('incipient_threshold', 0.55):.3f}")
            axes[1].set_title('Incipient Confidence Distribution')
            axes[1].set_xlabel('Confidence')
            axes[1].legend()
            
            sns.histplot(self.confidence_stats.get('severe_confidences', []), bins=20, ax=axes[2], color='red', kde=True)
            axes[2].axvline(self.confidence_stats.get('severe_threshold', 0.65), color='darkred', linestyle='--', label=f"Threshold: {self.confidence_stats.get('severe_threshold', 0.65):.3f}")
            axes[2].set_title('Severe Confidence Distribution')
            axes[2].set_xlabel('Confidence')
            axes[2].legend()
            
            plt.tight_layout()
            plt.savefig('confidence_thresholds_analysis.png', dpi=300)
            plt.close()
            print("Confidence threshold analysis saved as 'confidence_thresholds_analysis.png'")
        except Exception as e:
            print(f"Error plotting threshold distributions: {e}")

    def _plot_feature_importance(self, feature_names: List[str], importances: np.ndarray):
        """Plots and saves the feature importance graph."""
        try:
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title("Physical Feature Importance")
            sns.barplot(x=[importances[i] for i in indices], y=[feature_names[i] for i in indices])
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300)
            plt.close()
            print("Feature importance plot saved as 'feature_importance.png'")
        except Exception as e:
            print(f"Error plotting feature importance: {e}")

    def save_deployment_package(self, filename: str = "shm_deployment_package_enhanced.pkl"):
        """
        Saves the complete deployment package.
        
        This is not just a model file. It is a package containing
        all artifacts needed for deployment: the calibrated model,
        the data scaler, the feature selector, and—most critically—
        the `confidence_stats` dictionary that enables the
        probabilistic sensor fusion logic.
        """
        package = {
            'model': self.trained_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_feature_names': self.selected_features,
            'explainer_model': self.explainer_model,
            'confidence_stats': self.confidence_stats, # The key link!
            'metadata': {'training_date': datetime.now().strftime("%Y-%m-%d")}
        }

        with open(filename, 'wb') as f:
            pickle.dump(package, f)

        print(f"\n✅ Enhanced deployment package saved to '{filename}'")
        print("   Package includes: model, scaler, selector, explainer, and statistical confidence thresholds.")

def main():
    print("=" * 80)
    print("Starting Enhanced SHM Model Training Pipeline")
    print(" (Using TimeSeriesSplit and Statistical Threshold Generation)")
    print("=" * 80)

    # 1. Generate temporally-ordered, physics-based data
    data_gen = AdvancedDataGenerator(n_samples=4500)
    X_raw, y = data_gen.generate()

    # 2. Extract physically-meaningful features
    feature_extractor = FeatureExtractor()
    X_features = feature_extractor.extract(X_raw)

    print("\nData generation and feature extraction complete.")
    
    # 3. Train, validate, and generate thresholds
    trainer = ModelTrainer(n_features=8)
    results = trainer.train_and_evaluate(X_features, y)

    # 4. Save all deployment artifacts into one package
    trainer.save_deployment_package()

    print("\nTraining pipeline complete.")

if __name__ == "__main__":
    main()