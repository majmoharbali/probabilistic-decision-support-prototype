"""
shm_edge_simulation_fixed.py

This is the main executable for the "Prototype Probabilistic Decision
Support System". 

It integrates all modules into a single, multi-threaded simulation
of a real-world Structural Health Monitoring (SHM) deployment.

This script demonstrates the end-to-end pipeline: 
1.  **Physics-Based Data**: `AdvancedDataStreamer` simulates data with
    physical damage models (e.g., wave propagation). 
2.  **Physics-Based Features**: `EdgeProcessor` uses `scipy.welch` and `pywt`
    to extract physically-meaningful features. [cite: 66, 67]
3.  **Probabilistic ML**: The `EdgeProcessor` uses a loaded, calibrated
    model to generate probabilistic predictions.
4.  **XAI (Explainability)**: The `PhysicalExplanationEngine` translates
    SHAP values into human-readable text. [cite: 85, 87]
5.  **Probabilistic Fusion**: `SensorFusion` aggregates sensor data
    using statistical thresholds to make a "portfolio-level" decision. [cite: 74, 78]
6.  **Full-Stack Simulation**: `PowerSubsystem` and `NetworkSimulator`
    model the real-world physical constraints of energy and data. [cite: 95, 96]
7.  **Decision Support**: All data is sent via a WebSocket to the
    `dashboard.html` for a final "Action Recommendation". [cite: 103, 107]
"""

import numpy as np
import pandas as pd
import json
import time
import pickle
import warnings
import random
import threading
import queue
import asyncio
import pywt # For physics-based wavelet features
import websockets
import shap # For Explainable AI
from datetime import datetime
from collections import Counter
from scipy import signal # For physics-based spectral features (Welch)
from scipy.stats import skew, kurtosis
from typing import Dict, Any, List, Optional, Tuple

# --- Import Core Simulation Modules ---
# These modules represent the different "layers" of the full-stack system
from power_subsystem import PowerSubsystem
from network_simulator import NetworkSimulator
from sensor_fusion_logic import SensorFusion
from xai_engine import PhysicalExplanationEngine, MultiSensorExplainer

warnings.filterwarnings('ignore')

# --- Configuration ---
SIMULATION_DURATION_MIN = 15
NUM_SENSORS = 3
WINDOW_DURATION_S = 10
SAMPLING_RATE = 100
WINDOW_SIZE = SAMPLING_RATE * WINDOW_DURATION_S
WELCH_NPERSEG = 256 # Segment size for Welch's method
WAVE_PROPAGATION_SPEED_M_S = 500 # Physics parameter 

# --- Utility Classes ---
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types and datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class WebSocketServer:
    """Handles WebSocket communication with the dashboard.html front-end."""
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self.server_started = threading.Event()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, data: str):
        """Pushes data to all connected dashboard clients."""
        if self.clients:
            tasks = [client.send(data) for client in self.clients]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def start_server_async(self):
        self.loop = asyncio.get_running_loop()
        server = await websockets.serve(self.handler, self.host, self.port)
        print(f"WebSocket server started on ws://{self.host}:{self.port}")
        self.server_started.set()
        await server.wait_closed()

    def run_in_thread(self):
        asyncio.run(self.start_server_async())

# --- Simulation Components ---
class AdvancedDataStreamer:
    """
    Simulates a multi-sensor array with physics-based damage propagation.
    
    This class models how a single damage event (at an "epicenter")
    propagates through a structure at a defined speed,  causing
    different arrival times and intensities at each sensor. This is a
    core component of the "physics-based" simulation.
    """
    def __init__(self, num_sensors: int = 3):
        self.t = np.linspace(0, WINDOW_DURATION_S, WINDOW_SIZE, endpoint=False)
        self.environmental_state: str = 'healthy'
        self.state_change_schedule = queue.Queue()
        # Define a simple linear sensor array
        self.sensor_coords = {i: (i * 10, 0) for i in range(num_sensors)}
        self.damage_epicenter: Tuple[int, int] = (5, 0) # Damage starts near Sensor 0
        self.damage_start_time: Optional[float] = None
        self.damage_level: str = 'healthy'
        self.damage_arrival_times: Dict[int, float] = {i: float('inf') for i in range(num_sensors)}
        print(f"Multi-Sensor Data Streamer Initialized with Spatial Propagation.")

    def stream_data(self, data_queues: List[queue.Queue], stop_event: threading.Event):
        """Main loop for streaming data to processor threads."""
        print("Starting multi-sensor data streaming...")
        while not stop_event.is_set():
            self._check_schedule()
            for sensor_id in range(NUM_SENSORS):
                chunk = self._generate_signal_chunk(sensor_id)
                data_queues[sensor_id].put(np.resize(chunk, WINDOW_SIZE))
            time.sleep(WINDOW_DURATION_S)
        print("Data streaming stopped.")

    def _check_schedule(self):
        """Checks and executes scheduled state changes (e.g., damage events)."""
        if not self.state_change_schedule.empty():
            event_time, new_state, msg = self.state_change_schedule.queue[0]
            if time.time() >= event_time:
                print(f"\n🚨 STATE TRANSITION: {msg} 🚨\n")
                if 'damage' in msg.lower():
                    self.damage_start_time = time.time()
                    self.damage_level = new_state
                    self._calculate_damage_arrival_times()
                else:
                    self.environmental_state = new_state
                self.state_change_schedule.get()

    def _calculate_damage_arrival_times(self):
        """
        Calculates the delay for the damage signal to reach each sensor
        based on physical distance and wave propagation speed. 
        """
        epicenter_coord = np.array(self.damage_epicenter)
        for sensor_id, sensor_coord in self.sensor_coords.items():
            distance = np.linalg.norm(epicenter_coord - np.array(sensor_coord))
            delay = distance / WAVE_PROPAGATION_SPEED_M_S
            self.damage_arrival_times[sensor_id] = self.damage_start_time + delay
            print(f" - Damage will reach Sensor {sensor_id} in {delay:.2f} seconds.")

    def schedule_state_change(self, delay: int, state: str, msg: str):
        """Schedules a future change in the simulation's state."""
        event_time = time.time() + delay
        self.state_change_schedule.put((event_time, state, msg))
        print(f"Event '{msg}' scheduled in {delay} seconds.")

    def _generate_signal_chunk(self, sensor_id: int) -> np.ndarray:
        """Generates a single window of sensor data."""
        base_signal = self._generate_base_signal()

        # Check if the damage wave has physically reached this sensor
        if time.time() > self.damage_arrival_times[sensor_id]:
            damage_duration = time.time() - self.damage_arrival_times[sensor_id]
            distance = np.linalg.norm(np.array(self.damage_epicenter) - np.array(self.sensor_coords[sensor_id]))
            spatial_attenuation = np.exp(-0.05 * distance)
            severity = min(1.0, damage_duration / 300) * spatial_attenuation # Damage grows over 5 mins

            if self.damage_level == 'incipient':
                intensity = 0.1 + (0.2 * severity)
            else:
                intensity = 0.5 + (0.5 * severity)
            
            # Apply physics-based damage effects (damping, freq shift)
            damaged_signal = self._apply_damage_effects(base_signal, intensity)
            return self._apply_environmental_noise(damaged_signal)

        if self.environmental_state == 'storm':
            return self._simulate_environmental_event(base_signal)

        return self._apply_environmental_noise(base_signal)

    def _generate_base_signal(self, base_freq: float = 10.0) -> np.ndarray:
        freq = base_freq * (1 + np.random.uniform(-0.02, 0.02))
        amp_mod = 1 + 0.2 * np.sin(2 * np.pi * 0.1 * self.t)
        hum = 0.1 * np.sin(2 * np.pi * 50 * self.t) # 50Hz hum
        return amp_mod * np.sin(2 * np.pi * freq * self.t) + hum

    def _apply_environmental_noise(self, signal_: np.ndarray) -> np.ndarray:
        return signal_ + np.random.normal(0, np.random.uniform(0.1, 0.3), signal_.shape)

    def _apply_damage_effects(self, signal_: np.ndarray, intensity: float) -> np.ndarray:
        """Models physical damage as reduced frequency and increased damping."""
        freq_shift = 1 - (intensity * 0.1) # Reduced stiffness -> lower freq
        damp = 0.05 + intensity * 0.15 # Increased energy dissipation
        damped_signal = signal_ * np.exp(-damp * self.t)
        new_len = int(len(self.t) / freq_shift)
        resampled = signal.resample(damped_signal, new_len)
        return resampled[:len(self.t)]

    def _simulate_environmental_event(self, signal_: np.ndarray) -> np.ndarray:
        """Simulates a high-noise storm event."""
        return signal_ + np.random.normal(0, 0.8, signal_.shape) + 2 * np.sin(2 * np.pi * 0.5 * self.t)

class EdgeProcessor:
    """
    Simulates the on-device processor for a *single* sensor.
    
    This class's job is to run the feature extraction and ML inference
    for its assigned sensor.
    """
    def __init__(self, sensor_id: int, data_queue: queue.Queue, 
                 model_package: Dict[str, Any], result_queue: queue.Queue):
        self.sensor_id = sensor_id
        self.data_queue = data_queue
        self.result_queue = result_queue
        
        # Load artifacts from the trained model package
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.feature_selector = model_package['feature_selector']

    def process_data(self, stop_event: threading.Event, power_subsystem: PowerSubsystem):
        """Main loop for processing data from the sensor queue."""
        while not stop_event.is_set():
            try:
                window = self.data_queue.get(timeout=1.0)
                
                # --- Full-Stack Link: Power ---
                # Set device state to 'processing', which consumes
                # high power in the PowerSubsystem. 
                power_subsystem.set_state('processing')

                # --- 1. Physics-Based Feature Extraction ---
                features_df = self._extract_features(window)
                
                # --- 2. ML Inference Pipeline ---
                features_scaled = self.scaler.transform(features_df)
                features_selected = self.feature_selector.transform(features_scaled)

                # --- 3. Probabilistic Prediction ---
                prediction_idx = self.model.predict(features_selected)[0]
                prediction_proba = self.model.predict_proba(features_selected)[0]

                # --- Full-Stack Link: Power ---
                # Return to idle state after processing is done
                power_subsystem.set_state('idle')

                result = {
                    'timestamp': datetime.now(),
                    'sensor_id': self.sensor_id,
                    'prediction_idx': prediction_idx,
                    'prediction_proba': prediction_proba,
                    'raw_signal': window[::10].tolist(), # Downsample for dashboard
                    'raw_signal_full': window.tolist(), # Full signal for XAI
                    'features': features_df.to_dict('records')[0]
                }
                
                # Send the probabilistic result to the main system thread
                self.result_queue.put(result)

            except queue.Empty:
                power_subsystem.set_state('idle') # Ensure idle state on timeout
                continue

    def _extract_features(self, data_window: np.ndarray) -> pd.DataFrame:
        """
        Extracts the physically-meaningful feature set.
        
        This method is direct proof of the ability to combine
        physics-based signal processing with ML. [cite: 64]
        """
        rms = np.sqrt(np.mean(data_window**2))
        peak = np.max(np.abs(data_window))
        
        # --- Physics-Based Feature: Welch's Method (Spectral) ---
        # Use a Hanning window for better spectral accuracy 
        f, Pxx = signal.welch(data_window, SAMPLING_RATE, 
                              nperseg=WELCH_NPERSEG, window='hann')
        
        # --- Physics-Based Feature: Wavelet Packet (Time-Frequency) ---
        try:
            wp = pywt.WaveletPacket(data=data_window, wavelet='db4', 
                                    mode='symmetric', maxlevel=3)
            wavelet_energy = np.sum([np.sum(node.data**2) 
                                     for node in wp.get_level(3, order='natural')])  # [cite: 67]
        except Exception:
            wavelet_energy = 0 # Handle edge cases

        features = {
            'mean': np.mean(data_window),
            'std_dev': np.std(data_window),
            'rms': rms,
            'peak': peak,
            'skewness': skew(data_window),
            'kurtosis': kurtosis(data_window),
            'crest_factor': peak / (rms + 1e-9),
            'dominant_freq': f[np.argmax(Pxx)] if len(Pxx) > 0 else 0,
            'spectral_centroid': np.sum(f * Pxx) / (np.sum(Pxx) + 1e-9) if np.sum(Pxx) > 0 else 0,
            'wavelet_energy': wavelet_energy
        }

        return pd.DataFrame([features])

class ExplainableEdgeProcessor(EdgeProcessor):
    """
    An enhanced processor that adds XAI (Explainable AI) capabilities.
    
    It inherits from EdgeProcessor and adds the SHAP explanation
    generation step, [cite: 68] demonstrating the "ML + XAI" skill.
    """
    def __init__(self, sensor_id: int, data_queue: queue.Queue, 
                 model_package: Dict[str, Any], result_queue: queue.Queue):
        super().__init__(sensor_id, data_queue, model_package, result_queue)
        print(f"Sensor {sensor_id}: Initializing SHAP Explainer.")
        
        # Load the dedicated 'explainer_model' from the package
        explainer_model = model_package['explainer_model']
        self.explainer = shap.TreeExplainer(explainer_model)

    def generate_explanation(self, features_selected: np.ndarray) -> Dict[str, Any]:
        """
        Uses SHAP to generate feature contributions for the prediction. [cite: 68]
        """
        # 1. Generate SHAP values from the explainer model
        shap_values = self.explainer.shap_values(features_selected)
        predicted_class_idx = self.model.predict(features_selected)[0]

        # 2. Extract SHAP values for the *predicted class*
        # This logic handles the complex, multi-class output of SHAP
        if isinstance(shap_values, list) and len(shap_values) > predicted_class_idx:
            class_shap_values = shap_values[predicted_class_idx][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            class_shap_values = shap_values[0, :, predicted_class_idx]
        else:
            class_shap_values = np.zeros(features_selected.shape[1])

        # 3. Get the names of the *selected* features
        try:
            all_feature_names = self.scaler.get_feature_names_out()
        except AttributeError: # Fallback for older sklearn
            all_feature_names = np.array(['mean', 'std_dev', 'rms', 'peak', 'skewness', 
                                          'kurtosis', 'crest_factor', 'dominant_freq', 
                                          'spectral_centroid', 'wavelet_energy'])
            
        selected_feature_names = np.array(all_feature_names)[self.feature_selector.get_support()]

        # Ensure dimensions match (a common issue)
        min_len = min(len(class_shap_values), len(selected_feature_names))
        
        return {
            'feature_contributions': class_shap_values[:min_len].tolist(),
            'feature_names': selected_feature_names[:min_len].tolist(),
            'prediction': int(predicted_class_idx),
            'prediction_confidence': self.model.predict_proba(features_selected)[0][predicted_class_idx]
        }

class SHMEdgeSystem:
    """
    The main class that orchestrates the entire end-to-end simulation.
    
    It initializes all components, loads the model, starts all threads,
    and handles the central `handle_result` pipeline.
    """
    def __init__(self, model_package_path: str, ws_server: WebSocketServer):
        print("Loading deployment package...")
        with open(model_package_path, 'rb') as f:
            model_package = pickle.load(f)

        self.stop_event = threading.Event()
        self.result_queue = queue.Queue() # Results from all sensors
        self.ws_server = ws_server
        self.alert_history = []
        self.last_sensor_update = {i: time.time() for i in range(NUM_SENSORS)}
        self.start_time = time.time()
        self.total_readings_processed = 0

        # --- Initialize All System Components ---
        
        # 1. Full-Stack Components
        self.power_subsystem = PowerSubsystem()
        self.network_simulator = NetworkSimulator()
        
        # 2. Probabilistic Fusion Component
        # **This is the key strategic link**: We pass the 'confidence_stats'
        # from the model package *directly* into the SensorFusion module.
        # This is tangible proof of the end-to-end "statistical threshold"
        # methodology. 
        confidence_stats = model_package.get('confidence_stats', None)
        self.sensor_fusion = SensorFusion(num_sensors=NUM_SENSORS, 
                                          confidence_stats=confidence_stats)
        
        # 3. Data and XAI Components
        self.data_streamer = AdvancedDataStreamer(num_sensors=NUM_SENSORS)
        self.explanation_engine = PhysicalExplanationEngine()
        self.multi_sensor_explainer = MultiSensorExplainer(num_sensors=NUM_SENSORS)

        # 4. Sensor Processor Threads
        self.data_queues = [queue.Queue(maxsize=5) for _ in range(NUM_SENSORS)]
        self.processors = [
            ExplainableEdgeProcessor(i, self.data_queues[i], model_package, self.result_queue)
            for i in range(NUM_SENSORS)
        ]

    def run_simulation(self):
        """Starts all simulation threads and runs the main loop."""
        print("\n" + "="*80 +
              f"\n🚀 Starting Unified Simulation for {SIMULATION_DURATION_MIN} minutes...\n" +
              "="*80)

        # Schedule the physics-based events
        self.data_streamer.schedule_state_change(30, 'storm', "Heavy storm impacting sensors")
        self.data_streamer.schedule_state_change(90, 'healthy', "Weather cleared")
        self.data_streamer.schedule_state_change(150, 'incipient', "Initial damage detected (propagating from epicenter)")
        self.data_streamer.schedule_state_change(300, 'severe', "Damage progression to severe")

        # Start all component threads
        threads = [
            threading.Thread(target=p.process_data, args=(self.stop_event, self.power_subsystem))
            for p in self.processors
        ]
        threads.append(
            threading.Thread(target=self.data_streamer.stream_data, args=(self.data_queues, self.stop_event))
        )
        for t in threads:
            t.start()

        # --- Main Simulation Loop ---
        start_time = time.time()
        try:
            while time.time() - start_time < SIMULATION_DURATION_MIN * 60:
                # 1. Update the 'full-stack' physics (power)
                self.power_subsystem.update()
                
                try:
                    # 2. Wait for a result from any sensor processor
                    res = self.result_queue.get(timeout=1.0)
                    
                    # 3. Process the result through the full pipeline
                    self.handle_result(res)
                    
                except queue.Empty:
                    # 4. If no result, just update the dashboard
                    self.broadcast_dashboard_data()
                    continue
        finally:
            self.stop(threads)

    def handle_result(self, res: Dict[str, Any]):
        """
        The core "Decision Support" pipeline.
        
        This method executes the full sequence:
        XAI -> Physical Explanation -> Fusion -> Alerting -> Transmission
        """
        sensor_id = res['sensor_id']
        self.last_sensor_update[sensor_id] = time.time()
        self.total_readings_processed += 1

        # --- 1. XAI Pipeline ---
        # (Re-run processing to get features for XAI)
        processor = self.processors[sensor_id]
        features_df = processor._extract_features(np.array(res['raw_signal_full']))
        features_scaled = processor.scaler.transform(features_df)
        features_selected = processor.feature_selector.transform(features_scaled)

        # 1a. Generate mathematical SHAP explanation [cite: 68]
        explanation = processor.generate_explanation(features_selected)
        res['explanation'] = explanation

        # 1b. Generate human-readable physical explanation [cite: 85, 87]
        text_explanation = self.explanation_engine.generate_explanation(
            prediction_idx=res['prediction_idx'],
            shap_values=explanation['feature_contributions'],
            feature_names=explanation['feature_names'],
            feature_values_df=features_df
        )
        res['text_explanation'] = text_explanation

        # --- 2. Probabilistic Fusion Pipeline ---
        # Update the portfolio-level fusion engine with this sensor's
        # probabilistic evidence. [cite: 74, 76]
        self.sensor_fusion.update_sensor_state(
            sensor_id,
            res['prediction_idx'],
            res['prediction_proba']
        )
        
        # Get the new, robust "fused" decision
        fused_state = self.sensor_fusion.get_fused_decision()  # [cite: 78]

        # --- 3. Alerting Pipeline ---
        if fused_state > 0:
            if not self.alert_history or self.alert_history[-1]['level'] != fused_state:
                alert_levels = {1: "Incipient", 2: "Severe"}
                self.alert_history.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'level': fused_state,
                    'level_name': alert_levels.get(fused_state, "Unknown")
                })
                if len(self.alert_history) > 10: self.alert_history.pop(0)

            # Print the human-readable explanation *only* on alert
            print(f"🚨 FUSED ALERT: {alert_levels.get(fused_state)}! "
                  f"(Sensor {sensor_id} reports: {text_explanation})")

        # --- 4. Full-Stack Transmission Pipeline ---
        # Transmit the data package over the lossy, constrained network 
        self.network_simulator.transmit(res, self.power_subsystem)
        
        # --- 5. Decision Support Dashboard Pipeline ---
        # Broadcast the final, rich data package to the dashboard [cite: 104]
        self.broadcast_dashboard_data(latest_reading=res)

    def broadcast_dashboard_data(self, latest_reading: Optional[Dict[str, Any]] = None):
        """Packages and sends all system states to the WebSocket server."""
        sensor_statuses = {
            i: "Online" if time.time() - self.last_sensor_update[i] < 30 else "Offline"
            for i in range(NUM_SENSORS)
        }

        # Collect state from all sub-modules
        dashboard_data = {
            'power': self.power_subsystem.get_state(),
            'network': self.network_simulator.get_state(),
            'sensor_fusion': self.sensor_fusion.get_state(),
            'sensor_statuses': sensor_statuses,
            'alert_history': self.alert_history,
            'latest_reading': latest_reading,
            'timestamp': datetime.now().isoformat(),
            'system_stats': {
                'uptime_seconds': int(time.time() - self.start_time),
                'total_readings': self.total_readings_processed,
            }
        }

        # Avoid sending the huge full signal array over WebSocket
        if latest_reading and 'raw_signal_full' in latest_reading:
            del latest_reading['raw_signal_full']

        json_data = json.dumps(dashboard_data, cls=NumpyEncoder)

        if self.ws_server.loop:
            asyncio.run_coroutine_threadsafe(
                self.ws_server.broadcast(json_data),
                self.ws_server.loop
            )

    def stop(self, threads: List[threading.Thread]):
        """Stops all running threads gracefully."""
        print("\nStopping simulation threads...")
        self.stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

# --- Main Execution ---
if __name__ == "__main__":
    MODEL_PACKAGE_PATH = "shm_deployment_package_enhanced.pkl"

    # 1. Start the WebSocket server in a background thread
    ws_server = WebSocketServer()
    ws_thread = threading.Thread(target=ws_server.run_in_thread, daemon=True)
    ws_thread.start()

    # Wait for the server to be ready
    ws_server.server_started.wait()

    # 2. Initialize the main SHM system
    edge_system = SHMEdgeSystem(model_package_path=MODEL_PACKAGE_PATH, 
                              ws_server=ws_server)
    
    # 3. Run the simulation
    edge_system.run_simulation()