"""
power_subsystem.py

Module for simulating the power subsystem of a remote, solar-powered
Structural Health Monitoring (SHM) edge device.

This file is a key component of the 'full-stack' simulation, demonstrating
an understanding of the real-world physical and hardware constraints that
govern a practical, long-term deployment. It models:
- Battery capacity and voltage.
- Power consumption based on the device's operational state (e.g., 'idle', 'processing').
- Solar charging via a diurnal (day/night) cycle model.
- Stochastic weather events (cloud cover) impacting charging efficiency.
- Long-term battery health degradation based on charge cycles.
"""

import time
import numpy as np
from datetime import datetime
import threading
from typing import Dict, Any, Literal

# Define operational states for clarity
DeviceState = Literal['idle', 'sensing', 'processing', 'transmitting']

class PowerSubsystem:
    """
    Models the power subsystem of the edge device.

    This class simulates battery degradation, state-based power consumption,
    and solar charging influenced by environmental factors. It is designed
    to be thread-safe for use in a multi-threaded simulation environment.
    """

    def __init__(self, battery_capacity_mah: int = 2000, voltage: float = 3.7):
        """
        Initializes the power subsystem.

        Args:
            battery_capacity_mah (int): The nominal initial capacity of the battery in milliamp-hours.
            voltage (float): The operational voltage of the battery.
        """
        # --- Core Battery Parameters ---
        self.initial_capacity_wh: float = (battery_capacity_mah / 1000) * voltage
        self.max_capacity_wh: float = self.initial_capacity_wh  # Current max capacity (degrades over time)
        self.current_charge_wh: float = self.max_capacity_wh
        self.voltage: float = voltage
        
        # --- State and Time Parameters ---
        self.state: DeviceState = 'idle'
        self.last_update_time: float = time.time()
        
        # --- Environmental Simulation Parameters ---
        self.is_cloudy: bool = False
        self.last_weather_change: float = time.time()
        self.current_solar_rate_w: float = 0.0  # Instantaneous charging rate
        
        # --- Degradation and Stats Parameters ---
        # Tracks cumulative energy discharged to model long-term battery aging
        self.cumulative_discharge_wh: float = 0.0
        
        # --- Threading Locks ---
        # Ensures thread-safe access to power and weather variables
        self.power_lock = threading.Lock()
        self.weather_lock = threading.Lock()

        print(f"Power Subsystem Initialized. Initial Capacity: {self.initial_capacity_wh:.2f} Wh")

    def set_state(self, new_state: DeviceState):
        """
        Thread-safe method to change the device's operational state.
        
        This simulates the device controller changing power modes.
        """
        with self.power_lock:
            self.state = new_state

    def update(self):
        """
        Updates the battery's charge based on elapsed time, power draw, and solar charging.
        
        This method is intended to be called repeatedly in the main simulation loop.
        """
        elapsed_seconds = time.time() - self.last_update_time
        # Avoid redundant calculations if called too frequently
        if elapsed_seconds < 1:
            return

        # 1. --- Calculate Power Drain based on Current State ---
        # These values model a realistic power profile for an edge AI device,
        # where 'processing' (ML inference) is the most power-intensive task.
        power_draw_mw = {
            'idle': 25,          # Low-power sleep mode
            'sensing': 100,      # Activating sensor and ADC
            'processing': 1800,  # Running ML inference on a microcontroller/SoC
            'transmitting': 450  # LoRaWAN or NB-IoT transmission burst
        }[self.state]
        
        energy_drained_j = (power_draw_mw / 1000) * elapsed_seconds
        energy_drained_wh = energy_drained_j / 3600
        
        with self.power_lock:
            self.current_charge_wh -= energy_drained_wh
            self.cumulative_discharge_wh += energy_drained_wh

        # 2. --- Apply Solar Charging ---
        energy_gained_wh = self._get_solar_charge(elapsed_seconds)
        with self.power_lock:
            self.current_charge_wh += energy_gained_wh

        # 3. --- Simulate Battery Aging ---
        # Models long-term degradation, a critical factor for practical deployments.
        self._simulate_aging()

        # 4. --- Clamp Charge to Valid Range ---
        with self.power_lock:
            self.current_charge_wh = np.clip(self.current_charge_wh, 0, self.max_capacity_wh)
        
        self.last_update_time = time.time()

    def _get_solar_charge(self, elapsed_seconds: float) -> float:
        """
        Calculates solar energy gained based on time of day and weather.
        
        This simulates two key physical constraints:
        1. A diurnal (day/night) cycle using a sinusoidal model.
        2. Stochastic weather events (cloud cover) that reduce panel efficiency.
        """
        
        # Simulate a random change in weather (e.g., clouds rolling in/out)
        with self.weather_lock:
            # Check if 10 minutes (600s) have passed and a low-probability event occurs
            if (time.time() - self.last_weather_change > 600 and np.random.rand() < 0.001):
                self.is_cloudy = not self.is_cloudy
                self.last_weather_change = time.time()
        
        # Model the diurnal (day/night) solar intensity cycle
        current_time = datetime.now()
        seconds_into_day = (current_time - current_time.replace(
            hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        
        # Use a sine wave to model solar intensity (0 at midnight/sunrise/sunset, 1 at noon)
        solar_intensity = max(0, np.sin(2 * np.pi * seconds_into_day / 86400 - np.pi/2) + 0.5)
        
        # Apply efficiency based on simulated weather
        efficiency = 0.3 if self.is_cloudy else 0.9
        max_charge_power_w = 2.0  # Max power of the solar panel in Watts
        
        self.current_solar_rate_w = max_charge_power_w * solar_intensity * efficiency
        
        # Return energy gained in Watt-hours
        return (self.current_solar_rate_w * elapsed_seconds) / 3600

    def _simulate_aging(self):
        """
        Models long-term battery degradation based on cumulative charge cycles.
        
        This simulates cycle-based aging (e.g., 20% capacity loss after 500
        full charge/discharge cycles), which is a critical factor for
        assessing the long-term viability of a "practical solution"[cite: 97].
        """
        full_cycles = self.cumulative_discharge_wh / self.initial_capacity_wh
        
        # Simple linear degradation model: 20% loss after 500 cycles
        degradation_factor = 1.0 - (0.2 * min(full_cycles / 500.0, 1.0))
        
        with self.power_lock:
            self.max_capacity_wh = self.initial_capacity_wh * degradation_factor

    def get_battery_percentage(self) -> float:
        """Calculates the current battery percentage relative to its current max capacity."""
        if self.max_capacity_wh <= 0:
            return 0.0
        return (self.current_charge_wh / self.max_capacity_wh) * 100

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a thread-safe snapshot of the power subsystem's state.
        
        This method packages the complex internal state into a simple dictionary,
        making it easy to serialize to JSON for the dashboard[cite: 104].
        """
        with self.power_lock:
            # Create a snapshot of critical values
            battery_percentage = self.get_battery_percentage()
            current_charge = self.current_charge_wh
            max_capacity = self.max_capacity_wh
            current_state = self.state
            solar_rate = self.current_solar_rate_w
            
        # --- Human-Readable Status ---
        # Translates raw data into qualitative assessments for a
        # "decision support system".
        if battery_percentage > 75:
            battery_status = "Excellent"
        elif battery_percentage > 50:
            battery_status = "Good"
        elif battery_percentage > 25:
            battery_status = "Fair"
        else:
            battery_status = "Critical"

        # --- Return State Dictionary ---
        return {
            'battery_percentage': battery_percentage,
            'current_charge_wh': current_charge,
            'max_capacity_wh': max_capacity,  # Current degraded capacity
            'voltage': self.voltage,
            'solar_rate_w': solar_rate,
            'weather': "Cloudy ☁️" if self.is_cloudy else "Clear ☀️",
            'state': current_state,
            
            # --- Enhanced Dashboard Fields ---
            'battery_health': battery_status,
            'estimated_runtime_hours': self._estimate_runtime(current_charge, current_state),
            'charging_status': 'Charging' if solar_rate > 0.1 else 'Discharging',
            'power_efficiency': self._calculate_efficiency(),
            'cumulative_discharge_wh': self.cumulative_discharge_wh
        }
    
    def _estimate_runtime(self, current_charge_wh: float, state: DeviceState) -> float:
        """Estimates remaining runtime in hours based on current consumption."""
        # Power draw in Watts
        current_draw_w = {
            'idle': 0.025,
            'sensing': 0.1,
            'processing': 1.8,
            'transmitting': 0.45
        }.get(state, 0.1)  # Default to 'sensing' draw if state is unknown
        
        if current_draw_w <= 0:
            return float('inf')
        
        return current_charge_wh / current_draw_w
    
    def _calculate_efficiency(self) -> float:
        """
        Calculates a simplified overall power efficiency percentage.
        
        Models the difference between initial theoretical capacity and
        cumulative energy discharged, representing aging.
        """
        if self.cumulative_discharge_wh == 0:
            return 100.0
        
        # A simple model assuming a 10-cycle lifetime for 100% efficiency loss
        theoretical_max = self.initial_capacity_wh * 10
        efficiency = max(0.0, 100.0 - (self.cumulative_discharge_wh / theoretical_max * 100.0))
        return min(100.0, efficiency)