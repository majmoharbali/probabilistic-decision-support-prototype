"""
network_simulator.py

Module for simulating a constrained, lossy network connection for a remote
SHM edge device.

This file is a key component of the 'full-stack' simulation, demonstrating
an understanding of practical data transmission constraints. It models:
- Stochastic packet loss, a common issue in wireless/remote deployments.
- Data transmission retries to overcome packet loss.
- A daily data cap, modeling the cost constraints of cellular/satellite IoT plans.
- Power consumption linkage (via the PowerSubsystem) for data transmission.
"""

import numpy as np
import time
import random
import json
from datetime import datetime
from typing import Dict, Any, Union
# Import the PowerSubsystem class for type hinting
from power_subsystem import PowerSubsystem 

class NetworkSimulator:
    """
    Simulates transmitting data over a lossy, bandwidth-constrained network.

    This class models the real-world challenges of data exfiltration from
    a remote sensor node, linking data transmission attempts directly to
    power consumption and success/failure statistics.
    """
    def __init__(self, packet_loss_probability: float = 0.15, max_retries: int = 2):
        """
        Initializes the network simulator.

        Args:
            packet_loss_probability (float): The probability (0.0 to 1.0) that a packet is lost.
            max_retries (int): The number of additional attempts to send a packet after
                               the first one fails.
        """
        self.packet_loss_probability: float = packet_loss_probability
        self.max_retries: int = max_retries
        
        # --- Data Cap Simulation ---
        # This models the financial/cost constraints of a real-world deployment
        # (e.g., a cellular or satellite data plan).
        self.daily_bytes_sent: int = 0
        self.daily_cap: int = 100_000_000  # 100 MB daily cap

        # --- Transmission Statistics ---
        # These stats are crucial for a decision-support dashboard to
        # monitor the health of the data acquisition system itself.
        self.successful_transmissions: int = 0
        self.failed_transmissions: int = 0
        self.total_retries: int = 0
        
        print(f"Network Simulator Initialized. Packet Loss: {self.packet_loss_probability*100}%.")

    def transmit(self, payload: Dict[str, Any], power_subsystem: PowerSubsystem) -> bool:
        """
        Simulates transmitting a data payload over a lossy network with retries.

        This method models the core transmission logic:
        1. Checks against the daily data cap.
        2. Engages the 'transmitting' state on the power subsystem.
        3. Simulates a probabilistic packet loss.
        4. Implements a retry mechanism.
        5. Updates transmission statistics.

        Args:
            payload (Dict): The data (as a dictionary) to be sent.
            power_subsystem (PowerSubsystem): The device's power subsystem object
                                              to track energy consumption.

        Returns:
            bool: True if transmission was successful, False otherwise.
        """
        # 1. Check against data cap (cost constraint)
        if self.daily_bytes_sent >= self.daily_cap:
            print("⚠️ Daily data cap exceeded - transmission blocked")
            return False

        # Estimate payload size by serializing to JSON
        try:
            payload_str = json.dumps(payload, default=str)
            payload_size = len(payload_str.encode('utf-8'))
        except TypeError:
            payload_size = 256  # Assign a default size if serialization fails

        # 2. Loop for initial attempt + max_retries
        for attempt in range(self.max_retries + 1):
            # 3. Engage power subsystem: transmission is energy-intensive
            power_subsystem.set_state('transmitting')
            # Simulate the time taken to transmit
            time.sleep(0.5) 

            # 4. Simulate probabilistic packet loss
            if random.random() > self.packet_loss_probability:
                # --- Success ---
                self.successful_transmissions += 1
                self.daily_bytes_sent += payload_size
                print(f"✅ (Attempt {attempt + 1}) Transmission successful. Sent {payload_size} bytes.")
                power_subsystem.set_state('idle') # Return to low-power state
                return True
            else:
                # --- Failure ---
                print(f"❌ (Attempt {attempt + 1}) Packet loss detected. Retrying...")
                self.total_retries += 1
                # Short delay before retrying
                time.sleep(1)

        # 5. If all retries fail
        print("⛔ All transmission retries failed.")
        self.failed_transmissions += 1
        power_subsystem.set_state('idle') # Return to low-power state
        return False

    def reset_daily_stats(self):
        """Resets the daily byte counter (e.g., called once per 24h)."""
        self.daily_bytes_sent = 0

    def get_state(self) -> Dict[str, Union[int, float]]:
        """
        Returns a thread-safe snapshot of the network's state for the dashboard.

        This packages the internal state into a JSON-serializable dictionary.
        """
        success_rate = self.packet_loss_probability # Assumes a simple model
        if (self.successful_transmissions + self.failed_transmissions) > 0:
            # Calculate actual measured success rate
            success_rate = self.successful_transmissions / (self.successful_transmissions + self.failed_transmissions)
            
        return {
            'daily_bytes_sent': int(self.daily_bytes_sent),
            'daily_cap': int(self.daily_cap),
            'packet_loss_probability': float(self.packet_loss_probability),
            
            # --- Enhanced Dashboard Fields ---
            # These are key performance indicators (KPIs) for the system's health
            'transmission_success_rate': float(success_rate),
            'bandwidth_utilization': float(self.daily_bytes_sent / self.daily_cap) if self.daily_cap > 0 else 0.0,
            'retries_attempted': int(self.total_retries),
            'successful_transmissions': int(self.successful_transmissions),
            'failed_transmissions': int(self.failed_transmissions)
        }