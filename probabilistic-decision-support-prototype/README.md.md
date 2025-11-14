# **A Python-Based Prototype for a Probabilistic Decision Support System in SHM**

This repository contains the source code for a full-stack, end-to-end prototype of a **Probabilistic Decision Support System (PDSS)** for Structural Health Monitoring (SHM).

This system is designed as a **data-agnostic framework** to model the complete data-to-decision pipeline. It starts with raw, physics-based signal simulation and ends with a final, explainable "Action Recommendation" on a live dashboard.

The core of this research prototype is not tied to a specific sensor technology but rather to the **probabilistic methods** (e.g., data fusion, uncertainty quantification, explainable AI) required to handle the noisy, sparse, and uncertain data typical of real-world monitoring. It demonstrates a "full-stack" engineering mindset, modeling not just the ML algorithms but also the **real-world physical constraints** of a remote deployment, including power (battery, solar) and network (packet loss, data caps).

## **🏛️ Core Components**

This prototype is built from several interconnected Python modules that represent the different layers of a real-world deployment:

* **shm\_model\_trainer\_fixed\_v2.py (The "Methods-First" Pipeline):** This script trains the probabilistic classifier. It uses TimeSeriesSplit for rigorous validation and, most critically, analyzes the model's confidence on validation data to generate the **statistically-derived confidence thresholds** that are packaged with the final model.  
* **sensor\_fusion\_logic.py (The "Portfolio-Level" Logic):** This module implements a **Confidence-Weighted Fusion** algorithm. It is a data-agnostic method for aggregating uncertain probabilistic data from multiple sources (e.g., 3 sensors on one bridge, or 100 bridges in a portfolio) to arrive at a single, robust, and defensible decision.  
* **xai\_engine.py (The Explainable AI Module):** This PhysicalExplanationEngine addresses model trust and adoption. It translates abstract mathematical outputs (SHAP values) into human-readable, **physically-grounded explanations** (e.g., "a change in *signal impulsiveness*") for the decision-support dashboard.  
* **shm\_edge\_simulation\_fixed.py (The Main Simulator):** This file integrates all components into a multi-threaded simulation. It includes a **physics-based damage simulation** that models wave propagation and extracts physical features using advanced signal processing libraries (scipy.welch, Pywt).  
* **power\_subsystem.py / network\_simulator.py (The "Full-Stack" Modules):** These files model the **real-world physical and financial constraints** of a remote edge device. This includes modeling a complete energy economy (battery degradation, solar charging, sleep states) and a lossy, bandwidth-constrained network (packet loss, daily data caps).  
* **dashboard.html (The "Decision Support" Front-End):** A live dashboard that serves as the prototype for the final "tool." It visualizes the end product of the entire probabilistic pipeline, including the final **"Fused Health Status"** and **"Action Recommendation."**

## **🚀 How to Run**

This simulation has two parts: training the model (run once) and running the live simulation.

### **1\. Install Dependencies**

It is recommended to use a virtual environment.

pip install numpy pandas scipy Scikit-learn imbalanced-learn pywavelets matplotlib seaborn websockets shap

### **2\. Train the Model**

First, you must generate the deployment package that contains the model and the statistical thresholds.

python shm\_model\_trainer\_fixed\_v2.py

This will run the full training and validation pipeline and create a new file named **shm\_deployment\_package\_enhanced.pkl**.

### **3\. Run the Live Simulation & Dashboard**

Once the .pkl file exists, you can run the main simulation.

python shm\_edge\_simulation\_fixed.py

The simulation will start, initialize all components, and launch a WebSocket server on ws://localhost:8765. You will see status updates in your terminal.

### **4\. View the Dashboard**

Open the **dashboard.html** file in any modern web browser.

The dashboard will automatically connect to the running simulation and display the live, end-to-end system status.