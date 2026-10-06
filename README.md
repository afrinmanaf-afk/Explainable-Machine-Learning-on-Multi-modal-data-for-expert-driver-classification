Explainable Machine Learning on Multi-modal Data for Expert Driver Classification

A robust, transparent machine learning framework designed to analyze multi-modal driving data and accurately classify expert drivers while providing human-understandable explanations for its decisions.

🚗 Overview

Understanding what makes a driver truly "expert" goes beyond simple telemetry. This repository implements a multi-modal machine learning pipeline that fuses diverse data streams—such as CAN-bus telemetry, inertial measurement unit (IMU) sensor readings, and driver physiological or video behavioral cues—to classify driving expertise.

Crucially, safety-critical and intelligent transportation systems require transparency. By integrating State-of-the-Art Explainable AI (XAI) techniques (e.g., SHAP, LIME, Attention Mechanisms), this project uncovers why a model classifies a specific driving style as expert or novice, highlighting critical maneuvers, smooth throttle/brake transitions, and situational awareness.

✨ Key Features

Multi-Modal Data Fusion: Seamlessly integrates time-series telemetry, sensor arrays, and auxiliary features.

Expert Driver Classification: Deep learning and gradient-boosting architectures optimized for behavioral pattern recognition.

Explainable AI (XAI) Integration: Post-hoc interpretability tools to visualize feature importance, temporal impact, and decision pathways.

Modular Pipeline: Cleanly separated data preprocessing, feature extraction, model training, and evaluation scripts.

Reproducible Experiments: Config-driven execution for repeatable research and benchmarking.

📊 Methodology (Brief)

Data Alignment: High-frequency IMU and low-frequency CAN-bus data are synchronized using interpolation and sliding window segmentation.

Representation Learning: Spatial-temporal networks (e.g., CNN-LSTM or Multi-Modal Transformers) capture complex driving maneuvers over time.

Interpretability Layer: SHAP (Shapley Additive exPlanations) values are computed to attribute predictions back to specific sensor inputs (e.g., steering wheel angle variance, smooth braking profiles).
