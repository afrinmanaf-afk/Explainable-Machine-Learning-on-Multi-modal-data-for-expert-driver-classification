# **Multi-Modal EEG Driving Expertise Dataset**

Understanding Neural and Cognitive Mechanisms of Expert Driving in Naturalistic Urban Environments

# **Executive Summary & Abstract**

Modern autonomous driving algorithms operating in complex urban environments strive for high levels of safety, passenger comfort, and decision-making intelligence—qualities modeled after expert human drivers. Despite expert drivers serving as ideal targets for algorithmic imitation learning, a critical gap remains in understanding the underlying neural and cognitive mechanisms driving their tactical choices.

This project presents a multi-modal dataset comparing 10 expert drivers (`E01`–`E10`) and 10 novice drivers (`N01`–`N10`) across 13 naturalistic urban driving scenarios and baseline conditions. The dataset synchronizes electroencephalography (EEG) brain activity with vehicle CAN bus telemetry, surrounding traffic conditions, eye-tracking dynamics, and psychophysiological measures including Electrodermal Activity (EDA) and Blood Volume Pulse (BVP). Physiological and subjective feedback was also gathered from two passengers per trial to assess riding quality, supplemented by standardized pre/post questionnaires (NASA-TLX, Subjective Stress, DES, SAM) and semi-structured post-drive interviews.

# **Dataset Architecture & Multi-Modal Synchronization**

The dataset incorporates diverse sensor streams sampled at varying frequencies, unified through time-synchronization pipelines into standardized Parquet records.┌─────────────────────────────────────────────────────────────────────────┐

│                        MULTI-MODAL SENSOR STREAMS                        │

├──────────────┬───────────────┬───────────────┬──────────────┬───────────┤

│  EEG Signals │ CAN-Bus Data  │  Eye-Tracking │ Biometrics   │ Traffic   │

│  (100 Hz)    │ (Speed/Accel) │  (1920x1080)  │ (EDA & BVP)  │ Density   │

└──────┬───────┴───────┬───────┴───────┬───────┴──────┬───────┴─────┬─────┘

       │               │               │              │             │

       └───────────────┴───────┬───────┴──────────────┴─────────────┘

                               │

                               ▼

                   DataSynchronizer Pipeline

            (Timestamp Alignment & Forward/Back Fill)

                               │

                               ▼

                   Synchronized Parquet Output

                    (E01.parquet \- N10.parquet)

## **Data Channels & Modalities**

* **Electroencephalography (EEG)**: Multi-channel continuous neural recordings cropped and resampled to a common sampling frequency of 100 Hz. Spectral power density features are calculated across standard frequency bands: Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–45 Hz).  
* **Vehicle CAN-Bus Telemetry**: High-precision vehicle state parameters including longitudinal speed (`speed_mps`), calculated acceleration (`acceleration`), steering angle, braking force, jerk, and lateral lane deviation.  
* **Psychophysiologic Metrics**: Empatica E4 wristband recordings capturing Blood Volume Pulse (BVP at 64 Hz) and Electrodermal Activity (EDA).  
* **Eye-Tracking Telemetry**: Continuous spatial gaze coordinates (`Gaze point X`, `Gaze point Y`), gaze fixation duration, pupil diameter metrics, and saccade amplitudes mapped across screen coordinates.  
* **Traffic Environment**: Contextual indicators tracking ambient vehicle density, surrounding vehicle count, road segment complexity, and intersection classification.  
* **Psychometric & Scale Labels**: Standardized subjective mental workload and stress metrics mapped per subject and scenario, including NASA-TLX, Stress level, Differential Emotions Scale (DES), and Self-Assessment Manikin (SAM).

# **Repository Directory Structure**

├── 1-TrafficRecorder/        \# Traffic density and surrounding vehicle logs

├── 2-CANBus/                 \# CAN-Bus vehicle telemetry files (.csv)

├── 3-Driver/

│   ├── 1-EEG/                \# Raw EEGLAB neural recordings (.set files per scenario)

│   ├── 2-EyeTracking/        \# Raw eye-tracking spatial coordinates (.csv)

│   ├── 3-EDA/                \# Continuous Electrodermal Activity recordings

│   ├── 4-BVP/                \# Raw Blood Volume Pulse streams from Empatica E4

│   └── 5-Scale/              \# Subjective rating scales (NASA-TLX, Stress, DES, SAM)

├── outputs/                  \# Generated execution artifacts

│   ├── checkpoints/          \# Neural network model checkpoints (.pt)

│   ├── figures/              \# Training plots, spectral graphs, and CV diagrams

│   ├── logs/                 \# Execution logs and evaluation records

│   ├── processed/            \# Feature matrices and normalized arrays

│   ├── results/              \# Test evaluation metrics and inference outputs

│   └── synchronized/         \# Consolidated subject Parquet files (e.g., E01.parquet)

├── venv/                     \# Local Python virtual environment

├── acpf\_pipeline\_v7.py       \# Adaptive Cognitive Pressure Field (A-CPF) v7.0 Pipeline

├── acceleration\_calculation.py \# CAN-bus Speed-to-Acceleration calculation module

├── bvp2hrv\_calculation.py    \# BVP to Heart Rate Variability (RMSSD) extraction pipeline

├── etGrid\_calculation.py     \# Eye-tracking 3x3 spatial grid mapping engine

└── requirements.txt          \# Core project dependencies

# **Core Processing & Analysis Pipelines**

## **1\. Adaptive Cognitive Pressure Field (A-CPF) v7.0 Pipeline**

The main model architecture (`acpf_pipeline_v7.py`) implements an Adaptive Cognitive Pressure Field coupled with Koopman operator spectral dynamics and Dynamic Functional Connectivity Graphs (DFCG).                     ┌──────────────────────────────┐

                     │ Synchronized Feature Vectors │

                     └──────────────┬───────────────┘

                                    │

                                    ▼

                     ┌──────────────────────────────┐

                     │   Gated Feature Fusion Net   │

                     └──────────────┬───────────────┘

                                    │

                     ┌──────────────┴──────────────┐

                     ▼                             ▼

        ┌─────────────────────────┐   ┌───────────────────────────┐

        │  Koopman Encoder-Decoder│   │  DFCG Graph Construction  │

        │  (Latent Dynamics, Z)   │   │ (Dynamic Adjacency, Num=24)│

        └────────────┬────────────┘   └─────────────┬─────────────┘

                     │                              │

                     └──────────────┬───────────────┘

                                    ▼

                     ┌──────────────────────────────┐

                     │ Cognitive Pressure Estimator │

                     │   (5-Fold StratifiedGroupCV) │

                     └──────────────────────────────┘

### **Key Technical Features**

* **Spectral Radius Clipping**: Enforces numerical stability on Koopman operator transition matrices by bounding spectral eigenvalues.  
* **Dynamic Graph Connectivity**: DFCG node count is dynamically aligned to match Koopman latent dimension size (\$N \= 24\$).  
* **5-Fold StratifiedGroupKFold CV**: Cross-validation grouping maintains entire subject records intact within single validation folds to prevent data leakage across temporal sequences.  
* **Robust Safety Guards**: Built-in NaN/Inf tensor clamping, automatic non-finite gradient zeroing, mixed-precision auto-casting, and finite loss abort thresholds.

## **2\. BVP-to-HRV (RMSSD) Signal Pipeline**

The script `bvp2hrv_calculation.py` converts raw Blood Volume Pulse streams from the Empatica E4 device into time-domain Heart Rate Variability (HRV) metrics.

* **Peak Detection**: Utilizes `neurokit2.ppg_process()` at a base rate of 64 Hz to isolate systolic heartbeats.  
* **Sliding Window Metrics**: Computes Root Mean Square of Successive Differences (RMSSD) across 5-second windows (320 samples) with a 90% overlap step (288 samples):

\$\$\\text{RMSSD} \= \\sqrt{\\frac{1}{N-1} \\sum\_{i=1}^{N-1} (RR\_{i+1} \- RR\_i)^2}\$\$

## **3\. Eye-Tracking Grid Mapping Module**

The module `etGrid_calculation.py` maps spatial eye-gaze points on 1920×1080 resolution displays into 9 categorical screen zones.

 1 | 2 | 3

|---|

 4 | 5 | 6   (Grid 5 represents the central visual focus area)

\---+---+---

 7 | 8 | 9

* **Central Region**: Configured as \$\\frac{1}{7}\$ of the total screen width and height centered on the screen, explicitly tagged as **Grid 5**.  
* **Peripheral Regions**: The remaining display area is partitioned into an even 3×3 grid surrounding the center zone.

## **4\. Vehicle Acceleration Calculation Module**

The script `acceleration_calculation.py` reads CAN-bus telemetry and derives longitudinal acceleration from speed and timestamp deltas:

\$\$a\_t \= \\frac{v\_t \- v\_{t-1}}{\\Delta t}\$\$

Where vehicle speed \$v\$ is provided in meters per second (speed\_mps) and timestamps are recorded in nanoseconds (\$\\Delta t \= \\frac{\\text{timestamp}\_{\\text{diff}}}{10^9}\$).

# **Installation & Dependencies**

## **System Requirements**

* Python 3.8 or higher (Tested on Python 3.10–3.14)  
* CUDA-compatible GPU recommended for PyTorch model training

## **Environment Setup**

1. Clone the repository to your local workspace:git clone https\://github.com/your-username/eeg-driving-expertise.git  
     
   cd eeg-driving-expertise  
     
2. Create and activate a Python virtual environment:\# On Windows PowerShell  
     
   python \-m venv venv  
     
   .\\venv\\Scripts\\Activate.ps1  
     
   \# On Linux/macOS  
     
   python3 \-m venv venv  
     
   source venv/bin/activate  
     
3. Install required dependencies:pip install \--upgrade pip  
     
   pip install \-r requirements.txt

## **Core Dependencies (`requirements.txt`)**

* `torch >= 1.10.0`  
* `numpy`  
* `pandas`  
* `scipy`  
* `scikit-learn`  
* `matplotlib`  
* `mne`  
* `neurokit2`  
* `pyarrow`  
* `networkx`  
* `tqdm`  
* `openpyxl`

# **Quickstart & Usage Examples**

## **1\. Calculate Acceleration from CAN-Bus Files**

To update raw CAN-bus `.csv` files with derived acceleration values in place:from acceleration\_calculation import process\_csv\_files

\# Process all CAN-bus files under the 2-CANBus directory

process\_csv\_files("./2-CANBus")

## **2\. Map Eye-Tracking Gaze Coordinates to Screen Grids**

To generate spatial grid classifications for gaze tracking CSVs:from etGrid\_calculation import process\_directory

\# Process all eye-tracking files and generate \*\_grid.csv outputs

process\_directory("./3-Driver/2-EyeTracking")

## **3\. Compute Heart Rate Variability (RMSSD) from BVP Streams**

To process raw PPG/BVP streams into windowed HRV metrics:python bvp2hrv\_calculation.py

## **4\. Run Data Synchronization & A-CPF Model Training**

Execute the end-to-end A-CPF v7 pipeline (synchronization, feature extraction, cross-validation, and evaluation):python acpf\_pipeline\_v7.py

# **Dataset Schema & File Specifications**

Synchronized subject files are saved in Apache Parquet format (e.g., `outputs/synchronized/E01.parquet`).

| Column Category | Feature Name | Data Type | Description |
| :---- | :---- | :---- | :---- |
| **Metadata** | `scenario` | `string` | Urban driving task name (e.g., `3_U-Turn`, `0_Baseline`) |
| **EEG Features** | `EEG_theta`, `EEG_alpha` | `float64` | Spectral power densities in Theta (4–8Hz) & Alpha (8–13Hz) bands |
| **EEG Features** | `EEG_beta`, `EEG_gamma` | `float64` | Spectral power densities in Beta (13–30Hz) & Gamma (30–45Hz) bands |
| **CAN Bus** | `CAN_speed_mps` | `float64` | Vehicle speed measured in meters per second |
| **CAN Bus** | `CAN_acceleration` | `float64` | Longitudinal vehicle acceleration in \$m/s^2\$ |
| **CAN Bus** | `CAN_steering_angle` | `float64` | Steering wheel angle measurement |
| **Eye-Tracking** | `EyeTrack_Gaze_point_X` | `float64` | Screen X-coordinate for gaze position (0–1920) |
| **Eye-Tracking** | `EyeTrack_Gaze_point_Y` | `float64` | Screen Y-coordinate for gaze position (0–1080) |
| **Eye-Tracking** | `EyeTrack_Grid_Number` | `float64` | Categorical 3x3 screen zone (1–9, central focus area \= 5\) |
| **Biometrics** | `EDA_amplitude` | `float64` | Skin conductance level from wristband sensor |
| **Biometrics** | `BVP_RMSSD` | `float64` | Root Mean Square of Successive Differences HRV metric |
| **Environment** | `Traffic_density` | `float64` | Surrounding traffic density index |
| **Scales / Labels** | `NASA_TLX` | `float64` | Subjective mental workload score |
| **Scales / Labels** | `Stress` | `float64` | Self-reported driver stress/arousal rating |

# **Experimental Scenarios**

The driving protocol includes 1 baseline condition and 13 urban driving tasks:

1. `0_Baseline`: Resting baseline state  
2. `1_EnterAuxiliaryStreet`: Merging onto auxiliary road segments  
3. `2_EnterMainStreet`: Merging into primary traffic streams  
4. `3_U-Turn`: Executing U-turn maneuvers  
5. `4_StraightDrive`: Uninterrupted straight line urban driving  
6. `5_RightTurn-P1` / `6_RightTurn-P2`: Right turn phase maneuvers  
7. `7_LeftTurn-1` / `8_LeftTurn-2` / `13_LeftTurn-3`: Unprotected left turns  
8. `9_RightTurn-1` to `12_RightTurn-4`: Multi-lane right turn tasks

# **License & Attribution**

This project and its original source implementations are distributed under the BSD 3-Clause License.Copyright (c) 2026, EEG Driving Expertise Research Team

All rights reserved.

Redistribution and use in source and binary forms, with or without

modification, are permitted provided that the following conditions are met:

1\. Redistributions of source code must retain the above copyright notice, this

   list of conditions and the following disclaimer.

2\. Redistributions in binary form must reproduce the above copyright notice,

   this list of conditions and the following disclaimer in the documentation

   and/or other materials provided with the distribution.

3\. Neither the name of the copyright holder nor the names of its

   contributors may be used to endorse or promote products derived from

   this software without specific prior written permission.

For questions regarding access to the dataset, dataset citation requests, or modeling inquiries, please open an issue in the project repository or contact Person.