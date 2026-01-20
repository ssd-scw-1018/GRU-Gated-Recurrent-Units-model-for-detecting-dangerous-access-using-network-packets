AI-Based Security Policy Engine & Attack Scenario Testing

This project implements an AI-driven security policy engine designed to detect and respond to network attack scenarios (DDoS, Slow Scan, etc.) using the UNSW-NB15 dataset and a GRU-based dual-threshold model.

1. Environment & Prerequisites
The code has been tested and verified in Python 3.10.

Required Libraries
Install the necessary packages using the following command:

	pip install numpy pandas torch scikit-learn matplotlib seaborn pyarrow tqdm

2. Project Structure
Ensure the project root is organized as follows:

├── data_4_split/               # Raw UNSW-NB15 test CSV folder
│    └── UNSW-NB15_4.csv
├── build_scenarios_final.py    # Script for generating attack scenarios
├── feature_importance_auc.py   # Script for analyzing feature importance (AUC Drop)
├── test.py                     # Main security policy engine and testing script
├── gru_dual_threshold_model.pth # Pre-trained GRU model
├── preproc_params.json         # Preprocessing parameters (must be in the same folder as the model)
└── README.md                   # Project documentation

3. Execution Steps for Reproduction
To reproduce the results, execute the scripts in the following order:

[Step 1] Generate Scenario Data

	python build_scenarios_final.py

Result: Preprocesses raw data and generates attack scenarios (DDoS, Slow Scan) in Parquet format within the artifacts_parquet/ directory.

[Step 2] Calculate Feature Importance

	python feature_importance_auc.py

Result: Calculates feature importance based on AUC Drop.

Outputs:

	artifacts_parquet/feature_importance_top10_auc.png: Visualization of top 10 features.

	artifacts_parquet/core_features.json: Selected core features required for test.py.

[Step 3] Run Security Policy Engine & Visualization

	python test.py

test.py automatically performs the following tasks:

 - Verifies scenario quality based on core_features.json.

 - Executes security policy tests on valid scenarios.

 - Classifies each access event as NORMAL, WATCH, or BLOCKED.

 - Generates and provides Deception Files (honeyfiles) for suspicious/blocked activities.

4. Output Summary
Generated Files (in artifacts_parquet/)

	test_decisions.csv: Complete logs of the engine's decisions.

	scenario_y.parquet: Ground truth labels for the selected scenarios.

	Visualizations: Confusion Matrix, Attack Scenario Graphs, and IP Status Pie Charts.

	Deception Files (in fake_files/)
	Automatically generated directories for each scenario (scenario_ddos, scenario_slow_scan, etc.) containing decoy files to 	mislead attackers.

5. Key Features

Dual-Threshold Mechanism: Utilizes GRU model outputs to implement a sophisticated detection logic beyond simple binary classification.

Automated Response: Dynamically manages an IP blocklist and generates deception artifacts in real-time during the test simulation.

Comprehensive Visualization: Provides intuitive graphs to analyze engine performance and attack progression.