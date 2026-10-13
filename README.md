# hybrid-sdn-traffic-engineering
Hybrid LSTM-Transformer congestion prediction and PPO-DQN based adaptive routing for Software-Defined Networks.

Dataset Description

This repository contains an SDN network congestion dataset and selected Python scripts used for dataset preparation and preprocessing.

The dataset was generated using Mininet, Open vSwitch, iperf, and TShark.

Total records: 77,487

Raw dataset columns: 23

Congestion classes: Normal (0), Moderate (1), Heavy (2)

Records per class: 25,829

Repository Contents

dataset/: Original network congestion dataset in CSV format.

scripts/: Dataset generation, feature extraction, and preprocessing scripts included in this release.

preprocessing_metadata.json: Preprocessing configuration and feature information.

Preprocessing

The preprocessing pipeline includes feature engineering, segment-aware data splitting, MinMax normalization fitted on training data, and sequence generation.

The configured training, validation, and test split is 70%, 10%, and 20%, respectively. The final prediction input uses 24 features and a sequence length of 20.

Dataset Usage

The dataset can be used for academic research in network congestion classification, traffic analysis, congestion prediction, and Software-Defined Networking.

Access

This is a private repository. Access is restricted to authorized collaborators.
