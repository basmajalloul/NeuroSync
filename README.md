# NeuroSync: Multimodal Neurocognitive Analysis Platform

## Overview
NeuroSync is an open-source platform designed for **real-time neurocognitive data analysis** using synchronized **EEG, HRV, and video** modalities. This repository includes the **dashboard**, **analysis**, and **validation** tools, as well as a **data acquisition module** for transparency and reproducibility.

## Features
- **Real-time dashboard** for EEG, HRV, and video data visualization.
- **Comprehensive statistical analysis** of neurophysiological signals.
- **Validation tools** for synchronization accuracy.
- **Simulated data acquisition module** to generate test datasets.
- **Scalability** for additional physiological modalities.

## Repository Structure
```
📂 NeuroSync
│── 📂 data/                      # Sample simulated datasets
│── 📂 scripts/                   # Core application scripts
│   │── neurosync_dashboard.py    # Main dashboard for visualization
│   │── neurosync_analysis.py    # Data analysis script
│   │── neurosync_analysis_validation.py  # Analysis validation tool
│   │── neurosync_dashboard_validation.py # Dashboard validation tool
│   │── main.py                    # Data acquisition module (simulated data)
│── LICENSE                        # License file
│── README.md                      # Documentation
```

## Installation
To install dependencies, run:
```sh
pip install -r requirements.txt
```

## Usage
### Running the Dashboard
```sh
python scripts/neurosync_dashboard.py
```
This launches the **interactive dashboard**, where users can visualize and annotate EEG, HRV, and video data.

### Running Data Analysis
```sh
python scripts/neurosync_analysis.py
```
This script computes **EEG band power**, **HRV statistics**, and **cross-modality correlations**.

### Running Validation
To validate the analysis results, execute:
```sh
python scripts/neurosync_analysis_validation.py
```
To validate the dashboard functionality:
```sh
python scripts/neurosync_dashboard_validation.py
```

### Simulated Data Acquisition
```sh
python scripts/neurosync_main.py
```
This script generates **synthetic EEG and HRV signals** to test and validate the platform.

## Licensing
This project is licensed under the MIT License. See `LICENSE` for details.

## Contribution
Contributions are welcome! Please fork the repository and submit a pull request.

## Citation
If you use this project in your research, please cite our work:
```
@article{neurosync2024,
  author = {Jalloul, Basma and Chaabene, Siwar and Bouaziz, Bassem and Mahdi, Walid},
  title = {NeuroSync: A Multimodal Synchronization Framework for Neurocognitive Diagnostics},
  journal = {Journal of Systems Architecture},
  year = {2024},
  publisher = {Elsevier}
}
```
