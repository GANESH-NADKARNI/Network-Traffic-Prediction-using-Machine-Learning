# Network Traffic Prediction using Machine Learning

🚀 **IMPORTANT**

This repository supports **two different workflows**:

### Method 1 — Wireshark Traffic Prediction
Use this if you have:
- `.pcapng` files
- Wireshark packet captures
- Wireshark text exports

### Method 2 — Kaggle Dataset Training
Use this if you downloaded datasets such as:
- CIC-IDS2017
- UNSW-NB15
- KDD Cup 1999
- Network Traffic Flows

➡️ **If you are using a Kaggle dataset, skip directly to Method 2.**

---

# Features

## Wireshark Workflow
- Analyze real network captures
- Parse `.pcapng` and Wireshark exports
- Generate traffic statistics
- Predict future traffic metrics
- Generate reports and visualizations

## Kaggle Workflow
- Train ML models from network datasets
- Automatic dataset column mapping
- Forecast future traffic
- Save reusable models
- Generate feature importance charts

---

# Repository Structure

```text
Network-Traffic-Prediction-using-Machine-Learning/
│
├── run.py
├── traffic_predictor.py
├── visualize.py
├── generate_sample_data.py
│
├── kaggle_train.py
├── kaggle_predict.py
│
├── requirements.txt
└── README.md
```

---

# METHOD 1 — Wireshark Traffic Prediction

Use this workflow if you have real packet capture files.

## Supported Formats

| Format | Description |
|----------|-------------|
| .pcapng | Wireshark capture |
| .txt | Wireshark plain text export |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Using PCAPNG

```bash
python run.py capture.pcapng
```

---

## Run Using Wireshark Text Export

```bash
python run.py capture.txt
```

---

## Demo Mode

```bash
python run.py
```

---

## Available Options

```bash
--window N
```

Aggregation window in seconds.

```bash
--model-dir
```

Directory for trained models.

```bash
--output-dir
```

Directory for reports and charts.

```bash
--no-plots
```

Disable graph generation.

---

## Output Files

```text
output/
├── prediction_output.txt
├── prediction_report.json
└── plots/
    ├── dashboard.png
    ├── protocol_distribution.png
    └── traffic_timeseries.png
```

---

## Wireshark Export Instructions

1. Open capture in Wireshark
2. File → Export Packet Dissections
3. Select "As Plain Text"
4. Enable "Packet Bytes"
5. Save as `.txt`

---

## Wireshark Prediction Outputs

### Throughput / Bandwidth

Example:

```text
At 3:00 PM tomorrow,
the incoming traffic will be 1.2 Gbps.
```

### Packet Count

Example:

```text
We expect 15,000 packets per second
during the next prediction window.
```

### Link Utilization

Example:

```text
The backbone link will be at 78%
capacity between T1 and T2.
```

---

# METHOD 2 — Kaggle Dataset Training

⚠️ If you downloaded a dataset from Kaggle, start here.

---

## Recommended Datasets

### CIC-IDS2017 (Recommended)

https://www.kaggle.com/datasets/cicdataset/cicids2017

### UNSW-NB15

https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

### Network Traffic Flows

https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps

### KDD Cup 1999

https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data

---

## Download Dataset

### Option A — Manual Download

1. Open dataset page
2. Click Download
3. Extract ZIP
4. Place CSV file beside `kaggle_train.py`

---

### Option B — Kaggle API

Install Kaggle:

```bash
pip install kaggle
```

Create API token:

https://www.kaggle.com/settings

Download dataset:

```bash
kaggle datasets download -d cicdataset/cicids2017 --unzip
```

---

## Train Models

Basic training:

```bash
python kaggle_train.py --file dataset.csv
```

Example:

```bash
python kaggle_train.py --file Friday-WorkingHours.csv
```

Custom model directory:

```bash
python kaggle_train.py --file dataset.csv --model-dir kaggle_models
```

Limit rows:

```bash
python kaggle_train.py --file dataset.csv --rows 500000
```

---

## Models Trained

Three independent Gradient Boosting models are created:

| Model | Prediction |
|---------|-----------|
| Model 1 | Throughput/Bandwidth (Mbps) |
| Model 2 | Packet Count (pkt/s) |
| Model 3 | Link Utilization (%) |

---

## Training Output

```text
kaggle_models/
├── kaggle_throughput_mbps_model.pkl
├── kaggle_throughput_mbps_scaler.pkl
├── kaggle_packet_count_model.pkl
├── kaggle_packet_count_scaler.pkl
├── kaggle_link_util_pct_model.pkl
├── kaggle_link_util_pct_scaler.pkl
├── kaggle_training_metrics.json
└── feature_importance_plots
```

---

## Predict Future Traffic

Interactive mode:

```bash
python kaggle_predict.py --model-dir kaggle_models
```

Predict using new CSV:

```bash
python kaggle_predict.py \
    --model-dir kaggle_models \
    --file new_data.csv
```

Custom forecast:

```bash
python kaggle_predict.py \
    --model-dir kaggle_models \
    --steps 20 \
    --window 60
```

Disable graph:

```bash
python kaggle_predict.py \
    --model-dir kaggle_models \
    --no-plot
```

---

## Kaggle Prediction Outputs

### Throughput / Bandwidth

```text
At 3:00 PM tomorrow,
the incoming traffic will be X Mbps/Gbps.
```

### Packet Count

```text
We expect X,XXX packets per second
during the next prediction window.
```

### Link Utilization

```text
The backbone link will operate at X% capacity.
```

---

# Machine Learning Pipeline

```text
Raw Network Data
       │
       ▼
Feature Engineering
       │
       ▼
Traffic Metrics
       │
       ▼
Gradient Boosting Regressor
       │
       ├── Throughput Model
       ├── Packet Count Model
       └── Link Utilization Model
       │
       ▼
Future Traffic Forecast
       │
       ▼
Reports + Visualizations
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib

---

# Quick Start

## For Wireshark Users

```bash
pip install -r requirements.txt

python run.py capture.pcapng
```

---

## For Kaggle Users

```bash
pip install -r requirements.txt

python kaggle_train.py --file dataset.csv

python kaggle_predict.py --model-dir kaggle_models
```

---
