# Network Traffic Prediction System

Predicts future network traffic metrics from Wireshark captures using ML (Gradient Boosting).

## Predicted Outputs

- **Throughput/Bandwidth** — "At 3:00 PM tomorrow, the incoming traffic will be X Gbps."
- **Packet Count** — "We expect X,XXX packets per second during the next 10-minute window."
- **Link Utilization** — "The backbone link will be at X% capacity between T1 and T2."

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with your Wireshark data

**Hex-dump .txt file (from Wireshark Edit → Export → as Plain Text):**
```bash
python run.py your_capture.txt
```

**pcapng file:**
```bash
python run.py your_capture.pcapng
```

**Demo mode (no file needed):**
```bash
python run.py
```

### 3. Options
```
--window N      Aggregation window in seconds (default: 1)
--model-dir     Directory to save trained models (default: models/)
--output-dir    Directory for reports/charts (default: output/)
--no-plots      Skip matplotlib chart generation
```

## Output Files

```
output/
├── prediction_output.txt       ← Human-readable report
├── prediction_report.json      ← Machine-readable JSON
└── plots/
    ├── dashboard.png           ← Full prediction dashboard
    ├── protocol_distribution.png
    └── traffic_timeseries.png
models/
├── throughput_mbps_model.pkl
├── throughput_mbps_scaler.pkl
├── packet_count_model.pkl
├── packet_count_scaler.pkl
├── link_util_pct_model.pkl
└── link_util_pct_scaler.pkl
```

## Supported File Formats

| Format | Description |
|--------|-------------|
| `.txt` | Wireshark hex-dump text export |
| `.pcapng` | Wireshark native capture format |

### How to export hex dump from Wireshark
1. Open your `.pcapng` in Wireshark
2. Go to **File → Export Packet Dissections → As Plain Text**
3. Check **Packet bytes** option
4. Save as `.txt`

## Architecture

```
Raw Packets (.txt / .pcapng)
        │
        ▼
  Parser (Layer 2/3/4 decode)
        │
        ▼
  Feature Engineering (1s windows)
  • packet_count, total_bytes, throughput
  • protocol mix, unique IPs
  • lag features (1,2,3,5 steps)
  • rolling averages (3,5,10 windows)
        │
        ▼
  Gradient Boosting Regressor (×3)
  • Model 1 → Throughput (Mbps)
  • Model 2 → Packet Count (pkt/s)
  • Model 3 → Link Utilization (%)
        │
        ▼
  Predictions + Report + Charts
```

## No Paid APIs

Everything runs 100% locally using free open-source libraries:
- scikit-learn (ML)
- pandas / numpy (data)
- matplotlib (charts)
- joblib (model persistence)
