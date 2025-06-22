# SemiAuto-Clustring

**SemiAuto-Clustring** is a semi-automated pipeline for **unsupervised clustering** workflows. It automates data ingestion, preprocessing, clustering model training, evaluation, and report generation—while giving users control over parameters and input formats. It’s designed for **ML engineers**, **MLOps practitioners**, and **software developers** who want a flexible yet automated clustering solution.

---

## 🚀 Features

- 📦 **End-to-End Pipeline** – From raw data to cluster labels and evaluation reports.
- ⚙️ **Semi-Automated Design** – Combine user-specified configs (e.g. number of clusters) with auto-training logic.
- 🌐 **API + Web UI Support** – Invoke via CLI, REST API, or browser interface.
- 🧱 **Modular Architecture** – Easily plug in new clustering algorithms or metrics.
- 🐳 **Dockerized** – Build and deploy as a self-contained container.

---

## 🧱 Architecture Overview

```
SemiAuto-Clustring/
├── app.py                     # Main FastAPI/CLI app entrypoint
├── semiauto_clustering/       # Core modules (clustering, metrics, etc.)
│   ├── data_preprocessing.py
│   ├── model_building.py
│   ├── evaluation.py
│   └── ...
├── config/                    # Config files
├── data/                      # Raw and processed datasets
├── reports/                   # Output reports (CSV, JSON, HTML, plots)
├── templates/                 # HTML templates for web UI
├── static/                    # CSS/JS assets
├── Dockerfile                 # Container build config
├── requirements.txt           # Python dependencies
└── Makefile                   # Common CLI commands
```

---

## ⚙️ Installation

### 🔧 Requirements

- Python 3.8+
- `pip`, `venv`, or Conda
- (Optional) Docker

### 📦 Install via pip

```bash
pip install semiauto-clustering
```

### 📥 From Source

```bash
git clone https://github.com/Akshat-Sharma-110011/SemiAuto-Clustring.git
cd SemiAuto-Clustring
pip install -r requirements.txt
pip install -e .
```

### 🐳 Docker Setup

```bash
docker build -t semiauto-clustering .
docker run -p 5000:5000 semiauto-clustering
```

---

## 🔧 Usage Guide

### 🖥️ CLI Usage

```bash
semiauto-clustering --input data/mydata.csv --clusters 4 --output reports/
```

Or via config:

```bash
semiauto-clustering --config config.yaml
```

### 🌐 Web UI / API

Run:

```bash
python app.py --server
```

Then open:  
```
http://localhost:5000/
```

#### API Example

```bash
curl -X POST http://localhost:5000/cluster \
     -F file=@data.csv \
     -F clusters=3
```

### 📁 Inputs & Outputs

- **Input:** CSV file with feature columns (no labels).
- **Output:** Clustered data CSV, JSON summary, silhouette score, optional visualizations under `reports/`.

---

## 🛠️ Customization

- 🔧 **Algorithm & Metric Control** – Set via CLI or `intel.yaml` config file.
- 🧩 **Add New Algorithms** – Extend `model_building.py` with custom clustering logic.
- ⚙️ **Preprocessing Hooks** – Modify `data_preprocessing.py` to clean/scale/encode as needed.
- 🔌 **Integrate with Tools** – Use as a Python library or plug into MLOps workflows.

---

## ⚠️ Limitations & Roadmap

### ❌ Current Limitations

- Manual cluster count (`--clusters`) is required.
- Limited algorithms (KMeans, DBSCAN, etc. by default).
- In-memory processing only (no big data/distributed support).
- No experiment tracking (e.g. MLflow) yet.

### 📈 Roadmap

- Optimal cluster count suggestion
- Spectral & HDBSCAN integration
- MLflow or DVC support
- Enhanced web UI
- Scalable pipeline (e.g. Dask/Spark support)

---

## 📚 Credits & References

SemiAuto-Clustring is inspired by:

- **[H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)** – A leading AutoML platform.
- **AutoML4Clust** – Research on AutoML for clustering (Vogt et al., EDBT 2021).

> “Automated Machine Learning (AutoML) aims to automate the end-to-end process of applying machine learning to real-world problems.” – Hutter et al.

---

## 🪪 License

Licensed under the [MIT License](LICENSE).
