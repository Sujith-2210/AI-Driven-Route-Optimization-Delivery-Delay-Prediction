# 🚚 AI-Driven Route Optimization & Delivery Delay Prediction

A comprehensive logistics system that predicts delivery delays in real-time and optimizes delivery routes using machine learning and graph algorithms.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-green)

---

## 📋 Table of Contents

- [Objective](#-objective)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dashboard Views](#-dashboard-views)
- [Tech Stack](#-tech-stack)

---

## 🎯 Objective

Develop a system that:
- **Predicts delivery delays** in real-time using ML models
- **Optimizes delivery routes** using A* algorithm
- **Visualizes logistics data** through an interactive dashboard

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Delay Prediction** | Predicts if a delivery will be delayed with probability score |
| **Route Optimization** | A* algorithm for optimal delivery sequencing |
| **Traffic Heatmap** | Visualize delivery density and congestion zones |
| **Driver Analytics** | Performance metrics per driver |
| **Real-time Pipeline** | Prediction logging and model inference API |
| **Mini MLOps** | Automated retraining with model versioning |

---

## 📁 Project Structure

```
Delivery Delay Prediction/
├── Datasets/
│   ├── lade_shanghai_sample.csv      # LaDe dataset (5000 records)
│   ├── Delivery_Logistics.csv
│   └── ecommerce_delivery_analytics.csv
├── models/
│   ├── best_delay_prediction_model.pkl
│   ├── label_encoders.pkl
│   └── model_comparison_results.csv
├── logs/
│   └── predictions.csv              # Prediction logs
├── mlops/
│   └── retrain.py                   # Weekly retraining script
├── app.py                           # Streamlit dashboard
├── route_optimizer.py               # A* route optimization
├── prediction_pipeline.py           # Real-time prediction API
├── delivery_delay_prediction.ipynb  # Training notebook
├── prepared_logistics_dataset.csv   # Final dataset
├── style.css                        # Dashboard styling
└── requirements.txt
```

---

## 🛠 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd "Delivery Delay Prediction"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

---

## 🚀 Usage

### Run Dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### Test Route Optimizer
```bash
python route_optimizer.py
```

### Test Prediction Pipeline
```bash
python prediction_pipeline.py
```

### Retrain Model
```bash
python mlops/retrain.py
```

---

## 📊 Model Performance

Three models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 87.2% | 0.89 | 0.95 | 0.92 | 0.88 |
| XGBoost | 86.8% | 0.88 | 0.96 | 0.92 | 0.87 |
| **Gradient Boosting** | **87.5%** | **0.89** | **0.96** | **0.92** | **0.89** |

**Best Model:** Gradient Boosting Classifier

### Features Used
- Distance (km)
- Package Weight (kg)
- Vehicle Type
- Traffic Level
- Weather Condition
- Road Type

---

## 📱 Dashboard Views

### 1. 📊 Performance Dashboard
- KPIs: Total deliveries, On-time rate, Delays
- Delay analysis by traffic and weather
- Traffic heatmap
- Vehicle performance table

### 2. 🗺️ Route Optimizer
- Select number of delivery stops
- A* algorithm optimization
- Interactive route map
- Distance and time estimates

### 3. 🔮 Delay Predictor
- Input delivery parameters
- Real-time delay probability
- Risk level classification (Low/Medium/High/Critical)
- Actionable recommendations

### 4. 📈 Driver Analytics
- Best performing driver
- Busiest driver
- Top 10 driver performance chart
- Detailed statistics table

---

## 🔧 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.13 |
| **ML Framework** | Scikit-learn, XGBoost |
| **Dashboard** | Streamlit |
| **Visualization** | Plotly, Folium |
| **Data Processing** | Pandas, NumPy |
| **Maps** | Folium, streamlit-folium |

---

## 📦 Dataset

**Source:** Cainiao-AI/LaDe Dataset (Hugging Face)
- 5,000 records from Shanghai
- Real-world delivery data with GPS coordinates
- Timestamps for accept and delivery events

**Simulated Features:**
- Vehicle type, Traffic level, Weather condition, Road type

---

## 🔄 MLOps

The system includes automated retraining capabilities:

```python
from mlops.retrain import run_weekly_retrain
run_weekly_retrain()
```

Features:
- Weekly model retraining
- Performance logging to CSV
- Model versioning
- Automatic best model selection

---

## 📝 License

This project is for educational purposes.

---

## 👤 Author

Developed as part of AI-Driven Route Optimization & Delivery Delay Prediction project.
