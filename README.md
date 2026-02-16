<div align="center">

# 🧠 Smart Task Manager

### *AI-Powered System Process Monitor & Security Analyzer*

[![GitHub](https://img.shields.io/badge/💻_View-Repository-181717?style=for-the-badge&logo=github)](https://github.com/devanshrawat27/Smart-Task-Manager)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chart.js&logoColor=white)

*Intelligent system monitoring with machine learning-powered threat detection* 🛡️

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [ML Models](#-machine-learning-models) • [API](#-api-documentation)

---

</div>

## 🌟 About The Project

**Smart Task Manager** is an advanced system monitoring tool that combines real-time process tracking with machine learning algorithms to detect anomalies, predict malicious behavior, and optimize system performance. Built with Python and Flask, it provides a comprehensive web-based dashboard for system administrators and security professionals.

### 🎯 Why Smart Task Manager?

- **🤖 AI-Powered Detection** - Machine learning models identify suspicious processes
- **📊 Real-Time Monitoring** - Live system metrics and process analytics
- **🔒 Security Analysis** - Behavioral pattern recognition for threat detection
- **📈 Performance Optimization** - Resource usage insights and recommendations
- **💾 Data Logging** - Comprehensive process history for model training
- **🎨 Modern UI** - Intuitive web interface with interactive charts

---

## ✨ Features

<div align="center">

| Feature | Description |
|---------|-------------|
| 🔍 **Process Monitor** | Real-time tracking of all system processes with CPU, memory, and I/O metrics |
| 📊 **Analytics Dashboard** | Visual representation of system performance trends and patterns |
| 🛡️ **Security Scanner** | ML-based detection of potentially malicious or anomalous processes |
| 💻 **System Specifications** | Detailed hardware and software information display |
| 📝 **Behavioral Logging** | Automated data collection for continuous model improvement |
| 🤖 **Predictive Analysis** | AI models predict process behavior based on historical data |
| ⚡ **Auto-Kill Mode** | Automated termination of flagged malicious processes (Optional) |
| 📈 **Resource Optimization** | Identifies resource-heavy processes for performance tuning |

</div>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Interface (Flask)                    │
│              HTML • CSS • JavaScript • Chart.js              │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Flask Backend Server                      │
│              Routes • API Endpoints • Logic                  │
└────────────┬───────────────────────────────┬────────────────┘
             │                               │
┌────────────▼──────────────┐   ┌───────────▼────────────────┐
│   System Monitor Module   │   │   ML Prediction Engine     │
│   • psutil                │   │   • Scikit-learn           │
│   • Process Tracking      │   │   • Model Loading          │
│   • Metrics Collection    │   │   • Anomaly Detection      │
└────────────┬──────────────┘   └───────────┬────────────────┘
             │                               │
┌────────────▼───────────────────────────────▼────────────────┐
│                   Data Layer                                 │
│   • CSV Logging (logs/system_data_behavioral.csv)          │
│   • Trained Models (models/*.pkl)                           │
│   • Configuration Files                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Tech Stack

### 🎨 Core Technologies

```python
{
  "language": "Python 3.8+",
  "framework": "Flask",
  "ml_library": "Scikit-learn",
  "data_processing": ["Pandas", "NumPy"],
  "system_monitoring": "psutil",
  "frontend": ["HTML5", "CSS3", "JavaScript"],
  "visualization": "Chart.js",
  "data_storage": "CSV (Training Data)"
}
```

### 📚 Key Dependencies

| Package | Purpose |
|---------|---------|
| **Flask** | Web framework for API and UI |
| **psutil** | System and process monitoring |
| **scikit-learn** | Machine learning algorithms |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computing |
| **matplotlib** | Data visualization (training) |
| **joblib** | Model serialization |

---

## 🚀 Installation

### 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- 4GB RAM minimum
- Windows/Linux/macOS

### 📥 Step-by-Step Setup

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/devanshrawat27/Smart-Task-Manager.git
cd Smart-Task-Manager
cd custom_task_manager_pro_v7_20251111081100
```

#### 2️⃣ Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Unix/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**If `requirements.txt` is missing, install manually:**

```bash
pip install flask psutil scikit-learn pandas numpy matplotlib joblib
```

#### 4️⃣ Run the Application

```bash
python -m app.main
```

#### 5️⃣ Access the Dashboard

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

🎉 **You're all set!** The dashboard should now be running.

---

## 📖 Usage

### 🖥️ Dashboard Overview

The web interface provides five main tabs:

#### 1. **Processes Tab** 🔍
- View all running processes in real-time
- Monitor CPU usage, memory consumption, and I/O operations
- Search and filter processes
- Kill processes (with confirmation)
- Color-coded risk indicators

#### 2. **Analytics Tab** 📊
- Interactive charts showing system trends
- CPU and memory usage over time
- Process count distribution
- Resource utilization graphs
- Export data functionality

#### 3. **Security Tab** 🛡️
- ML-powered threat detection
- Anomaly score for each process
- Behavioral pattern analysis
- Suspicious process alerts
- Auto-kill toggle (disabled by default)

#### 4. **System Specs Tab** 💻
- Complete hardware information
- OS details and version
- CPU specifications
- RAM and disk usage
- Network information

#### 5. **About Us Tab** ℹ️
- Project information
- Developer details
- Documentation links
- Version history

---

## 🤖 Machine Learning Models

### 📊 Data Collection

The system automatically logs process behavior to:
```
logs/system_data_behavioral.csv
```

**Logged Features:**
- Process Name
- CPU Usage (%)
- Memory Usage (MB)
- I/O Read/Write (bytes)
- Thread Count
- Handle Count (Windows)
- Network Connections
- Timestamp

### 🧪 Model Training

Once sufficient data is collected (recommended: 1000+ samples), train the ML models:

```bash
python train_models_v8.py
```

**Training Process:**
1. Loads historical data from CSV logs
2. Performs feature engineering
3. Trains multiple ML algorithms:
   - Random Forest Classifier
   - Gradient Boosting
   - Isolation Forest (Anomaly Detection)
4. Evaluates model performance
5. Saves trained models to `models/` directory

### 🎯 Model Architecture

```python
# Supported ML Algorithms
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Isolation Forest": IsolationForest(contamination=0.1),
    "SVM": SVC(kernel='rbf'),
    "Neural Network": MLPClassifier(hidden_layers=(100, 50))
}
```

## 🔧 Configuration

### ⚙️ Application Settings

Edit `app/config.py` (if available) or modify these parameters:

```python
# Server Configuration
HOST = "127.0.0.1"
PORT = 5000
DEBUG = False  # Set to False in production

# Security Settings
AUTO_KILL_ENABLED = False  # Enable automatic process termination
THREAT_THRESHOLD = 0.75    # Anomaly score threshold (0-1)

# Data Collection
LOG_INTERVAL = 60          # Seconds between data logging
MAX_LOG_SIZE = 10000       # Maximum CSV entries before rotation

# Model Settings
MODEL_PATH = "models/trained_model.pkl"
RETRAIN_INTERVAL = 7       # Days before model retraining
```

## 📡 API Documentation

### REST Endpoints

#### Get All Processes
```http
GET /api/processes
```
**Response:**
```json
{
  "processes": [
    {
      "pid": 1234,
      "name": "chrome.exe",
      "cpu_percent": 5.2,
      "memory_mb": 450.3,
      "status": "running",
      "threat_score": 0.15
    }
  ]
}
```

#### Kill Process
```http
POST /api/kill-process
Content-Type: application/json

{
  "pid": 1234
}
```

#### Get System Stats
```http
GET /api/system-stats
```

#### Get Security Report
```http
GET /api/security/scan
```

---

## 📁 Project Structure

```
Smart-Task-Manager/
│
├── custom_task_manager_pro_v7_20251111081100/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # Application entry point
│   │   ├── routes.py            # Flask routes
│   │   ├── monitor.py           # Process monitoring logic
│   │   └── ml_predictor.py      # ML inference engine
│   │
│   ├── models/
│   │   └── trained_model.pkl    # Serialized ML models
│   │
│   ├── logs/
│   │   └── system_data_behavioral.csv  # Training data
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css
│   │   └── js/
│   │       └── dashboard.js
│   │
│   ├── templates/
│   │   └── index.html           # Main dashboard template
│   │
│   ├── train_models_v8.py       # Model training script
│   ├── requirements.txt         # Python dependencies
│   └── README.md
│
└── README.md                    # This file
```

---

## 🧪 Testing

### Manual Testing
```bash
# Test process monitoring
python -m app.monitor

# Test ML predictions
python -m app.ml_predictor

# Run Flask in debug mode
python -m app.main --debug
```

### Unit Tests (if implemented)
```bash
pytest tests/
```

## 🐛 Troubleshooting

### Common Issues

**❌ Permission Denied Error**
```bash
# Run with elevated privileges (Windows)
Run Command Prompt as Administrator

# Unix/macOS
sudo python -m app.main
```

**❌ Port Already in Use**
```bash
# Change port in app/main.py
app.run(host='127.0.0.1', port=5001)
```

**❌ Model File Not Found**
```bash
# Train models first
python train_models_v8.py
```

**❌ psutil Access Denied**
```
# Some system processes require admin rights
# This is expected behavior - skip those processes
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🔀 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation accordingly
- Test on multiple OS platforms

---

## 🙏 Acknowledgments

- **psutil** - Cross-platform process monitoring
- **scikit-learn** - Machine learning framework
- **Flask** - Web framework
- **Chart.js** - Interactive visualizations
- Open-source community for inspiration

---

## 👨‍💻 Developer

<div align="center">

### **Devansh Rawat**

*Software Engineering Student | ML Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/devanshrawat27)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](#)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](#)

</div>
---

## 📊 Project Stats

<div align="center">

![GitHub Stars](https://img.shields.io/github/stars/devanshrawat27/Smart-Task-Manager?style=social)
![GitHub Forks](https://img.shields.io/github/forks/devanshrawat27/Smart-Task-Manager?style=social)
![GitHub Watchers](https://img.shields.io/github/watchers/devanshrawat27/Smart-Task-Manager?style=social)

</div>

---

<div align="center">

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=devanshrawat27/Smart-Task-Manager&type=Date)](https://star-history.com/#devanshrawat27/Smart-Task-Manager&Date)

---

**Made with ❤️ and 🤖 by [Devansh Rawat](https://github.com/devanshrawat27)**

*"Securing systems, one process at a time"* 🛡️

### ⭐ If this project helped you, consider giving it a star!

</div>
