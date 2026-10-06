# 🎯 RiskRadar — AI Student Dropout Risk Prediction & Faculty Copilot

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-7.3-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

**An intelligent early-warning platform that predicts student dropout risk using machine learning, explainable AI, and an interactive faculty copilot.**

[Explore Features](#-key-features) • [Quickstart Guide](#-quickstart--installation) • [System Architecture](#-system-architecture) • [API Reference](#-api-endpoints) • [AI Copilot Commands](#-faculty-ai-copilot)

</div>

---

## 📖 Overview

Student dropout is one of higher education's most urgent challenges. Traditional institutional retention efforts are often **reactive** — intervening only when students are already failing or withdrawing.

**RiskRadar** (DropGuard AI) is a proactive, explainable retention analytics platform. It combines:
- **Ensemble Machine Learning** (Random Forest, XGBoost, LightGBM, and temporal LSTM sequence modeling).
- **Explainable AI (XAI)** powered by SHAP values and heuristic feature attributions to answer *why* a student is at risk.
- **Real-Time Faculty Analytics Dashboard** providing bird's-eye metrics, dynamic cohort distributions, and actionable intervention plans.
- **Integrated AI Faculty Copilot** offering conversational student lookups, class summaries, and tailored early warnings right from the dashboard.

---

## ✨ Key Features

### 🛡️ 1. Multi-Tiered Dropout Risk Prediction
- **Ensemble ML Pipeline**: Analyzes academic performance (GPA, backlog status), behavioral engagement (attendance rate, assignment completion), and socio-demographic indicators.
- **Calibrated Risk Levels**: Categorizes students into **Low Risk**, **Medium Risk**, and **High Risk** tiers with calibrated confidence scores.

### 📊 2. Modern Faculty Analytics Dashboard
- **Executive KPI Cards**: Real-time counts of total enrolled students, high-risk flags, average class attendance, and model AUC-ROC metrics.
- **Dynamic Data Visualizations**: Recharts-powered Risk Distribution Doughnuts and Attendance vs. Risk correlation plots.
- **Student Risk Directory**: Searchable, filterable student registry with granular metrics, individual risk badges, and recommended actions.

### 🤖 3. AI Faculty Copilot (Chatbot Assistant)
- Accessible anytime via the **sidebar navigation** or the floating **"Ask AI Copilot"** widget.
- Natural language query parser responding to student USN lookups, risk factor breakdowns, class summaries, and urgent warnings.

### 📂 4. Drag-and-Drop Batch Ingestion
- Upload institutional CSV records through an intuitive upload interface.
- Automatically processes tabular datasets through the preprocessing & feature engineering pipeline.

### 💡 5. Prescriptive Interventions
- Moves beyond passive alerts by generating actionable intervention recommendations (e.g., *1-on-1 Academic Mentoring*, *Counselor Meeting*, *Peer Tutoring*, *Financial Aid Consultation*).

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[Student Records / CSV Upload] --> B[Data Cleaning & Imputation]
        B --> C[Feature Engineering & Scaling]
    end

    subgraph ML & Intelligence Engine
        C --> D[Ensemble Classifier\nRandom Forest + XGBoost + LightGBM]
        C --> E[Temporal LSTM Sequence Model]
        D & E --> F[Risk Calibration & SHAP Factor Extraction]
    end

    subgraph Backend API (FastAPI)
        F --> G[FastAPI Service\nPort 8000]
        G --> H[/api/student-predictions]
        G --> I[/api/upload-student-data]
        G --> J[/api/chat NLP Engine]
    end

    subgraph Faculty Dashboard (React + Vite)
        H & I --> K[Executive Dashboard & Metrics]
        H --> L[Student Risk Directory]
        J --> M[AI Copilot Assistant Modal]
    end
```

---

## 🔬 Model Performance & Evaluation

The predictive engine is trained on multi-attribute student behavioral data with class balancing (SMOTE) and feature engineering:

| Metric | Score / Benchmark |
| :--- | :--- |
| **Model ROC-AUC** | **80.6%** |
| **Prediction Accuracy** | **~85%** |
| **Key Risk Indicators** | Attendance Rate, GPA, LMS Activity, Assignment Delays, Travel Distance |
| **Explainability** | SHAP Summary Plots & Local Feature Attributions |

---

## 🚀 Quickstart & Installation

### Prerequisites
- **Python 3.10+** (Python 3.12 recommended)
- **Node.js v18+** & **npm**
- **Git**

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Archi-jain19/RiskRadar.git
cd RiskRadar
```

---

### 2️⃣ Backend Setup (FastAPI)

```bash
# Navigate to backend directory
cd backend

# (Optional) Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The backend server will start at:
- **API URL**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 3️⃣ Frontend Setup (React + Vite)

In a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install

# Start Vite dev server
npm run dev -- --host 127.0.0.1
```

Open your browser at:
👉 **[http://127.0.0.1:5173/](http://127.0.0.1:5173/)**

---

## 💬 Faculty AI Copilot Commands

The built-in conversational copilot supports prompt queries such as:

| Command Prompt | Example Query | Action |
| :--- | :--- | :--- |
| **Check Student** | `Check student USN: 1RV22CS002` | Fetches student risk score, health index, and key factors |
| **Explain Risk** | `Why is student 1RV22CS005 at risk?` | Highlights specific triggering metrics (e.g. low attendance, GPA) |
| **High Risk List** | `Show high risk students` | Returns sorted list of all critically flagged students |
| **Class Summary** | `Show class risk summary` | Summarizes cohort health and priority cases |
| **Top At-Risk** | `Show top 5 students at risk` | Ranks students needing immediate intervention |
| **Warning Signals**| `Show warning signals` | Highlights leading indicators across the cohort |
| **General Help** | `help` | Lists all supported queries and syntax |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check and engine status |
| `GET` | `/api/student-predictions` | Returns full student roster with calculated risk predictions |
| `POST` | `/api/upload-student-data` | Upload and batch-evaluate student CSV files |
| `POST` | `/api/chat` | AI Copilot conversational endpoint |
| `POST` | `/api/predict` | Predicts dropout probability for a single student JSON payload |
| `GET` | `/students/high-risk` | Retrieves list of students exceeding risk threshold |
| `GET` | `/students/{usn}` | Looks up individual student profile by USN |

---

## 📁 Repository Structure

```text
RiskRadar/
├── backend/
│   ├── main.py                  # FastAPI server & route controllers
│   ├── model.py                 # Ensemble model interface & feature engineering
│   ├── chatbot.py               # NLP rule & intent parser for Faculty Copilot
│   ├── supabase_client.py       # Cloud DB integration (with offline fallback)
│   ├── data/
│   │   ├── students.csv         # Baseline student records
│   │   └── presentation_students.csv
│   └── requirements.txt         # Backend Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx    # Analytics cards & Recharts visualizations
│   │   │   ├── Students.jsx     # Student directory page
│   │   │   ├── StudentTable.jsx # Interactive student table & detail drawer
│   │   │   ├── UploadData.jsx   # CSV upload panel
│   │   │   ├── Sidebar.jsx      # Navigation sidebar with Copilot launcher
│   │   │   └── ChatbotButton.jsx# Floating & modal Faculty Copilot
│   │   ├── App.jsx              # Main routing & state provider
│   │   └── index.css            # Custom responsive design system
│   ├── package.json             # Frontend dependencies
│   └── vite.config.js           # Vite build configuration
│
├── models/                      # ML pipeline training & artifacts
│   ├── ensemble_model.py
│   ├── lstm_model.py
│   ├── train_model.py
│   └── *.png                    # Confusion matrix & SHAP importance plots
│
├── PRD.md                       # Product Requirements Document
└── README.md                    # Project documentation
```

---

## 🛠️ Tech Stack

- **Frontend**: React 18, Vite, React Router 7, Recharts, Lucide Icons, Vanilla CSS Design System.
- **Backend**: FastAPI, Uvicorn, Pydantic, Python Dotenv.
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, PyTorch (LSTM), Joblib, Pandas, NumPy.
- **Database / Cloud**: Supabase (PostgreSQL) with seamless local offline storage support.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
