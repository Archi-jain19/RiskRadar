# 📋 Product Requirements Document (PRD)

## AI Student Dropout Risk Prediction System — "DropGuard AI"

**Version:** 1.0  
**Date:** March 6, 2026  
**Team:** Novathon  
**Hackathon:** AI Student Dropout Risk Prediction Hackathon

---

## 1. Executive Summary

**DropGuard AI** is an intelligent early-warning platform that predicts student dropout risk using machine learning on academic, behavioral, personal, and engagement data. Unlike conventional dropout prediction tools that merely flag at-risk students, DropGuard AI goes further by providing **explainable AI insights**, **automated intervention workflows**, **student self-assessment**, and a **gamified engagement system** — creating a holistic ecosystem that doesn't just predict dropouts but actively prevents them.

---

## 2. Problem Statement

### The Crisis
Student dropout is one of the most pressing challenges in education. In India alone, higher education dropout rates exceed **40%** in many institutions. The consequences are severe:
- **Students** lose years of investment, face reduced career opportunities
- **Institutions** suffer reputational damage, lose funding, face regulatory scrutiny
- **Society** loses potential contributors to the workforce and economy

### Why Current Solutions Fail
- Most systems are **reactive** — they flag students only after it's too late
- Predictions are **black-box** — administrators don't know *why* a student is at risk
- No **actionable follow-up** — knowing a student is at risk doesn't mean anyone intervenes
- **Student voice is missing** — students themselves are not part of the loop

### Our Solution
DropGuard AI addresses all these gaps with a **proactive, explainable, and intervention-driven** platform.

---

## 3. Target Users

| User Persona | Role | Key Needs |
|:---|:---|:---|
| 🎓 **Institution Admin** | College/University administrator | Bird's-eye view of dropout risk across departments, actionable insights |
| 👨‍🏫 **Faculty / Mentor** | Teachers, HODs, counselors | Identify at-risk students in their classes, get intervention recommendations |
| 🧑‍🎓 **Student** | Undergraduate/postgraduate student | Self-assessment, support resources, engagement tracking |
| 📊 **Data Analyst** | Institutional research team | Advanced analytics, export reports, model performance monitoring |

---

## 4. Dataset Strategy

### 4.1 Primary Dataset

We will use a **synthetic + real-world hybrid** dataset approach:

#### Core Features (Columns)

| Category | Features | Data Type |
|:---|:---|:---|
| **Demographics** | `student_id`, `age`, `gender`, `socioeconomic_status`, `first_gen_student`, `urban_rural` | Categorical/Numeric |
| **Academic** | `gpa`, `cgpa`, `credits_completed`, `credits_required`, `failed_subjects_count`, `backlog_count`, `assignment_avg_score`, `exam_avg_score`, `semester` | Numeric |
| **Attendance & Behavior** | `attendance_pct`, `classes_missed_streak`, `disciplinary_incidents`, `late_submissions_pct` | Numeric |
| **Engagement** | `lms_logins_per_week`, `library_visits_per_month`, `portal_active_hours`, `forum_posts_count`, `club_memberships` | Numeric |
| **Financial** | `fee_payment_status`, `scholarship_holder`, `part_time_job`, `financial_aid_received` | Categorical/Boolean |
| **Personal** | `distance_from_campus_km`, `family_support_score`, `mental_health_flag`, `peer_relationship_score` | Numeric/Boolean |
| **Target** | `dropout` (1 = Dropped out, 0 = Continued) | Binary |

#### Dataset Size
- **Training:** ~10,000–15,000 student records (synthetic generation using realistic distributions)
- **Validation:** 20% holdout split
- **Test:** 10% holdout split

#### Data Sources
1. **UCI Machine Learning Repository** — "Student Performance" and "Student Dropout" datasets as base
2. **Kaggle** — "Predict Students' Dropout and Academic Success" dataset
3. **Synthetic augmentation** — Using Python `Faker` + statistical distributions to fill gaps and increase diversity

### 4.2 Data Preprocessing Pipeline

```
Raw Data → Missing Value Imputation (KNN Imputer)
         → Outlier Detection (IQR method)
         → Feature Encoding (One-Hot for categorical, Label for ordinal)
         → Feature Scaling (StandardScaler for numeric)
         → Class Balancing (SMOTE for minority class oversampling)
         → Feature Selection (Mutual Information + Recursive Feature Elimination)
         → Train/Validation/Test Split (70/20/10)
```

---

## 5. AI/ML Model Architecture

### 5.1 Model Pipeline

We use an **ensemble stacking** approach for maximum accuracy:

```
         ┌─────────────────┐
         │   Input Data     │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌─────────┐
│ XGBoost│  │ Random   │  │LightGBM │
│        │  │ Forest   │  │         │
└───┬────┘  └────┬─────┘  └────┬────┘
    │             │             │
    └─────────────┼─────────────┘
                  ▼
         ┌────────────────┐
         │  Meta-Learner  │
         │ (Logistic Reg) │
         └───────┬────────┘
                 ▼
         ┌────────────────┐
         │  Risk Score     │
         │  (0.0 – 1.0)   │
         └────────────────┘
```

### 5.2 Model Details

| Component | Technology | Why |
|:---|:---|:---|
| **Base Model 1** | XGBoost | Best-in-class for tabular data, handles missing values natively |
| **Base Model 2** | Random Forest | Robust, interpretable feature importances |
| **Base Model 3** | LightGBM | Fast training, great for large datasets |
| **Meta-Learner** | Logistic Regression | Combines base predictions, adds calibrated probabilities |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Feature-level explanations for each prediction |

### 5.3 Model Evaluation Metrics

| Metric | Target | Why It Matters |
|:---|:---|:---|
| **AUC-ROC** | ≥ 0.88 | Measures overall discrimination ability |
| **Recall (Sensitivity)** | ≥ 0.85 | Critical — we must catch at-risk students (minimize false negatives) |
| **Precision** | ≥ 0.75 | Avoid alarm fatigue from too many false positives |
| **F1-Score** | ≥ 0.80 | Balanced measure |
| **Accuracy** | ≥ 0.82 | General correctness |

### 5.4 Libraries & Tools

```
Python 3.10+
├── pandas, numpy          — Data manipulation
├── scikit-learn           — ML pipelines, preprocessing, metrics
├── xgboost                — XGBoost classifier
├── lightgbm               — LightGBM classifier
├── imbalanced-learn       — SMOTE oversampling
├── shap                   — Explainable AI
├── matplotlib, seaborn    — Visualization
├── joblib / pickle        — Model serialization
├── FastAPI                — Backend API server
└── streamlit / React      — Frontend dashboard
```

---

## 6. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FRONTEND                              │
│  ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────┐  │
│  │  Upload  │ │  Dashboard   │ │  Student  │ │  Chatbot  │  │
│  │  CSV     │ │  Analytics   │ │  Portal   │ │  Support  │  │
│  └────┬─────┘ └──────┬───────┘ └─────┬─────┘ └─────┬─────┘  │
│       └──────────────┼───────────────┼──────────────┘        │
└──────────────────────┼───────────────┼───────────────────────┘
                       │               │
                  ┌────▼───────────────▼────┐
                  │      FastAPI Backend     │
                  │  ┌─────────────────────┐ │
                  │  │  /predict           │ │
                  │  │  /explain           │ │
                  │  │  /interventions     │ │
                  │  │  /chatbot           │ │
                  │  │  /analytics         │ │
                  │  │  /student-self-assess│ │
                  │  └─────────────────────┘ │
                  └────────────┬─────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────┐  ┌──────────────┐
      │  ML Pipeline │ │ Database │  │  SHAP Engine │
      │  (Ensemble)  │ │ (SQLite/ │  │  (Explain)   │
      │              │ │ Supabase)│  │              │
      └──────────────┘ └──────────┘  └──────────────┘
```

---

## 7. Core Features (MVP)

### 7.1 CSV Upload & Batch Prediction
- Upload CSV with student data
- Automatic validation & schema checking
- Batch prediction with progress indicator
- Download results as CSV/PDF

### 7.2 Risk Dashboard
- **Risk Distribution Chart** — Pie/donut chart showing Low/Medium/High distribution
- **Student Risk Table** — Sortable, searchable table with risk scores
- **Department/Class Heatmap** — Visual breakdown by department/class
- **Trend Analysis** — Semester-over-semester risk trends

### 7.3 Individual Student Risk Card
- Detailed risk profile for each student
- Risk score with confidence interval
- Top 5 contributing factors (powered by SHAP)
- Historical risk trajectory

### 7.4 Intervention Recommendations
- Rule-based + AI-generated recommendations per risk level:
  - **High Risk (0.7–1.0):** Immediate counseling referral, mentor assignment, financial aid review
  - **Medium Risk (0.4–0.7):** Weekly check-ins, academic tutoring, peer mentoring
  - **Low Risk (0.0–0.4):** Engagement monitoring, optional workshops

---

## 8. 🌟 Unique Differentiating Features

These features set DropGuard AI apart from typical hackathon submissions:

### 8.1 🧠 Explainable AI Dashboard (SHAP Integration)

**Why it's unique:** Most hackathon projects show a risk score but never explain *why*. We provide visual SHAP waterfall charts for every student showing exactly which factors push their risk up or down.

- Per-student SHAP waterfall plots
- Global feature importance rankings
- "What-if" scenarios — e.g., "If this student's attendance improved by 10%, their risk would drop by X%"

### 8.2 🎮 Student Engagement Gamification Engine

**Why it's unique:** Instead of being passive subjects of prediction, students become active participants in reducing their own risk.

- **Risk Reduction Quests:** "Attend 5 classes in a row" → earn points
- **Achievement Badges:** "Perfect Attendance Week", "Assignment Streak", "Library Warrior"
- **Leaderboard:** Anonymous class-level engagement leaderboard
- **Reward Tiers:** Bronze → Silver → Gold → Platinum based on engagement score

### 8.3 🤖 AI Support Chatbot (LLM-Powered)

**Why it's unique:** A conversational interface for students to get personalized support.

- Powered by a lightweight LLM (OpenAI API / local Ollama model)
- Answers questions like:
  - "How can I improve my GPA?"
  - "I'm struggling with finances, what aid is available?"
  - "I feel overwhelmed, what should I do?"
- Escalation protocol: Auto-flags conversations indicating severe distress for counselor notification

### 8.4 📱 Student Self-Assessment Portal

**Why it's unique:** Gives students agency. They can self-report factors that data alone can't capture.

- Weekly 2-minute check-in survey:
  - "How are you feeling about your academics this week?" (1–5 scale)
  - "Are you facing any financial difficulties?" (Yes/No)
  - "How many hours did you sleep on average?" (Slider)
- Self-reported data is fed back into the model as supplementary features
- Students see their own risk trend over time (anonymized, non-alarming language)

### 8.5 🔔 Smart Alert System with Escalation Workflows

**Why it's unique:** Prediction without action is useless. This feature automates the intervention pipeline.

- **Threshold-based triggers:**
  - Risk crosses 0.7 → Auto-email to assigned mentor
  - Risk crosses 0.85 → Alert to HOD + counselor
  - Student misses 5+ classes consecutively → Attendance alert
- **Escalation tiers:** Mentor → Faculty → HOD → Dean (configurable)
- **Action tracking:** Log interventions and track outcomes

### 8.6 📊 Institutional Analytics & Reporting

**Why it's unique:** Provides admin-level intelligence, not just student-level predictions.

- **Department Risk Comparison:** Which departments have the highest dropout risk?
- **Cohort Analysis:** Compare risk profiles across admission years
- **Intervention Effectiveness Report:** Which interventions actually reduced dropout rates?
- **Exportable Board Reports:** PDF/PowerPoint-ready reports for management

### 8.7 🔮 Temporal Risk Forecasting

**Why it's unique:** Predicts not just *if* but *when* a student might drop out.

- Time-series analysis on longitudinal student data
- Predicts risk trajectory for the next 1–2 semesters
- Early warning for "silent drifters" — students whose risk is gradually increasing

---

## 9. Tech Stack

| Layer | Technology | Justification |
|:---|:---|:---|
| **Frontend** | React.js + Tailwind CSS (or Streamlit for rapid prototype) | Modern, responsive UI; Streamlit for quick MVP |
| **Backend** | FastAPI (Python) | Async, fast, auto-generates OpenAPI docs |
| **ML/AI** | Scikit-learn, XGBoost, LightGBM, SHAP | Industry-standard ML stack |
| **Database** | SQLite (local) / Supabase (cloud) | Easy setup for hackathon; Supabase for scaling |
| **Chatbot** | OpenAI API / Ollama (local) | Conversational AI support |
| **Visualization** | Plotly, Matplotlib, Chart.js | Interactive, publication-quality charts |
| **Deployment** | Docker + Render / Railway / Vercel | Quick cloud deployment |
| **Version Control** | Git + GitHub | Standard collaborative development |

---

## 10. API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/predict` | Upload CSV, return risk predictions |
| `POST` | `/api/predict/single` | Predict for a single student (JSON input) |
| `GET` | `/api/explain/{student_id}` | Get SHAP explanation for a student |
| `GET` | `/api/dashboard/stats` | Get aggregate dashboard statistics |
| `GET` | `/api/students` | List all students with risk scores |
| `GET` | `/api/students/{id}` | Get detailed student risk profile |
| `GET` | `/api/interventions/{risk_level}` | Get recommended interventions |
| `POST` | `/api/chatbot` | Send message, get AI chatbot response |
| `POST` | `/api/self-assess` | Submit student self-assessment |
| `GET` | `/api/analytics/department` | Department-level risk analytics |
| `GET` | `/api/analytics/trends` | Risk trends over time |
| `GET` | `/api/alerts` | Get active alerts |
| `PUT` | `/api/alerts/{id}/resolve` | Mark alert as resolved |
| `GET` | `/api/reports/export` | Export report as PDF |

---

## 11. UI/UX Design Specifications

### 11.1 Dashboard Layout

```
┌────────────────────────────────────────────────────┐
│  🛡️ DropGuard AI                    [Upload] [🔔]  │
├──────────┬─────────────────────────────────────────┤
│          │                                         │
│  📊      │   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
│ Dashboard│   │Total │ │High  │ │Medium│ │ Low  │  │
│          │   │ 1247 │ │ 156  │ │ 389  │ │ 702  │  │
│  👥      │   └──────┘ └──────┘ └──────┘ └──────┘  │
│ Students │                                         │
│          │   ┌────────────────┐ ┌────────────────┐  │
│  🧠      │   │  Risk Distrib. │ │  Dept Heatmap  │  │
│ Explain  │   │  [Donut Chart] │ │  [Heatmap]     │  │
│          │   └────────────────┘ └────────────────┘  │
│  🤖      │                                         │
│ Chatbot  │   ┌──────────────────────────────────┐  │
│          │   │  Student Risk Table               │  │
│  📱      │   │  Name | GPA | Attend | Risk | ▼  │  │
│ Self     │   │  ─────────────────────────────── │  │
│ Assess   │   │  John | 2.1 | 62%    | 0.89 | 🔴│  │
│          │   │  Sara | 3.5 | 91%    | 0.23 | 🟢│  │
│  ⚙️      │   │  Amit | 2.8 | 74%    | 0.65 | 🟡│  │
│ Settings │   └──────────────────────────────────┘  │
│          │                                         │
└──────────┴─────────────────────────────────────────┘
```

### 11.2 Color Scheme

| Risk Level | Color | Hex Code |
|:---|:---|:---|
| 🟢 Low (0.0–0.4) | Emerald Green | `#10B981` |
| 🟡 Medium (0.4–0.7) | Amber | `#F59E0B` |
| 🔴 High (0.7–1.0) | Rose Red | `#EF4444` |
| Background | Dark Slate | `#0F172A` |
| Card Surface | Dark Blue-Gray | `#1E293B` |
| Primary Accent | Electric Blue | `#3B82F6` |

---

## 12. Demo Flow (Hackathon Presentation)

### Presentation Script (5–7 minutes)

| Step | Duration | Action | What to Show |
|:---|:---|:---|:---|
| 1 | 30s | **Problem Hook** | Statistics on dropout rates, cost to students & institutions |
| 2 | 45s | **Solution Overview** | Architecture diagram, unique value proposition |
| 3 | 60s | **Live Demo: Upload** | Upload CSV → watch predictions populate in real-time |
| 4 | 60s | **Dashboard Tour** | Risk distribution, dept heatmap, student table |
| 5 | 60s | **SHAP Explanation** | Click a student → show SHAP waterfall → "what-if" scenario |
| 6 | 45s | **Chatbot Demo** | Ask chatbot a support question → show response |
| 7 | 30s | **Gamification** | Show student portal, badges, engagement quests |
| 8 | 30s | **Smart Alerts** | Show escalation workflow, email trigger |
| 9 | 30s | **Impact & Future** | Potential scale, real-world deployment, roadmap |

---

## 13. Development Phases

### Phase 1: Foundation (Day 1 — First 6 hours)
- [ ] Set up project structure (FastAPI + frontend)
- [ ] Prepare and preprocess dataset
- [ ] Build and train ensemble ML model
- [ ] Evaluate model performance (target: AUC ≥ 0.85)
- [ ] Build `/predict` and `/explain` API endpoints

### Phase 2: Core Dashboard (Day 1 — Next 6 hours)
- [ ] Build CSV upload functionality
- [ ] Create risk dashboard with charts
- [ ] Implement student list with filtering/sorting
- [ ] Add SHAP explanations per student
- [ ] Build intervention recommendations system

### Phase 3: Unique Features (Day 2 — First 6 hours)
- [ ] Implement AI chatbot integration
- [ ] Build student self-assessment portal
- [ ] Create smart alert system
- [ ] Add gamification engine (badges, quests)
- [ ] Build institutional analytics views

### Phase 4: Polish & Deploy (Day 2 — Last 6 hours)
- [ ] UI/UX polish — animations, responsiveness
- [ ] Docker containerization
- [ ] Deploy to cloud (Render / Railway)
- [ ] Prepare demo script and rehearse
- [ ] Create backup demo video

---

## 14. Risk Mitigation

| Risk | Mitigation |
|:---|:---|
| Model accuracy is low | Pre-train on proven Kaggle datasets; use ensemble for robustness |
| LLM chatbot fails | Fallback to rule-based FAQ responses |
| Time crunch | Prioritize MVP (upload → predict → dashboard) first |
| Data quality issues | Use synthetic data generator as backup |
| Deployment issues | Keep Streamlit as fallback frontend (single-file deployment) |
| Demo failure | Pre-record backup video of the full demo flow |

---

## 15. Judging Alignment

| Judging Criteria | How We Excel |
|:---|:---|
| **Innovation** | SHAP explainability, gamification, student self-assessment, temporal forecasting — none of these are standard |
| **Real-world Impact** | Direct application in any university; addresses a $100B+ global problem |
| **Working Prototype** | Full end-to-end pipeline: upload → predict → explain → intervene |
| **Data Usage & AI** | Ensemble stacking, SHAP explanations, class balancing — well beyond basic ML |
| **Clear UI/Dashboard** | Dark-themed professional dashboard with interactive charts and risk cards |

---

## 16. Success Metrics

| Metric | Definition | Target |
|:---|:---|:---|
| Model AUC-ROC | Discrimination ability | ≥ 0.88 |
| Prediction Latency | Time to predict 1000 students | < 3 seconds |
| Dashboard Load Time | Time to render full dashboard | < 2 seconds |
| Feature Coverage | Unique features implemented | ≥ 5 of 7 planned |
| Demo Smoothness | Successful end-to-end demo without failures | 100% |

---

## 17. Future Roadmap (Post-Hackathon)

1. **Real-time LMS Integration** — Connect with Moodle/Google Classroom APIs for live data
2. **Mobile App** — React Native app for students with push notifications
3. **Multi-language Support** — Hindi, regional languages for Indian institutions
4. **Federated Learning** — Cross-institution model training without sharing raw data
5. **Government Integration** — AISHE (All India Survey on Higher Education) data pipeline
6. **Parent/Guardian Portal** — Controlled risk visibility for parents

---

> **"Don't just predict dropouts. Prevent them."**  
> — DropGuard AI

---

*This document is a living PRD and will be updated as development progresses during the hackathon.*
