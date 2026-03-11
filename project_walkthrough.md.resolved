# 🎓 AI Student Performance Prediction - Project Walkthrough

## Project Successfully Built! ✅

The complete **AI-Based Student Performance Prediction and Academic Risk Analysis System** is now running.

![App Homepage](C:\Users\DHARUN TEJA\.gemini\antigravity\brain\9c0f7d31-2372-4595-863a-a8a8da91cc9c\streamlit_home_page_1772908491199.png)

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive web dashboard |
| **Backend** | Python + Scikit-Learn | ML models & data processing |
| **Database** | Supabase (PostgreSQL) + Local JSON fallback | Store predictions & metrics |
| **Visualization** | Plotly | Interactive charts & graphs |

---

## 📁 Project Structure

```
Mini-Project_3rd Year/
├── app.py                          ← Main entry point (Home page)
├── config.py                       ← All settings & configuration
├── requirements.txt                ← Python dependencies
├── .env                            ← Supabase credentials (template)
├── .gitignore                      ← Git ignore rules
├── models/
│   ├── train_model.py              ← ML training pipeline (3 algorithms)
│   ├── predictor.py                ← Prediction logic
│   └── saved_models/
│       └── best_model.pkl          ← Trained Random Forest model
├── utils/
│   ├── data_preprocessing.py       ← Data cleaning & feature engineering
│   ├── risk_advisor.py             ← Academic Risk Advisor Agent (10 rules)
│   └── db_manager.py               ← Database operations
├── pages/
│   ├── 1_📊_Dashboard.py           ← Analytics dashboard
│   ├── 2_🔮_Predict.py             ← Individual prediction
│   ├── 3_📁_Batch_Upload.py        ← Batch CSV upload
│   ├── 4_🤖_Model_Training.py      ← Model training & comparison
│   └── 5_📜_History.py             ← Prediction history
└── datasets/
    ├── Student_Performance(1).csv   ← 10,000 records
    ├── StudentsPerformance.csv      ← 1,000 records
    └── student/
        └── student-mat.csv          ← Primary dataset (395 records)
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML model (optional - already done)
python models/train_model.py

# 3. Run the Streamlit app
python -m streamlit run app.py
```

> [!TIP]
> The app is already running at **http://localhost:8501**

---

## 📄 Pages Overview

### 1. 🏠 Home Page ([app.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/app.py))
- Welcome screen with gradient hero section
- System status indicators (Dataset, Model, Risk Advisor)
- Quick statistics (395 students, 67.1% pass rate)
- Feature overview grid

### 2. 📊 Dashboard ([Dashboard.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/pages/1_%F0%9F%93%8A_Dashboard.py))
- 10+ interactive Plotly charts
- Grade distribution, Pass/Fail pie chart
- Risk level distribution, Grade trends (G1→G2→G3)
- Absences vs Performance scatter, Study time impact
- Feature correlation heatmap
- Multi-dataset analysis (3 datasets)

### 3. 🔮 Predict ([Predict.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/pages/2_%F0%9F%94%AE_Predict.py))
- Comprehensive student input form (30+ fields)
- Real-time AI prediction (Pass/Fail)
- Risk level classification with gauge chart
- AI Risk Advisor recommendations with priority badges

### 4. 📁 Batch Upload ([Batch_Upload.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/pages/3_%F0%9F%93%81_Batch_Upload.py))
- Upload CSV with multiple students
- Batch prediction processing
- Results visualization (charts + table)
- Download results as CSV

### 5. 🤖 Model Training ([Model_Training.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/pages/4_%F0%9F%A4%96_Model_Training.py))
- Train 3 ML algorithms with progress bar
- Side-by-side performance comparison
- Confusion matrices for each model
- Feature importance visualization (Random Forest)

### 6. 📜 History ([History.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/pages/5_%F0%9F%93%9C_History.py))
- All past predictions with filters
- Summary metrics & charts
- Export to CSV
- Clear history option

---

## 🤖 ML Algorithms Used

| Algorithm | Description | Advantage |
|-----------|-------------|-----------|
| **Logistic Regression** | Linear classifier | Fast, interpretable |
| **Decision Tree** | Rule-based | Easy to understand |
| **Random Forest** | Ensemble of trees | Highest accuracy ✅ |

> [!NOTE]
> The system automatically selects the best model (Random Forest won with the primary dataset).

---

## 🧠 Academic Risk Advisor Agent

The AI Risk Advisor uses **10 rule-based checks** to generate personalized recommendations:

1. **Attendance** — Flags high absence counts
2. **Study Habits** — Checks weekly study time
3. **Grade Trends** — Detects declining G1→G2 patterns
4. **Internal Marks** — Flags below-threshold grades
5. **Past Failures** — Considers failure history
6. **Social Behavior** — Monitors going out & alcohol patterns
7. **Support System** — Checks school & family support
8. **Health** — Flags poor health status
9. **Higher Education Aspiration** — Checks motivation
10. **Internet Access** — Identifies resource gaps

Each recommendation has a **priority level**: 🚨 Critical, ⚠️ Important, 💡 Suggested

---

## 🗄️ Database Setup (Optional)

The app works without Supabase using **local JSON storage**. To enable Supabase:

1. Create a free account at [supabase.com](https://supabase.com)
2. Create a new project
3. Create tables using the SQL in [.env](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/.env)
4. Add credentials to [.env](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/.env) file

---

## 📝 Key Files Explained

| File | Purpose | Comments |
|------|---------|----------|
| [config.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/config.py) | All constants & settings | Risk thresholds, paths, colors |
| [data_preprocessing.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/utils/data_preprocessing.py) | Data cleaning pipeline | Load, encode, scale, split |
| [train_model.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/models/train_model.py) | ML training | 3 algorithms, evaluation, save |
| [predictor.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/models/predictor.py) | Prediction logic | Single & batch predictions |
| [risk_advisor.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/utils/risk_advisor.py) | Recommendation engine | 10 academic rules |
| [db_manager.py](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/utils/db_manager.py) | Database operations | Supabase + JSON fallback |
