# Ad Revenue Optimization System Using Machine Learning

## Overview
A machine learning project that predicts advertising revenue and recommends the most profitable ad strategies by analyzing historical campaign data. Built with Python and deployed as an interactive web dashboard using Streamlit.

---

## Key Features
- **Revenue Prediction** — Trains a Random Forest Regressor to predict ad revenue based on platform, ad type, target audience, impressions, and ad duration.
- **Model Evaluation** — Displays R² Score and Mean Absolute Error so you can measure how well the model performs.
- **Strategy Optimization** — Compares all platform × ad type combinations to identify which strategy yields the highest predicted revenue.
- **Top Ad Recommendations** — Ranks all ads in the dataset by predicted revenue and surfaces the top 5.
- **Interactive Dashboard** — Upload your own CSV and explore results instantly via a Streamlit web app.

---

## Tech Stack
| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas | Data loading and manipulation |
| scikit-learn | ML model, preprocessing, and evaluation |
| Streamlit | Web dashboard |
| matplotlib | Charts and visualizations |

---

## Dataset (`ad_data.csv`)
10,000 rows of advertising campaign data with the following columns:

| Column | Type | Description |
|---|---|---|
| platform | Categorical | TV, Social Media, Radio, Online, Billboard, Print Media |
| ad_type | Categorical | Product-Based / Service-Based |
| category | Categorical | Industry category (e.g. Finance, Education, Health) |
| target_audience | Categorical | Professionals, Teens, Adults, Students, Families, Seniors |
| Impressions | Numerical | Total ad views |
| ad_duration | Numerical | Duration of ad (days) |
| Location | Categorical | City where the ad ran |
| Revenue | Numerical | **Target variable** — actual revenue generated |

> **Note:** Columns like Clicks, Conversions, and ad_spend are excluded as features because they are only available *after* a campaign runs, which would cause data leakage.

---

## How It Works

```
1. Upload CSV  →  2. Data Cleaning  →  3. Preprocessing
     ↓
4. Train RandomForestRegressor (80/20 split)
     ↓
5. Evaluate with R² Score + MAE  →  6. Optimize & Recommend
```

### Step-by-step
1. **Load data** — Read the CSV and preview the first few rows.
2. **Clean data** — Drop post-campaign columns to prevent data leakage; remove nulls.
3. **Preprocess** — `StandardScaler` for numbers, `OneHotEncoder` for text columns.
4. **Train model** — `RandomForestRegressor` with `max_depth=10` to avoid overfitting.
5. **Evaluate** — R² and MAE on the held-out 20% test set, plus an Actual vs Predicted scatter plot.
6. **Optimize** — Test all 12 platform × ad type combinations using a representative sample row.
7. **Recommend** — Rank all ads by predicted revenue and display the top 5.

---

## How to Run

```bash
# 1. Install dependencies
pip install streamlit pandas scikit-learn matplotlib

# 2. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser and upload `ad_data.csv`.

---

## Sample Output

**Model Performance**
- R² Score: ~0.85 (model explains ~85% of revenue variance)
- MAE: ~$1,200 (average prediction error)

**Top Strategy Example**
- Social Media — Product-Based → Highest predicted revenue

---

## What I Learned
- How to build a complete ML pipeline using `scikit-learn` (preprocessing → model → evaluation)
- Why data leakage matters and how to prevent it
- How to deploy a machine learning model as an interactive web app with Streamlit
- How to use `itertools` to compare combinations and find the optimal strategy

---

## Project Structure
```
ad-optimization/
│
├── app.py              # Main Streamlit application
├── ad_data.csv         # Dataset (10,000 ad records)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```
