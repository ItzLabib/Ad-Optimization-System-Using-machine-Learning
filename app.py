# ============================================================
# Ad Revenue Optimization — Streamlit Dashboard
# This file is the LIVE DEMO only.
# For step-by-step analysis, see ad_optimization.ipynb
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(page_title="Ad Optimization Dashboard", layout="wide")
st.title("💰 Ad Revenue Optimization Dashboard")
st.markdown("Upload your dataset to instantly train the model and explore the best ad strategies.")

# ----------------------------------------------------------
# Upload
# ----------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload ad_data.csv", type="csv")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    # ---- Clean ----
    cols_to_drop = ['Ad ID', 'Product/Service Name', 'Date',
                    'Clicks', 'Conversions', 'ad_spend', 'Location', 'CTR']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    data = data.dropna()

    # ---- Features ----
    X = data.drop(columns=['Revenue'])
    y = data['Revenue']
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols   = X.select_dtypes(include=['number']).columns.tolist()

    # ---- Pipeline ----
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(),                        numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'),  categorical_cols)
    ])
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100, max_depth=8,
            min_samples_leaf=10, random_state=42))
    ])

    # ---- Train ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # ---- Metrics ----
    y_pred = pipeline.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # ==============================================================
    # Layout: 3 tabs
    # ==============================================================
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Model Performance", "🚀 Optimization"])

    # ------ TAB 1: Data Overview ------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(data.head(10))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Ads",     f"{len(data):,}")
        col2.metric("Platforms",     data['platform'].nunique())
        col3.metric("Categories",    data['category'].nunique())
        col4.metric("Avg Revenue",   f"${data['Revenue'].mean():,.0f}")

        st.subheader("Average Revenue by Platform")
        avg_platform = data.groupby('platform')['Revenue'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(avg_platform.index, avg_platform.values, color='steelblue')
        ax.bar_label(bars, fmt='$%.0f', fontsize=8, padding=3)
        ax.set_ylabel("Avg Revenue ($)")
        ax.set_title("Average Revenue by Platform")
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Average Revenue by Category")
        avg_cat = data.groupby('category')['Revenue'].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        bars2 = ax2.bar(avg_cat.index, avg_cat.values, color='coral')
        ax2.bar_label(bars2, fmt='$%.0f', fontsize=8, padding=3)
        ax2.set_ylabel("Avg Revenue ($)")
        ax2.tick_params(axis='x', rotation=20)
        plt.tight_layout()
        st.pyplot(fig2)

    # ------ TAB 2: Model Performance ------
    with tab2:
        st.subheader("Model Performance")
        c1, c2 = st.columns(2)
        c1.metric("R² Score",            f"{r2:.2f}",    help="Closer to 1.0 is better")
        c2.metric("Mean Absolute Error", f"${mae:,.0f}", help="Average dollar error in predictions")

        fig3, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Actual vs Predicted
        axes[0].scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=15, edgecolors='none')
        axes[0].plot([y_test.min(), y_test.max()],
                     [y_test.min(), y_test.max()], 'r--', linewidth=1.5, label='Perfect prediction')
        axes[0].set_xlabel("Actual Revenue ($)")
        axes[0].set_ylabel("Predicted Revenue ($)")
        axes[0].set_title("Actual vs Predicted Revenue")
        axes[0].legend()

        # Feature importance (top 10)
        ohe_features = (pipeline.named_steps['preprocessor']
                                .named_transformers_['cat']
                                .get_feature_names_out(categorical_cols).tolist())
        all_features = numerical_cols + ohe_features
        importances  = pipeline.named_steps['model'].feature_importances_
        feat_df = (pd.DataFrame({'Feature': all_features, 'Importance': importances})
                   .sort_values('Importance', ascending=False).head(10))
        axes[1].barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1], color='steelblue')
        axes[1].set_xlabel("Importance Score")
        axes[1].set_title("Top 10 Feature Importances")

        plt.tight_layout()
        st.pyplot(fig3)

    # ------ TAB 3: Optimization ------
    with tab3:
        st.subheader("Best Platform × Ad Type Combinations")

        if st.button("🔍 Run Optimization"):
            combos  = list(itertools.product(data['platform'].unique(), data['ad_type'].unique()))
            results = []
            for platform, ad_type in combos:
                sample = {col: data[col].median() for col in numerical_cols}
                sample['platform'] = platform
                sample['ad_type']  = ad_type
                for col in categorical_cols:
                    if col not in ['platform', 'ad_type']:
                        sample[col] = data[col].mode()[0]
                pred = pipeline.predict(pd.DataFrame([sample]))[0]
                results.append({'Platform': platform, 'Ad Type': ad_type,
                                'Predicted Revenue ($)': round(pred, 2)})

            res_df = pd.DataFrame(results).sort_values(
                'Predicted Revenue ($)', ascending=False).reset_index(drop=True)

            top5 = res_df.head(5).copy()
            top5['Label'] = top5['Platform'] + ' + ' + top5['Ad Type']

            fig4, ax4 = plt.subplots(figsize=(9, 4))
            bars4 = ax4.barh(top5['Label'], top5['Predicted Revenue ($)'], color='steelblue')
            ax4.bar_label(bars4, fmt='$%.0f', padding=5, fontsize=10)
            ax4.set_xlabel("Predicted Revenue ($)")
            ax4.set_title("Top 5 Ad Strategies by Predicted Revenue")
            ax4.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig4)

            st.markdown("**All combinations ranked:**")
            st.dataframe(res_df)

        st.subheader("💡 Top 5 Recommended Ads")
        if st.button("Show Top 5 Ads"):
            preds = pipeline.predict(X)
            rec   = data.copy()
            rec['Predicted Revenue ($)'] = preds.round(2)
            disp  = ['platform', 'ad_type', 'category', 'target_audience',
                     'Impressions', 'ad_duration', 'Predicted Revenue ($)']
            st.dataframe(rec[[c for c in disp if c in rec.columns]]
                         .sort_values('Predicted Revenue ($)', ascending=False)
                         .head(5).reset_index(drop=True))

else:
    st.info("👆 Upload your ad_data.csv file to get started.")
