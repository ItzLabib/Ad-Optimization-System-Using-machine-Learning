import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="Ad Optimization Dashboard", layout="wide")
st.title("💰 Ad Revenue Optimization Dashboard")
st.markdown("Upload your dataset and explore the most profitable ad strategies.")

# ------------------------------
# Upload Data
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload your ad_data.csv", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(data.head())

    # ------------------------------
    # Clean Data
    # ------------------------------
    data = data.dropna()
    data = data.drop(columns=['CTR'], errors='ignore')  # ignore if not present

    # ------------------------------
    # Feature / Target Split
    # ------------------------------
    if 'Revenue' not in data.columns:
        st.error("⚠️ 'Revenue' column is missing from your dataset.")
    else:
        X = data.drop(columns=['Revenue'])
        y = data['Revenue']

        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['number']).columns

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        model = RandomForestRegressor(random_state=42)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # ------------------------------
        # Train-Test Split
        # ------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        st.success("✅ Model trained successfully!")

        # ------------------------------
        # Optimization Function
        # ------------------------------
        def optimize_ads(data, model):
            platforms = data['platform'].unique()
            ad_types = data['ad_type'].unique()
            combinations = list(itertools.product(platforms, ad_types))
            results = []

            for combo in combinations:
                test_data = data.copy()
                test_data['platform'] = combo[0]
                test_data['ad_type'] = combo[1]
                test_data = test_data.drop(columns='Revenue', errors='ignore')
                predicted = model.predict(test_data)
                total_predicted_revenue = predicted.sum()
                results.append((combo, total_predicted_revenue))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:5]

        # ------------------------------
        # Recommendation Function
        # ------------------------------
        def recommend_ads(data, model, top_n=5):
            predictions = model.predict(data.drop(columns='Revenue'))
            recommendations = pd.DataFrame({
                'Ad ID': data['Ad ID'],
                'Product/Service Name': data['Product/Service Name'],
                'Category': data['category'],
                'Target Audience': data['target_audience'],
                'Platform': data['platform'],
                'Impressions': data['Impressions'],
                'Predicted_Revenue': predictions
            })
            return recommendations.sort_values(by='Predicted_Revenue', ascending=False).head(top_n)

        # ------------------------------
        # Buttons
        # ------------------------------
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 Find Top Ad Strategies"):
                optimized_ads = optimize_ads(data, pipeline)
                strategies = [f"{combo[0]} - {combo[1]}" for combo, _ in optimized_ads]
                revenues = [rev for _, rev in optimized_ads]

                st.subheader("Top 5 Optimized Ad Strategies")
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.barh(strategies, revenues)
                ax.set_xlabel("Predicted Total Revenue")
                ax.invert_yaxis()
                st.pyplot(fig)

        with col2:
            if st.button("💡 Show Top 5 Recommended Ads"):
                recommended = recommend_ads(data, pipeline)
                st.subheader("Top 5 Recommended Ads by Predicted Revenue")
                st.dataframe(recommended)
else:
    st.info("👆 Upload your CSV file to get started.")
