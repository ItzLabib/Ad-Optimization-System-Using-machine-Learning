# Ad Optimization System for Maximizing ROI

## **Overview**
This project leverages machine learning to optimize advertising strategies by predicting revenue and identifying the most profitable combinations of ad platforms and ad types. By analyzing historical ad performance data, the system helps businesses allocate resources effectively to maximize ROI.

---

## **Key Features**
1. **Revenue Prediction**: Uses a Random Forest Regressor to predict ad revenue based on features like platform, ad type, impressions, and ad duration.
2. **Strategy Optimization**: Evaluates all possible combinations of platforms and ad types to identify the most profitable ad strategies.
3. **Top Recommendations**: Ranks and visualizes the top-performing ad strategies to aid decision-making.

---

## **Technologies Used**
- **Python**: For data processing and model development.
- **Libraries**:
  - `pandas`: Data manipulation.
  - `scikit-learn`: Machine learning and pipeline management.
  - `matplotlib`: Visualization of results.
  - `itertools`: Combination generation for optimization.

---

## **Workflow**
1. **Data Preprocessing**:
   - Handles missing values.
   - Encodes categorical variables and scales numerical features.
2. **Model Training**:
   - Trains a Random Forest Regressor to predict revenue based on historical data.
3. **Optimization**:
   - Uses all combinations of `platform` and `ad_type` to find strategies maximizing predicted revenue.
4. **Visualization and Recommendations**:
   - Displays the top 5 ad strategies with predicted revenue.

---

## **How It Works**
1. Load the dataset and preprocess it (e.g., encoding and scaling).
2. Train the Random Forest Regressor on the dataset to predict `Revenue`.
3. Use optimization logic to generate combinations of `platform` and `ad_type`.
4. Predict revenue for each combination and rank them by performance.
5. Visualize the top strategies and provide actionable insights.

---

## **Example Output**
- **Top 5 Optimized Strategies**:
![image](https://github.com/user-attachments/assets/a83ed038-1615-4e0f-8a4d-1f57c44b4e48)

