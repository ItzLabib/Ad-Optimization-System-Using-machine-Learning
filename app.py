from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the models and feature columns
conv_model = joblib.load('conversion_model.pkl')
rev_model = joblib.load('revenue_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    data = request.json
    input_df = pd.DataFrame([data])
    
    # Ensure columns match the model's expected features
    input_df = pd.get_dummies(input_df).reindex(columns=feature_columns, fill_value=0)
    
    # Predict conversions and revenue
    predicted_conversions = conv_model.predict(input_df)[0]
    predicted_revenue = rev_model.predict(input_df)[0]
    
    # Return predictions
    return jsonify({
        'Predicted_Conversions': predicted_conversions,
        'Predicted_Revenue': predicted_revenue
    })

if __name__ == '__main__':
    app.run(debug=True)
