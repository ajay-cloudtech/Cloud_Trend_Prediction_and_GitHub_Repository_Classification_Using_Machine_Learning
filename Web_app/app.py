import pandas as pd
from prophet import Prophet
from flask import Flask, jsonify, request
import requests
import joblib
import numpy as np
import base64
from datetime import datetime
from sklearn.metrics import mean_absolute_error, accuracy_score
import time

# Initialize Flask application
app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv("github_repos_filtered.csv")
df['created_at'] = pd.to_datetime(df['created_at'], format='mixed')
full_size = len(df)


# Load pre-trained models and encoders
xgb_model = joblib.load("xgboost_model.pkl")
language_encoder = joblib.load("language_encoder.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to forecast trends using Prophet
def forecast_trend(df, col_name, label, periods=365):
    df_trend = df[df[col_name] == 1].groupby('created_at').size().reset_index(name='y')
    df_trend.rename(columns={'created_at': 'ds'}, inplace=True)

    if df_trend.empty:
        return {"error": f"No data available for {label}"}, 400
    
    model = Prophet()
    model.fit(df_trend)

    historical = df_trend.groupby(df_trend['ds'].dt.to_period('M'))['y'].sum().reset_index()
    historical = historical.rename(columns={'ds': 'month'})
    last_historical_date = df_trend['ds'].max()

    future = model.make_future_dataframe(periods=periods, include_history=False)
    forecast = model.predict(future)

    forecast['month'] = forecast['ds'].dt.to_period('M')
    monthly_forecast = forecast.groupby('month')[['yhat', 'yhat_lower', 'yhat_upper']].sum().reset_index()

    result = {
        "category": label,
        "historical": [{"month": str(row['month']), "actual_value": int(row['y'])} for _, row in historical.iterrows()],
        "forecast": [
            {"month": str(row['month']), "predicted_value": round(row['yhat']), "lower_bound": round(row['yhat_lower']), "upper_bound": round(row['yhat_upper'])}
            for _, row in monthly_forecast.iterrows()
        ]
    }
    return result, 200

# Function to measure Prophet model performance
def measure_prophet_performance(df_subset, col_name="is_cloud_project", label="Overall Cloud Projects", periods=365):
    start_time = time.time()
    df_trend = df_subset[df_subset[col_name] == 1].groupby('created_at').size().reset_index(name='y')
    df_trend.rename(columns={'created_at': 'ds'}, inplace=True)
    if df_trend.empty:
        return None
    
    model = Prophet(
        changepoint_prior_scale=0.1,
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(df_trend)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    end_time = time.time()
    
    latency = end_time - start_time
    throughput = len(forecast) / latency
    dataset_size = len(df_subset)
    historical_forecast = forecast[forecast['ds'].isin(df_trend['ds'])]['yhat']
    mae = mean_absolute_error(df_trend['y'], historical_forecast)
    
    return {
        "label": label,
        "latency": round(latency, 2),
        "throughput": round(throughput, 2),
        "dataset_size": dataset_size,
        "mae": round(mae, 2)
    }

# Function to measure XGBoost model performance
def measure_xgboost_performance(df_subset, label):
    start_time = time.time()
    
    descriptions = df_subset['description'].fillna('')
    description_tfidf = vectorizer.transform(descriptions)
    features = pd.DataFrame(description_tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(523)])
    
    languages = ['JavaScript', 'Python', 'TypeScript', 'Jupyter Notebook', 'Java', 'C#', 'Go', 'PHP', 'C++', 'Vue', 'Bicep', 'Kotlin', 'Dart', 'Rust', 'C', 'Ruby']
    for lang in languages:
        features[lang] = (df_subset['language'] == lang).astype(int)
    
    features['year'] = 2025
    features['month'] = 3
    features['day'] = 22
    features['week'] = 12
    features['size'] = df_subset['size']
    features['is_cloud_project'] = df_subset['description'].str.lower().str.contains('cloud').fillna(False).astype(int)
    features['language'] = df_subset['language'].apply(lambda x: language_encoder.transform([x])[0] if x in language_encoder.classes_ else -1)
    
    expected_columns = ['size', 'language'] + languages + ['year', 'month', 'day', 'week', 'is_cloud_project'] + [f'tfidf_{i}' for i in range(523)]
    features = features[expected_columns]
    
    predictions = xgb_model.predict(features)
    if predictions.ndim == 2:
        predictions = (predictions > 0.5).astype(int)
    
    end_time = time.time()
    
    latency = end_time - start_time
    throughput = len(df_subset) / latency
    dataset_size = len(df_subset)
    true_labels = df_subset[['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'DevOps']].values
    hamming_score = accuracy_score(true_labels, predictions, normalize=True)
    
    return {
        "label": label,
        "latency": round(latency, 2),
        "throughput": round(throughput, 2),
        "dataset_size": dataset_size,
        "hamming_score": round(hamming_score, 2)
    }

# Function to predict repository tags using XGBoost
def predict_tags(owner, repo_name):
    repo_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    repo_data = requests.get(repo_url).json()

    if "message" in repo_data and repo_data["message"] == "Not Found":
        return {"error": "Repository not found"}, 404

    repo_title = repo_data['name']
    repo_description = repo_data['description'] if repo_data['description'] else ''
    repo_language = repo_data['language'] if repo_data['language'] else 'Unknown'
    repo_size = repo_data['size']

    readme_url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/README.md"
    readme_data = requests.get(readme_url).json()
    content = base64.b64decode(readme_data['content']).decode('utf-8') if readme_data.get('content') else ''

    current_date = datetime.now()
    year, month, day, week = current_date.year, current_date.month, current_date.day, current_date.isocalendar()[1]

    description_tfidf = vectorizer.transform([repo_description])
    features = pd.DataFrame(description_tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(523)])

    repo_language_encoded = -1 if repo_language not in language_encoder.classes_ else language_encoder.transform([repo_language])[0]
    languages = ['JavaScript', 'Python', 'TypeScript', 'Jupyter Notebook', 'Java', 'C#', 'Go', 'PHP', 'C++', 'Vue', 'Bicep', 'Kotlin', 'Dart', 'Rust', 'C', 'Ruby']
    for lang in languages:
        features[lang] = 1 if lang == repo_language else 0
    features['year'] = year
    features['month'] = month
    features['day'] = day
    features['week'] = week
    features['size'] = repo_size
    features['is_cloud_project'] = 1 if 'cloud' in repo_description.lower() else 0
    features['language'] = repo_language_encoded

    expected_columns = ['size', 'language'] + languages + ['year', 'month', 'day', 'week', 'is_cloud_project'] + [f'tfidf_{i}' for i in range(523)]
    features = features[expected_columns]

    try:
        predictions = xgb_model.predict(features)
        if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
            predictions = (predictions > 0.5).astype(int)
        
        tags = ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Terraform', 'DevOps']
        predicted_tags = {tags[i]: int(predictions[0][i]) for i in range(len(tags))}
        
        result = {
            "repo": f"{owner}/{repo_name}",
            "tags": [tag for tag, pred in predicted_tags.items() if pred == 1]
        }
        return result, 200
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}, 500

# API endpoint to forecast trends
@app.route('/forecast', methods=['POST'])
def predict_trend():
    data = request.get_json()
    if not data or 'category' not in data:
        return jsonify({"error": "Missing 'category' in request"}), 400
    
    category = data['category']
    periods = data.get('periods', 365)

    category_map = {
        "Overall Cloud Projects": "is_cloud_project",
        "AWS Projects": "AWS",
        "Azure Projects": "Azure",
        "GCP Projects": "GCP",
        "Docker Projects": "Docker",
        "Kubernetes Projects": "Kubernetes",
        "Terraform Projects": "Terraform",
        "JavaScript Projects": "JavaScript",
        "Python Projects": "Python",
        "TypeScript Projects": "TypeScript"
    }

    if category not in category_map:
        return jsonify({"error": f"Invalid category. Choose from {list(category_map.keys())}"}), 400

    col_name = category_map[category]
    result, status = forecast_trend(df, col_name, category, periods)
    return jsonify(result), status

# API endpoint to get Prophet performance metrics
@app.route('/prophet-performance', methods=['GET'])
def get_prophet_performance():
    sizes = [100, 1000, 10000, 25000, 50000, full_size]
    results = []

    for size in sizes:
        if size <= full_size:
            sampled_df = df.sample(n=size, random_state=42)
        else:
            repeat_factor = (size // full_size) + 1
            sampled_df = pd.concat([df] * repeat_factor, ignore_index=True).iloc[:size]
            sampled_df['created_at'] = pd.to_datetime(sampled_df['created_at'], format='mixed')
        result = measure_prophet_performance(sampled_df, "is_cloud_project", f"Overall Cloud Projects ({size} repos)")
        if result:
            results.append(result)

    return jsonify({"performance": results}), 200

# API endpoint to get XGBoost performance metrics
@app.route('/xgboost-performance', methods=['GET'])
def get_xgboost_performance():
    sizes = [100, 1000, 10000, 25000, 50000, full_size]
    results = []

    for size in sizes:
        if size <= full_size:
            sampled_df = df.sample(n=size, random_state=42)
        else:
            repeat_factor = (size // full_size) + 1
            sampled_df = pd.concat([df] * repeat_factor, ignore_index=True).iloc[:size]
        result = measure_xgboost_performance(sampled_df, f"XGBoost Inference ({size} repos)")
        if result:
            results.append(result)

    return jsonify({"performance": results}), 200

# API endpoint to predict repository tags
@app.route('/predict-tags', methods=['POST'])
def predict_repo_tags():
    data = request.get_json()
    if not data or 'owner' not in data or 'repo_name' not in data:
        return jsonify({"error": "Missing 'owner' or 'repo_name' in request"}), 400
    
    owner = data['owner']
    repo_name = data['repo_name']
    result, status = predict_tags(owner, repo_name)
    return jsonify(result), status

@app.route('/')
def serve_frontend():
    with open("index.html", "r") as f:
        return f.read()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)