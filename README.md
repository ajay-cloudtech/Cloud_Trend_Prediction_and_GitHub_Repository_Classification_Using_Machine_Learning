# Cloud Trend Prediction and GitHub Repository Classification Using Machine Learning

This project applies machine learning to analyze cloud technology trends and classify GitHub repositories based on their cloud stack. It uses time-series forecasting to track adoption patterns and supervised learning to tag repositories with relevant technologies.

## Project Objectives

- **Predict Cloud Adoption Trends:** Forecast the growth of cloud platforms (AWS, Azure, GCP), tools (Docker, Kubernetes, Terraform), and languages (Python, JavaScript, etc.) using the Prophet model.
- **Classify GitHub Repositories:** Use XGBoost to assign technology tags to cloud-related repositories.
- **Data Collection & Preprocessing:** Gather public GitHub data using the GitHub API, clean and engineer features (TF-IDF, metadata).
- **Cloud Deployment:** Train and deploy models on AWS EC2 with a Flask web app for real-time predictions.
- **Evaluation:** Measure model accuracy, precision, recall, F1-score, and inference speed.

## Technologies Used

- **Languages & Frameworks:** Python, Flask
- **ML Models:** Prophet (for forecasting), XGBoost (for classification)
- **Cloud Services:** AWS EC2, GitHub API

## Methodology

1. **Data Collection:** Extracted repositories using GitHub API with keywords related to cloud platforms and technologies.
2. **Data Cleaning:** Removed duplicates, filtered by size, and enriched missing descriptions using README files.
3. **Feature Engineering:** Applied TF-IDF on textual features and normalized metadata for ML compatibility.
4. **Trend Prediction:** Used Prophet to forecast repository growth trends over time.
5. **Classification:** Trained XGBoost model to label repositories with relevant cloud technologies.
6. **Web Deployment:** Built a Flask web app hosted on AWS EC2 for real-time interaction with both models.

## Evaluation Summary

- **Prophet Model:** Effective for small datasets, but accuracy drops with scale. Suitable for initial trend insights.
- **XGBoost Model:** Achieved 99.44% accuracy and 99.71% F1-score. High throughput and reliable classification even on large datasets.

<img width="609" alt="image" src="https://github.com/user-attachments/assets/540bd247-c0d3-4b6c-a238-407ba624e8cf" />

![image](https://github.com/user-attachments/assets/394a584d-73bc-4285-9903-235e3cfc0c43)

<img width="363" alt="image" src="https://github.com/user-attachments/assets/f4d93c4c-3f4a-40b3-b127-cf49fb28abf8" />

![image](https://github.com/user-attachments/assets/774515c6-997a-43d7-a9c0-fca833fe08a0)

![image](https://github.com/user-attachments/assets/87df2029-e813-4d58-b9fe-1880b32284c5)



