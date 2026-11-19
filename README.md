# ChurnIQ — Customer Churn Prediction

ChurnIQ is a machine learning-powered customer intelligence application that estimates customer churn risk and supports customer retention decisions.

## Features

- Customer churn risk prediction
- Bulk prediction through CSV upload
- Customer risk segmentation
- Interactive business dashboard
- Model performance evaluation
- Confusion matrix and threshold comparison
- Feature importance analysis
- Admin section with password protection

## Machine Learning

- Algorithm: Random Forest Classifier
- Input features: 19
- Production decision threshold: 0.30
- Evaluation: Stratified 80/20 holdout split

## Model Evaluation

| Metric | Score |
|---|---:|
| Accuracy | 74.1% |
| Precision | 50.8% |
| Recall | 77.8% |
| F1 Score | 61.5% |

These results are based on the current Telco Customer Churn dataset and may differ on other customer populations.

## Technology Stack

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Run Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Configure your local Streamlit secrets.
5. Run:

   streamlit run app.py

## Security

- Never commit passwords, API keys, or secrets.
- Keep `.streamlit/secrets.toml` private.
- Do not upload private customer information.
- Dataset and model files are excluded until their source and distribution permissions are verified.

## Limitations

Predictions are estimates, not guarantees. Model performance depends on dataset quality and customer population. Validate the model on recent business data before making operational decisions.

## Project Status

Portfolio project — deployment preparation in progress.
