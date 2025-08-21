# Customer Churn Prediction (MLOps Project)

This project is a machine learning pipeline to predict customer churn based on structured telco service data. It explores various ML models and covers key stages of the MLOps lifecycle, including preprocessing, training, evaluation, and (optionally) deployment via Docker.

---

## Project Structure

The repository is organized as follows:

📁 Project Structure

dataset/               # Cleaned datasets (e.g., both_serv.csv, only_int.csv)
models/                # Trained models and logs
----- xgb/             # XGBoost-related models and logs
----- adab/            # AdaBoost-related models and logs
----- rf/              # RandomForest-related models and logs
deployment/            # Docker-related files for API deployment
----- app.py           # FastAPI inference API
----- Dockerfile       # Docker image setup
----- requirements.txt # Python dependencies
client.py              # Example script for calling the API
README.md              # Project documentation

---

## ⚙️ Models Explored

- ✅ XGBoost (`xgboost.XGBClassifier`)
- ✅ AdaBoost (`AdaBoostClassifier` with `DecisionTreeClassifier` base)
- ⏳ RandomForest (Initial baseline)
- 🔬 GridSearchCV used for hyperparameter tuning (ROC-AUC as metric)

---

## 🧪 Training Pipeline

1. **Preprocessing:**
   - One-hot encoding for categorical features
   - Binary conversion for Yes/No features
   - Handled mixed binary + magnitude features

2. **Model Training:**
   - Split: 80% train / 20% test
   - GridSearchCV for hyperparameter search
   - Metrics: ROC AUC, Classification Report, Confusion Matrix

3. **Model Saving:**
   - Saved with `joblib`
   - JSON logs created for every run
   - Model files too large for GitHub are excluded

---

## 🚀 Deployment

- Inference API built with **FastAPI**
- Dockerized for consistent deployment
- Use `client.py` to test prediction endpoints

### Example Docker Commands:

```bash
# Build image
docker build -t churn-api .

# Run container
docker run -d -p 8000:8000 churn-api
