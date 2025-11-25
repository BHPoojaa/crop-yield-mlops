import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

DATA_PATH = "data/crop_yield.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_yield_model.pkl")

def load_or_create_data():
    if os.path.exists(DATA_PATH):
        print(f"📂 Loading data from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print(f"⚠️ {DATA_PATH} not found. Creating synthetic dataset...")
        np.random.seed(42)
        n_samples = 200
        rainfall = np.random.uniform(400, 1200, n_samples)
        temp = np.random.uniform(18, 35, n_samples)
        fertilizer = np.random.uniform(50, 250, n_samples)
        soil_ph = np.random.uniform(5.5, 7.5, n_samples)
        yield_tons = (
            0.003 * rainfall +
            0.05 * fertilizer -
            0.1 * (temp - 25) +
            0.3 * (7 - np.abs(soil_ph - 7)) +
            np.random.normal(0, 0.3, n_samples)
        )
        df = pd.DataFrame({
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "fertilizer_kg_per_ha": fertilizer,
            "soil_ph": soil_ph,
            "yield_tons_per_ha": yield_tons
        })
        df.to_csv(DATA_PATH, index=False)
    return df

def train_model(df):
    feature_cols = ["rainfall_mm", "avg_temp_c", "fertilizer_kg_per_ha", "soil_ph"]
    target_col = "yield_tons_per_ha"
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"📏 MAE: {mae:.3f}")
    print(f"📈 R²: {r2:.3f}")
    return model, mae, r2

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

def log_with_mlflow(model, mae, r2):
    mlflow.set_experiment("crop_yield_prediction")
    with mlflow.start_run():
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, artifact_path="model")

def main():
    df = load_or_create_data()
    model, mae, r2 = train_model(df)
    save_model(model)
    log_with_mlflow(model, mae, r2)

if __name__ == "__main__":
    main()
