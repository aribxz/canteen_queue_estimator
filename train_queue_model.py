from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Waiting_Time"
FEATURES = ["Number_of_People", "Counters_Open", "Avg_Service_Time", "Time_of_Day", "Queue_Type"]
NUMERIC = ["Number_of_People", "Counters_Open", "Avg_Service_Time", "Time_of_Day"]
CATEGORICAL = ["Queue_Type"]


def path():
    base = Path(__file__).resolve().parent
    csv_path = base / "queue.csv"
    model_path = base / "canteen_queue_model.pkl"
    plots_dir = base / "plots"
    plots_dir.mkdir(exist_ok=True)
    return base, csv_path, model_path, plots_dir


def data_details(csv_path):
    df = pd.read_csv(csv_path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    print("Loading queue.csv ...")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values: {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print(df[FEATURES + [TARGET]].describe().round(2).to_string())
    print(f"Queue_Type values: {df['Queue_Type'].unique().tolist()}")
    X = df[FEATURES]
    y = df[TARGET]
    return df, X, y


def pipeline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Split: {len(X_train)} train / {len(X_test)} test (random_state=42)")
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)
    print("Random Forest trained (n_estimators=200).")
    pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, pred


def statistics(model, X_test, y_test, pred, model_path):
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    print(f"Test results: MAE={mae:.2f} min, RMSE={rmse:.2f} min, R2={r2:.4f}")
    sample = pd.DataFrame({
        "Actual (min)": y_test.reset_index(drop=True)[:8],
        "Predicted (min)": np.round(pred[:8], 2),
    })
    sample["Error (min)"] = (sample["Actual (min)"] - sample["Predicted (min)"]).abs().round(2)
    print("Actual vs predicted (first 8 test rows):")
    print(sample.to_string(index=False))
    joblib.dump(model, model_path)
    print(f"Saved pipeline to {model_path.name}")
    loaded = joblib.load(model_path)
    demo = pd.DataFrame([{"Number_of_People": 40, "Counters_Open": 3,
                          "Avg_Service_Time": 5, "Time_of_Day": 13,
                          "Queue_Type": "Office"}])
    print(f"Demo prediction (40 people, 3 counters): {float(loaded.predict(demo)[0]):.2f} minutes")


def plots(y_test, pred, plots_dir):
    y_true = y_test.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(y_true, pred, s=10, alpha=0.4)
    ax.plot([0, 1200], [0, 1200], linestyle="--", linewidth=1)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1200)
    ax.set_xlabel("Actual Waiting_Time (minutes)")
    ax.set_ylabel("Predicted Waiting_Time (minutes)")
    ax.set_title("Actual vs Predicted waiting time (zoomed 0-1200 min)\nDashed line = perfect prediction")
    fig.tight_layout()
    fig.savefig(plots_dir / "model_actual_vs_predicted_zoomed.png", bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_true, pred - y_true, s=10, alpha=0.4)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Actual Waiting_Time (minutes)")
    ax.set_ylabel("Residual = Predicted - Actual (minutes)")
    ax.set_title("Prediction errors (residuals)\nPoints should scatter evenly around 0")
    fig.tight_layout()
    fig.savefig(plots_dir / "model_residuals.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved 2 plots: model_actual_vs_predicted_zoomed.png, model_residuals.png")
    print("DONE.")


base, csv_path, model_path, plots_dir = path()
df, X, y = data_details(csv_path)
model, X_train, X_test, y_train, y_test, pred = pipeline(X, y)
statistics(model, X_test, y_test, pred, model_path)
plots(y_test, pred, plots_dir)
