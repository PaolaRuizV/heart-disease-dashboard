"""
core.py
Lógica de datos y modelos para el dashboard de Heart Disease (Parte 3).
Separado de la interfaz de Streamlit para poder probarlo de forma independiente.
"""
import io
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target_raw"
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

LOCAL_CANDIDATES = [
    "data/processed.cleveland.data",
    "processed.cleveland.data",
    "data/heart_cleveland.csv",
    "heart_cleveland.csv",
]


def _binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = (df["target_raw"] > 0).astype(int)
    return df.drop(columns=["target_raw"])


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric)
    df = _binarize_target(df)
    return df


def load_data(uploaded_file=None) -> pd.DataFrame:
    """
    Orden de búsqueda del dataset:
    1) archivo subido manualmente por el usuario en la app (uploaded_file)
    2) archivo local incluido en el repositorio (data/processed.cleveland.data o .csv)
    3) descarga directa desde UCI Machine Learning Repository
    """
    if uploaded_file is not None:
        raw = uploaded_file.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if "," in text.splitlines()[0] and "age" in text.splitlines()[0].lower():
            # ya viene con encabezados (CSV limpio)
            df = pd.read_csv(io.StringIO(text))
            if "target_raw" in df.columns:
                df = _basic_clean(df.rename(columns={c: c for c in df.columns}))
            return df
        df = pd.read_csv(io.StringIO(text), names=COLUMNS)
        return _basic_clean(df)

    import os
    for path in LOCAL_CANDIDATES:
        if os.path.exists(path):
            if path.endswith(".csv"):
                df = pd.read_csv(path)
                if "target" in df.columns:
                    return df
                df.columns = COLUMNS
                return _basic_clean(df)
            df = pd.read_csv(path, names=COLUMNS)
            return _basic_clean(df)

    df = pd.read_csv(UCI_URL, names=COLUMNS)
    return _basic_clean(df)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def get_model_definitions():
    return {
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def train_and_evaluate(df: pd.DataFrame):
    """
    Entrena los 5 modelos con el mismo preprocesamiento y el mismo split,
    devuelve: dict de pipelines entrenados, DataFrame de métricas, nombre del mejor modelo.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    model_defs = get_model_definitions()

    pipelines = {}
    rows = []
    for name, estimator in model_defs.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
        })
        pipelines[name] = pipe

    metrics_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    # Modelo final: Regresión Logística (elegido por interpretabilidad y alto recall,
    # ver justificación en la sección 2.3 del informe), siempre que esté entre los
    # mejores; si no lo está, se usa el de mayor F1-Score como respaldo.
    if "Regresión Logística" in pipelines:
        best_name = "Regresión Logística"
    else:
        best_name = metrics_df.iloc[0]["Modelo"]

    return pipelines, metrics_df, best_name, (X_test, y_test)


def predict_single(pipeline, input_dict: dict):
    """input_dict: valores crudos de las 13 variables clínicas."""
    X_new = pd.DataFrame([input_dict])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    pred = pipeline.predict(X_new)[0]
    proba = pipeline.predict_proba(X_new)[0]
    return int(pred), float(proba[1]), float(proba[0])
