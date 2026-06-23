"""
app.py — Dashboard interactivo Heart Disease (PC2 — Parte 3)

Panel A: Análisis de Datos (EDA interactivo con filtros y KPIs)
Panel B: Análisis Predictivo (formulario + inferencia con el mejor modelo)

Para correr localmente:
    streamlit run app.py

Para desplegar en Streamlit Community Cloud:
    1) Sube este archivo + requirements.txt + core.py (+ opcionalmente data/processed.cleveland.data)
       a un repositorio de GitHub.
    2) Entra a https://share.streamlit.io, conecta el repo y selecciona app.py como archivo principal.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import core

st.set_page_config(
    page_title="Heart Disease — Dashboard",
    page_icon="❤️",
    layout="wide",
)

LABELS = {
    "sex": {0: "Mujer", 1: "Hombre"},
    "cp": {1: "Angina típica", 2: "Angina atípica", 3: "Dolor no anginoso", 4: "Asintomático"},
    "fbs": {0: "≤ 120 mg/dl", 1: "> 120 mg/dl"},
    "restecg": {0: "Normal", 1: "Anomalía ST-T", 2: "Hipertrofia ventricular izq."},
    "exang": {0: "No", 1: "Sí"},
    "slope": {1: "Ascendente", 2: "Plana", 3: "Descendente"},
    "ca": {0.0: "0 vasos", 1.0: "1 vaso", 2.0: "2 vasos", 3.0: "3 vasos"},
    "thal": {3.0: "Normal", 6.0: "Defecto fijo", 7.0: "Defecto reversible"},
    "target": {0: "Sin presencia de enfermedad", 1: "Con presencia/riesgo de enfermedad"},
}


@st.cache_data(show_spinner="Cargando dataset Heart Disease...")
def get_data(uploaded_bytes=None):
    return core.load_data(uploaded_bytes)


@st.cache_resource(show_spinner="Entrenando y comparando 5 modelos...")
def get_models(df: pd.DataFrame):
    return core.train_and_evaluate(df)


# ---------------------------------------------------------------------------
# Sidebar: fuente de datos
# ---------------------------------------------------------------------------
st.sidebar.title("❤️ Heart Disease Dashboard")
st.sidebar.caption("PC2 — Agentes Inteligentes — USIL 2026-1")

st.sidebar.markdown("---")
st.sidebar.subheader("Fuente de datos")
uploaded = st.sidebar.file_uploader(
    "Sube processed.cleveland.data o un CSV limpio (opcional)",
    type=["data", "csv", "txt"],
)

try:
    df = get_data(uploaded)
except Exception as e:
    st.error(
        "No se pudo cargar el dataset automáticamente (sin conexión a UCI y sin "
        f"archivo local). Sube el archivo manualmente en la barra lateral. Detalle: {e}"
    )
    st.stop()

pipelines, metrics_df, best_model_name, (X_test, y_test) = get_models(df)
best_pipeline = pipelines[best_model_name]

tab_a, tab_b = st.tabs(["📊 Panel A — Análisis de Datos", "🔮 Panel B — Análisis Predictivo"])

# ===========================================================================
# PANEL A — ANÁLISIS DE DATOS
# ===========================================================================
with tab_a:
    st.header("📊 Panel A — Análisis de Datos")
    st.caption("Explora el dataset Heart Disease — Cleveland de forma interactiva.")

    with st.sidebar:
        st.markdown("---")
        st.subheader("Filtros — Panel A")
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        rango_edad = st.slider("Rango de edad", age_min, age_max, (age_min, age_max))

        sexo_opt = st.selectbox("Sexo", ["Todos", "Hombre", "Mujer"])
        cp_opt = st.selectbox(
            "Tipo de dolor de pecho (cp)",
            ["Todos"] + list(LABELS["cp"].values()),
        )
        target_opt = st.selectbox(
            "Diagnóstico",
            ["Todos", "Sin presencia de enfermedad", "Con presencia/riesgo de enfermedad"],
        )

    df_f = df[(df["age"] >= rango_edad[0]) & (df["age"] <= rango_edad[1])]
    if sexo_opt != "Todos":
        codigo_sexo = 1 if sexo_opt == "Hombre" else 0
        df_f = df_f[df_f["sex"] == codigo_sexo]
    if cp_opt != "Todos":
        codigo_cp = [k for k, v in LABELS["cp"].items() if v == cp_opt][0]
        df_f = df_f[df_f["cp"] == codigo_cp]
    if target_opt != "Todos":
        codigo_target = 1 if "riesgo" in target_opt else 0
        df_f = df_f[df_f["target"] == codigo_target]

    if len(df_f) == 0:
        st.warning("No hay registros que cumplan con los filtros seleccionados.")
        st.stop()

    df_plot = df_f.copy()
    df_plot["Diagnóstico"] = df_plot["target"].map(LABELS["target"])

    # --- KPIs ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros filtrados", f"{len(df_f)} / {len(df)}")
    c2.metric("% con riesgo cardíaco", f"{(df_f['target'].mean() * 100):.1f}%")
    c3.metric("Edad promedio", f"{df_f['age'].mean():.1f} años")

    st.markdown("---")

    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.histogram(
            df_plot, x="age", color="Diagnóstico", nbins=20, barmode="overlay",
            title="Distribución de edad por diagnóstico",
            labels={"age": "Edad"},
            color_discrete_map={
                "Sin presencia de enfermedad": "#2E86AB",
                "Con presencia/riesgo de enfermedad": "#E63946",
            },
        )
        st.plotly_chart(fig1, use_container_width=True)

    with g2:
        fig2 = px.scatter(
            df_plot, x="chol", y="thalach", color="Diagnóstico",
            title="Colesterol vs. frecuencia cardíaca máxima",
            labels={"chol": "Colesterol (mg/dl)", "thalach": "Frec. cardíaca máx."},
            color_discrete_map={
                "Sin presencia de enfermedad": "#2E86AB",
                "Con presencia/riesgo de enfermedad": "#E63946",
            },
        )
        st.plotly_chart(fig2, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        target_counts = df_plot["Diagnóstico"].value_counts().reset_index()
        target_counts.columns = ["Diagnóstico", "Casos"]
        fig3 = px.bar(
            target_counts, x="Diagnóstico", y="Casos", color="Diagnóstico",
            title="Distribución de la variable objetivo (subconjunto filtrado)",
            color_discrete_map={
                "Sin presencia de enfermedad": "#2E86AB",
                "Con presencia/riesgo de enfermedad": "#E63946",
            },
        )
        st.plotly_chart(fig3, use_container_width=True)

    with g4:
        corr = df_f[core.NUMERIC_FEATURES + ["target"]].corr(numeric_only=True)
        fig4 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Mapa de calor de correlaciones (variables numéricas)",
        )
        st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Ver tabla de datos filtrados"):
        st.dataframe(df_f, use_container_width=True)

# ===========================================================================
# PANEL B — ANÁLISIS PREDICTIVO
# ===========================================================================
with tab_b:
    st.header("🔮 Panel B — Análisis Predictivo")
    st.caption(
        f"Modelo activo: **{best_model_name}** "
        f"(F1-Score = {metrics_df.set_index('Modelo').loc[best_model_name, 'F1-Score']:.3f} en el conjunto de prueba)"
    )

    with st.form("formulario_prediccion"):
        st.subheader("Ingresa los datos clínicos del paciente")

        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Edad", 18, 100, 55)
            trestbps = st.slider("Presión arterial en reposo (mmHg)", 80, 220, 130)
            chol = st.slider("Colesterol sérico (mg/dl)", 100, 600, 240)
            thalach = st.slider("Frecuencia cardíaca máxima alcanzada", 60, 220, 150)
        with col2:
            sex = st.selectbox("Sexo", list(LABELS["sex"].values()))
            cp = st.selectbox("Tipo de dolor de pecho", list(LABELS["cp"].values()))
            fbs = st.selectbox("Azúcar en sangre en ayunas", list(LABELS["fbs"].values()))
            restecg = st.selectbox("Resultado electrocardiográfico", list(LABELS["restecg"].values()))
        with col3:
            exang = st.selectbox("Angina inducida por ejercicio", list(LABELS["exang"].values()))
            oldpeak = st.slider("Oldpeak (depresión ST por ejercicio)", 0.0, 6.5, 1.0, step=0.1)
            slope = st.selectbox("Pendiente del segmento ST", list(LABELS["slope"].values()))
            ca = st.selectbox("Número de vasos principales (ca)", list(LABELS["ca"].values()))
            thal = st.selectbox("Tipo de defecto (thal)", list(LABELS["thal"].values()))

        submitted = st.form_submit_button("🔍 Predecir", use_container_width=True)

    if submitted:
        inv = {k: {v2: k2 for k2, v2 in v.items()} for k, v in LABELS.items()}
        input_dict = {
            "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach,
            "oldpeak": oldpeak,
            "sex": inv["sex"][sex], "cp": inv["cp"][cp], "fbs": inv["fbs"][fbs],
            "restecg": inv["restecg"][restecg], "exang": inv["exang"][exang],
            "slope": inv["slope"][slope], "ca": inv["ca"][ca], "thal": inv["thal"][thal],
        }

        pred, prob_riesgo, prob_sin_riesgo = core.predict_single(best_pipeline, input_dict)

        st.markdown("---")
        r1, r2 = st.columns([1, 1])
        with r1:
            if pred == 1:
                st.error(f"### ⚠️ {LABELS['target'][1]}")
                st.progress(prob_riesgo)
                st.metric("Probabilidad de riesgo", f"{prob_riesgo * 100:.1f}%")
            else:
                st.success(f"### ✅ {LABELS['target'][0]}")
                st.progress(prob_sin_riesgo)
                st.metric("Probabilidad de ausencia de enfermedad", f"{prob_sin_riesgo * 100:.1f}%")

        with r2:
            st.info(
                "**¿Qué significa este resultado?**\n\n"
                + (
                    "El modelo estima que, según los valores clínicos ingresados, el paciente "
                    "presenta un patrón similar al de personas con diagnóstico positivo de "
                    "enfermedad cardíaca en el dataset de entrenamiento. Esto **no es un "
                    "diagnóstico médico**: es una estimación estadística que debe ser "
                    "validada por un profesional de la salud con exámenes clínicos "
                    "adicionales."
                    if pred == 1 else
                    "El modelo estima que, según los valores clínicos ingresados, el patrón "
                    "del paciente es más similar al de personas sin diagnóstico de "
                    "enfermedad cardíaca en el dataset de entrenamiento. Esto **no descarta** "
                    "una condición cardíaca: ante cualquier síntoma, se recomienda evaluación "
                    "médica profesional."
                )
            )

    with st.expander("📈 Ver comparación de los 5 modelos entrenados"):
        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.4f}", "F1-Score": "{:.4f}",
                "Precision": "{:.4f}", "Recall": "{:.4f}",
            }),
            use_container_width=True,
        )
        st.caption(
            f"Se seleccionó **{best_model_name}** como modelo final por su buen equilibrio "
            "entre F1-Score, Accuracy y Recall de la clase positiva, además de su mayor "
            "interpretabilidad frente a modelos como SVM o Random Forest en un contexto de salud."
        )

st.markdown("---")
st.caption(
    "Dashboard desarrollado con Streamlit · Dataset: Heart Disease — Cleveland (UCI Machine "
    "Learning Repository) · Práctica Calificada 2 — Agentes Inteligentes — USIL 2026-1"
)
