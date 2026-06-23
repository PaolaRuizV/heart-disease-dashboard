# heart-disease-dashboard
# Heart Disease — Dashboard Interactivo

Dashboard de Machine Learning para la predicción de presencia/ausencia de enfermedad
cardíaca, desarrollado para la **Práctica Calificada 2 — Agentes Inteligentes (USIL, 2026-1)**.

🔗 **App: <https://heart-disease-dashboard-proyect.streamlit.app/>**

## Descripción

La aplicación tiene dos paneles:

- **Panel A — Análisis de Datos:** explorador interactivo del dataset con filtros
  (edad, sexo, tipo de dolor de pecho, diagnóstico), KPIs y 4 visualizaciones (Plotly).
- **Panel B — Análisis Predictivo:** formulario con las 13 variables clínicas del
  paciente; devuelve la clase predicha, la probabilidad asociada y una explicación
  contextual del resultado.

## Dataset

**Heart Disease — Cleveland**, UCI Machine Learning Repository.
Fuente: <https://archive.ics.uci.edu/dataset/45/heart+disease>
Archivo utilizado: `processed.cleveland.data` (303 registros, 13 variables clínicas + 1
variable objetivo, binarizada como 0 = sin enfermedad / 1 = con presencia o riesgo).

## Modelos

Se entrenaron y compararon 5 modelos de clasificación (scikit-learn) con el mismo
pipeline de preprocesamiento (imputación, `StandardScaler` para variables numéricas,
`OneHotEncoder` para categóricas, partición 80/20 estratificada, `random_state=42`):

| Modelo | Accuracy | F1-Score | Precision | Recall |
|---|---|---|---|---|
| Regresión Logística   | 0.8852 | 0.8814 | 0.8387 | 0.9286 |
| SVM | 0.8852 | 0.8814 | 0.8387 | 0.9286 |
| KNN | 0.8852 | 0.8772 | 0.8621 | 0.8929 |
| Random Forest | 0.8689 | 0.8621 | 0.8333 | 0.8929 |
| Árbol de Decisión | 0.7377 | 0.7241 | 0.7000 | 0.7500 |

Modelo final: **Regresión Logística**, por su buen equilibrio entre F1-Score y Recall,
y por ser más interpretable en un contexto clínico.

## Estructura del repositorio

```
.
├── app.py                          # Aplicación Streamlit (interfaz)
├── core.py                         # Lógica de datos y modelos (sin UI)
├── requirements.txt                # Dependencias
└── data/
    └── processed.cleveland.data    # Dataset (opcional, recomendado)
```

## Cómo correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue
<https://heart-disease-dashboard-proyect.streamlit.app/>


Desplegado en [Streamlit Community Cloud]().
