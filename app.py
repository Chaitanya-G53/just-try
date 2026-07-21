"""
Dynamic Model Inspection & Multi-Interface Inference Engine.

Production-grade Streamlit application that inspects machine learning objects (.pkl),
automatically infers intelligent feature input types (categorical, boolean, numeric, text),
detects domain themes (Healthcare, Finance, Real Estate, Education, Cyber, etc.), and renders
optimized interfaces and tailored chart palettes.

Authors: Senior AI Engineering & UX Team
Version: 2.0.0
PEP8 Compliant.
"""

import io
import logging
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ModelInspectorApp")

# ==============================================================================
# 1. PAGE SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Universal Model Inspector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 2. INSPECTION ENGINE & METADATA EXTRACTION
# ==============================================================================
class ModelInspector:
    """Provides methods to safely load, inspect, and extract capabilities from serialized objects."""

    @staticmethod
    def load_object(file_bytes: bytes) -> Tuple[Optional[Any], Optional[str]]:
        """Safely unpickles the uploaded file bytes."""
        try:
            obj = pickle.loads(file_bytes)
            return obj, None
        except Exception as e:
            logger.error(f"Failed to unpickle object: {str(e)}")
            return None, f"Failed to parse pickle file: {str(e)}"

    @classmethod
    def analyze_object(cls, obj: Any) -> Dict[str, Any]:
        """Deeply inspects an unpickled object to extract architectural details and categorical features."""
        analysis = {
            "type": type(obj).__name__,
            "module": type(obj).__module__,
            "is_pipeline": False,
            "pipeline_steps": [],
            "estimator_type": "Unknown",
            "feature_names": [],
            "n_features": 0,
            "params": {},
            "classes": [],
            "coefficients": None,
            "intercept": None,
            "categorical_maps": {},
            "sklearn_version": getattr(obj, "_sklearn_version", "Unknown"),
        }

        # Check for Pipeline
        if hasattr(obj, "steps"):
            analysis["is_pipeline"] = True
            analysis["pipeline_steps"] = [step[0] for step in obj.steps]
            final_estimator = obj.steps[-1][1]

            # Inspect pipeline encoders for categories
            for name, step in obj.steps:
                if hasattr(step, "categories_") and hasattr(step, "feature_names_in_"):
                    for f_name, cats in zip(step.feature_names_in_, step.categories_):
                        analysis["categorical_maps"][f_name] = list(cats)
        else:
            final_estimator = obj

        # Extract Estimator Type
        if hasattr(final_estimator, "_estimator_type"):
            analysis["estimator_type"] = getattr(final_estimator, "_estimator_type")
        elif "Regressor" in type(final_estimator).__name__:
            analysis["estimator_type"] = "regressor"
        elif "Classifier" in type(final_estimator).__name__:
            analysis["estimator_type"] = "classifier"
        elif "Cluster" in type(final_estimator).__name__:
            analysis["estimator_type"] = "clusterer"

        # Feature Names
        if hasattr(obj, "feature_names_in_"):
            analysis["feature_names"] = list(getattr(obj, "feature_names_in_"))
        elif hasattr(final_estimator, "feature_names_in_"):
            analysis["feature_names"] = list(getattr(final_estimator, "feature_names_in_"))

        # Number of Features
        if hasattr(obj, "n_features_in_"):
            analysis["n_features"] = getattr(obj, "n_features_in_")
        elif hasattr(final_estimator, "n_features_in_"):
            analysis["n_features"] = getattr(final_estimator, "n_features_in_")
        elif analysis["feature_names"]:
            analysis["n_features"] = len(analysis["feature_names"])

        if analysis["n_features"] > 0 and not analysis["feature_names"]:
            analysis["feature_names"] = [f"Feature_{i+1}" for i in range(analysis["n_features"])]

        # Classification classes
        if hasattr(final_estimator, "classes_"):
            analysis["classes"] = list(getattr(final_estimator, "classes_"))

        # Coefficients & Intercept
        if hasattr(final_estimator, "coef_"):
            analysis["coefficients"] = getattr(final_estimator, "coef_")
        if hasattr(final_estimator, "intercept_"):
            analysis["intercept"] = getattr(final_estimator, "intercept_")

        # Hyperparameters
        if hasattr(obj, "get_params"):
            try:
                analysis["params"] = obj.get_params(deep=False)
            except Exception:
                analysis["params"] = {}

        return analysis


# ==============================================================================
# 3. DYNAMIC MULTI-INTERFACE THEME ENGINE
# ==============================================================================
class ThemeEngine:
    """Infers domain themes and chart color palettes based on feature heuristics and metadata."""

    THEMES = {
        "education": {
            "name": "Education Analytics",
            "title": "Academic Performance & Student Portal",
            "icon": "🎓",
            "primary": "#4C6EF5",
            "secondary": "#15AABF",
            "bg_gradient": "linear-gradient(135deg, #101426 0%, #172554 100%)",
            "card_bg": "rgba(255, 255, 255, 0.06)",
            "accent": "#38BDF8",
            "chart_colors": ["#38BDF8", "#818CF8", "#F472B6", "#34D399"],
            "plotly_scale": "Viridis",
        },
        "real_estate": {
            "name": "Real Estate",
            "title": "Property Valuation & Real Estate Hub",
            "icon": "🏠",
            "primary": "#E05A47",
            "secondary": "#2C3E50",
            "bg_gradient": "linear-gradient(135deg, #1A1C23 0%, #2D1B1E 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#FF7A00",
            "chart_colors": ["#E05A47", "#FF7A00", "#4AB8B8", "#6C5CE7"],
            "plotly_scale": "Tealrose",
        },
        "finance": {
            "name": "Financial Services",
            "title": "Financial Risk & Portfolio Portal",
            "icon": "📈",
            "primary": "#00B894",
            "secondary": "#0984E3",
            "bg_gradient": "linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#55E6C1",
            "chart_colors": ["#00B894", "#0984E3", "#FDCB6E", "#6C5CE7"],
            "plotly_scale": "Emerald",
        },
        "healthcare": {
            "name": "Healthcare & Life Sciences",
            "title": "Medical & Diagnostic Dashboard",
            "icon": "🏥",
            "primary": "#00CEC9",
            "secondary": "#0984E3",
            "bg_gradient": "linear-gradient(135deg, #134E5E 0%, #1A365D 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#81ECEC",
            "chart_colors": ["#00CEC9", "#0984E3", "#A29BFE", "#FD79A8"],
            "plotly_scale": "Ice",
        },
        "cybersecurity": {
            "name": "Security & Fraud",
            "title": "Cyber Threat & Anomaly Monitoring",
            "icon": "🛡️",
            "primary": "#FF2A6D",
            "secondary": "#05D9E8",
            "bg_gradient": "linear-gradient(135deg, #0D0221 0%, #02010A 100%)",
            "card_bg": "rgba(255, 255, 255, 0.07)",
            "accent": "#05D9E8",
            "chart_colors": ["#FF2A6D", "#05D9E8", "#D1F7FF", "#FF6584"],
            "plotly_scale": "Plasma",
        },
        "nlp": {
            "name": "NLP & Text Processing",
            "title": "NLP Analysis & Sentiment Studio",
            "icon": "💬",
            "primary": "#A29BFE",
            "secondary": "#6C5CE7",
            "bg_gradient": "linear-gradient(135deg, #2D132C 0%, #801336 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#FD79A8",
            "chart_colors": ["#A29BFE", "#FD79A8", "#FFEAA7", "#55E6C1"],
            "plotly_scale": "Magma",
        },
        "default": {
            "name": "General Analytics",
            "title": "Universal Model Intelligence Platform",
            "icon": "⚡",
            "primary": "#6C5CE7",
            "secondary": "#00CEC9",
            "bg_gradient": "linear-gradient(135deg, #141E30 0%, #243B55 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#A29BFE",
            "chart_colors": ["#6C5CE7", "#00CEC9", "#FF7675", "#FDCB6E"],
            "plotly_scale": "Deep",
        },
    }

    KEYWORD_MAPPINGS = {
        "education": [
            "student", "school", "grade", "study", "education", "exam", "score",
            "attendance", "gpa", "course", "teacher", "parent_education", "hours",
            "extracurricular"
        ],
        "real_estate": [
            "house", "housing", "living area", "bedroom", "bathroom", "floor",
            "renovation", "lot", "price", "sqft", "zipcode", "property", "built year"
        ],
        "finance": [
            "credit", "balance", "transaction", "amount", "fraud", "loan",
            "income", "debt", "payment", "account", "stock", "portfolio"
        ],
        "healthcare": [
            "patient", "blood", "age", "diagnosis", "glucose", "bmi", "heart",
            "hospital", "disease", "medical", "symptom"
        ],
        "cybersecurity": [
            "ip", "packet", "attack", "threat", "anomaly", "port", "login",
            "flag", "security", "bytes"
        ],
        "nlp": [
            "text", "sentiment", "embedding", "word", "tfidf", "corpus",
            "token", "phrase", "review", "comment"
        ],
    }

    @classmethod
    def infer_theme(cls, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Scores theme keywords against extracted feature names and attributes."""
        scores = {key: 0 for key in cls.KEYWORD_MAPPINGS.keys()}

        search_corpus = " ".join([
            analysis.get("type", ""),
            analysis.get("module", ""),
            " ".join(analysis.get("feature_names", [])),
            " ".join([str(c) for c in analysis.get("classes", [])]),
        ]).lower()

        for theme_key, keywords in cls.KEYWORD_MAPPINGS.items():
            for kw in keywords:
                if kw in search_corpus:
                    scores[theme_key] += 1

        best_theme_key = max(scores, key=scores.get)

        if scores[best_theme_key] == 0:
            best_theme_key = "default"

        logger.info(f"Inferred Theme: {best_theme_key} (Scores: {scores})")
        return cls.THEMES[best_theme_key]


# ==============================================================================
# 4. CUSTOM STYLING & GLASSMORPHISM
# ==============================================================================
def inject_custom_styles(theme: Dict[str, Any]):
    """Injects responsive dark-mode styling into Streamlit."""
    custom_css = f"""
    <style>
        .stApp {{
            background: {theme['bg_gradient']};
            color: #E2E8F0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        div[data-testid="stMetric"] {{
            background: {theme['card_bg']};
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 18px 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.2s ease-in-out;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-3px);
            border-color: {theme['accent']};
        }}
        
        h1, h2, h3, h4 {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }}
        
        .stButton>button {{
            background: linear-gradient(90deg, {theme['primary']} 0%, {theme['secondary']} 100%);
            color: #FFFFFF !important;
            border: None;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            opacity: 0.9;
            transform: scale(1.02);
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 23, 42, 0.85) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 8px;
            border-radius: 12px;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 45px;
            border-radius: 8px;
            color: #A0AEC0;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {theme['primary']} !important;
            color: #FFFFFF !important;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# ==============================================================================
# 5. SMART FEATURE WIDGET GENERATOR
# ==============================================================================
def create_smart_input_widget(
    feature_name: str,
    categorical_maps: Dict[str, List[Any]]
) -> Tuple[Any, str]:
    """Infers the best widget type (dropdown, checkbox, slider, text, float) for a feature."""
    f_lower = feature_name.lower().strip()

    # 1. Pipeline Preserved Categories
    if feature_name in categorical_maps:
        options = categorical_maps[feature_name]
        val = st.selectbox(feature_name, options=options)
        return val, "categorical"

    # 2. Gender / Sex Detection
    if "gender" in f_lower or f_lower == "sex":
        val = st.selectbox(feature_name, options=["Female", "Male", "Other"])
        return val, "gender"

    # 3. Education Level Detection
    if "education" in f_lower or "degree" in f_lower:
        options = ["High School", "Bachelor's", "Master's", "Doctorate", "None"]
        val = st.selectbox(feature_name, options=options)
        return val, "categorical"

    # 4. Binary/Boolean Flags
    if any(f_lower.startswith(prefix) for prefix in ["is_", "has_", "contains_"]) or f_lower in ["internet_access", "extracurricular", "waterfront present"]:
        val = st.radio(feature_name, options=["Yes", "No"], horizontal=True)
        return (1 if val == "Yes" else 0), "boolean"

    # 5. Year / Built Detection
    if "year" in f_lower or "built" in f_lower:
        val = st.number_input(feature_name, min_value=1800, max_value=2030, value=2010, step=1)
        return val, "numeric"

    # 6. Age
    if f_lower == "age":
        val = st.number_input(feature_name, min_value=0, max_value=120, value=22, step=1)
        return val, "numeric"

    # 7. Scores / Percentages / Rates
    if "rate" in f_lower or "score" in f_lower or "percentage" in f_lower:
        val = st.number_input(feature_name, min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        return val, "numeric"

    # 8. Hours / Counts
    if "hours" in f_lower or "count" in f_lower or "bedroom" in f_lower or "bathroom" in f_lower or "floor" in f_lower:
        val = st.number_input(feature_name, min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        return val, "numeric"

    # 9. Text / Speech / NLP Features
    if any(keyword in f_lower for keyword in ["text", "comment", "review", "description", "message", "title"]):
        val = st.text_input(feature_name, value="Sample text entry")
        return val, "text"

    # Fallback Standard Numeric Input
    val = st.number_input(feature_name, value=1.0, step=0.1)
    return val, "numeric"


# ==============================================================================
# 6. UI RENDERERS
# ==============================================================================
def render_header(theme: Dict[str, Any], analysis: Dict[str, Any]):
    """Renders top header banner."""
    st.markdown(
        f"""
        <div style="padding: 20px 0px 10px 0px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 25px;">
            <h1 style="margin: 0; font-size: 2.3rem;">{theme['icon']} {theme['title']}</h1>
            <p style="color: #A0AEC0; font-size: 1.05rem; margin-top: 5px;">
                Interface Domain: <span style="color:{theme['accent']}; font-weight:600;">{theme['name']}</span> | 
                Architecture: <span style="color:{theme['accent']}; font-weight:600;">{analysis['type']}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_tab(analysis: Dict[str, Any], theme: Dict[str, Any]):
    """Renders high-level metrics and model details."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Architecture", analysis["type"])
    with col2:
        st.metric("Estimator Class", analysis["estimator_type"].capitalize())
    with col3:
        st.metric("Feature Count", analysis["n_features"])
    with col4:
        st.metric("scikit-learn Version", str(analysis.get("sklearn_version", "N/A")))

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("⚙️ Model Parameters")
        if analysis["params"]:
            params_df = pd.DataFrame(
                list(analysis["params"].items()),
                columns=["Parameter", "Value"],
            )
            st.dataframe(params_df, use_container_width=True, height=350)
        else:
            st.info("No explicit hyperparameters recorded in object.")

    with c2:
        st.subheader("🧩 Pipeline & Hierarchy")
        if analysis["is_pipeline"]:
            st.markdown("**Pipeline Operations Sequence:**")
            for idx, step in enumerate(analysis["pipeline_steps"], 1):
                st.markdown(f"`Step {idx}` : **{step}**")
        else:
            st.info("Object is a direct standalone estimator (non-Pipeline).")

        st.subheader("🏷️ Features Identified")
        if analysis["feature_names"]:
            st.write(analysis["feature_names"])


def render_visualization_tab(analysis: Dict[str, Any], theme: Dict[str, Any]):
    """Generates charts using the theme's color palette."""
    st.subheader("📊 Model Visualizations")

    if analysis["coefficients"] is not None and len(analysis["feature_names"]) > 0:
        coefs = np.array(analysis["coefficients"]).flatten()
        features = analysis["feature_names"]

        if len(coefs) == len(features):
            df_coef = pd.DataFrame(
                {"Feature": features, "Coefficient": coefs}
            ).sort_values(by="Coefficient", ascending=True)

            fig = px.bar(
                df_coef,
                x="Coefficient",
                y="Feature",
                orientation="h",
                title="Linear Coefficients / Feature Importance",
                color="Coefficient",
                color_continuous_scale=theme["plotly_scale"],
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Coefficients dimensions do not directly match feature names length.")
    else:
        st.info("Feature coefficient plot not supported for this object type.")


def render_prediction_tab(obj: Any, analysis: Dict[str, Any], theme: Dict[str, Any]):
    """Dynamically builds inputs with categorical dropdowns, switches, and float inputs."""
    st.subheader("🔮 Dynamic Inference Portal")

    if analysis["n_features"] == 0:
        st.warning("Cannot construct prediction inputs: Feature counts could not be auto-detected.")
        return

    st.write("Adjust feature values below to generate real-time inferences:")

    input_data = {}
    cols = st.columns(2)

    for idx, feature_name in enumerate(analysis["feature_names"]):
        col = cols[idx % 2]
        with col:
            val, f_type = create_smart_input_widget(
                feature_name, analysis["categorical_maps"]
            )
            
            # Map categorical string inputs to simple numerical placeholders if non-pipeline raw model requires numeric input
            if f_type in ["gender", "categorical"] and isinstance(val, str) and not analysis["is_pipeline"]:
                mapping = {"Female": 0, "Male": 1, "Other": 2, "High School": 0, "Bachelor's": 1, "Master's": 2, "Doctorate": 3, "None": 0}
                input_data[feature_name] = mapping.get(val, 0)
            else:
                input_data[feature_name] = val

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Execute Prediction", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = obj.predict(input_df)

            st.markdown("---")
            st.subheader("Inference Outcome")

            res_val = prediction[0]
            if isinstance(res_val, (int, float, np.number)):
                st.markdown(
                    f"""
                    <div style="background:{theme['card_bg']}; border:2px solid {theme['primary']}; border-radius:12px; padding:20px; text-align:center;">
                        <h3 style="margin:0; color:#A0AEC0;">Predicted Output</h3>
                        <h1 style="margin:0; color:{theme['accent']}; font-size: 3rem;">{res_val:,.2f}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.success(f"Predicted Class: **{res_val}**")

            if hasattr(obj, "predict_proba"):
                try:
                    proba = obj.predict_proba(input_df)[0]
                    st.subheader("Class Probabilities")
                    proba_df = pd.DataFrame({
                        "Class": analysis["classes"],
                        "Probability": proba
                    })
                    fig = px.bar(
                        proba_df,
                        x="Class",
                        y="Probability",
                        color="Probability",
                        color_continuous_scale=theme["plotly_scale"],
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as pe:
                    logger.warning(f"Probability prediction unavailable: {str(pe)}")

        except Exception as e:
            st.error(f"Error during prediction execution: {str(e)}")


# ==============================================================================
# 7. MAIN APPLICATION ENTRYPOINT
# ==============================================================================
def main():
    st.sidebar.title("📦 Object Loader")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Serialized Model (.pkl)", type=["pkl", "pickle"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        obj, err = ModelInspector.load_object(file_bytes)

        if err:
            st.error(f"⚠️ Failed to load pickle: {err}")
            return

        analysis = ModelInspector.analyze_object(obj)

        # Allow user override for theme interface or use auto-detected theme
        auto_theme = ThemeEngine.infer_theme(analysis)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Interface Theme")
        override = st.sidebar.checkbox("Override Auto-Detected Theme")

        if override:
            selected_theme_key = st.sidebar.selectbox(
                "Choose Domain Interface",
                options=list(ThemeEngine.THEMES.keys()),
                format_func=lambda x: ThemeEngine.THEMES[x]["name"],
            )
            theme = ThemeEngine.THEMES[selected_theme_key]
        else:
            theme = auto_theme

        inject_custom_styles(theme)
        render_header(theme, analysis)

        tab1, tab2, tab3 = st.tabs(
            ["🔍 Model Overview", "📈 Visualizations", "⚡ Live Prediction"]
        )

        with tab1:
            render_overview_tab(analysis, theme)

        with tab2:
            render_visualization_tab(analysis, theme)

        with tab3:
            render_prediction_tab(obj, analysis, theme)

    else:
        default_theme = ThemeEngine.THEMES["default"]
        inject_custom_styles(default_theme)

        st.markdown(
            """
            <div style="text-align: center; padding: 50px 20px;">
                <h1 style="font-size: 3rem;">🤖 Universal Model Analytics Platform</h1>
                <p style="color: #A0AEC0; font-size: 1.2rem; max-width: 600px; margin: 0 auto 30px auto;">
                    Upload any scikit-learn model or pipeline (.pkl) in the sidebar to generate custom feature controls, multi-domain interface layouts, and optimized visualization palettes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
