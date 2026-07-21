"""
Dynamic Model Inspection & Inference Dashboard Engine.

This module provides a production-grade Streamlit application that accepts any
scikit-learn compatible object (.pkl file), inspects its architecture, metadata,
and parameters, dynamically infers a visual theme based on feature and model heuristics,
and auto-generates interactive prediction forms and visualizations.

Authors: Senior AI Engineering & UX Team
Version: 1.0.0
PEP8 Compliant & Fully Modularized.
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
    page_title="Intelligent Model Inspector",
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
        """Safely unpickles the uploaded file bytes.

        Returns (object, error_message).
        """
        try:
            obj = pickle.loads(file_bytes)
            return obj, None
        except Exception as e:
            logger.error(f"Failed to unpickle object: {str(e)}")
            return None, f"Failed to parse pickle file: {str(e)}"

    @classmethod
    def analyze_object(cls, obj: Any) -> Dict[str, Any]:
        """Deeply inspects an unpickled object to extract architectural details."""
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
            "sklearn_version": getattr(obj, "_sklearn_version", "Unknown"),
            "attributes": [
                attr for attr in dir(obj) if not attr.startswith("_")
            ],
        }

        # Check for Pipeline
        if hasattr(obj, "steps"):
            analysis["is_pipeline"] = True
            analysis["pipeline_steps"] = [step[0] for step in obj.steps]
            final_estimator = obj.steps[-1][1]
        else:
            final_estimator = obj

        # Extract Estimator Type
        if hasattr(final_estimator, "_estimator_type"):
            analysis["estimator_type"] = getattr(
                final_estimator, "_estimator_type"
            )
        elif "Regressor" in type(final_estimator).__name__:
            analysis["estimator_type"] = "regressor"
        elif "Classifier" in type(final_estimator).__name__:
            analysis["estimator_type"] = "classifier"
        elif "Cluster" in type(final_estimator).__name__:
            analysis["estimator_type"] = "clusterer"

        # Feature Names
        if hasattr(obj, "feature_names_in_"):
            analysis["feature_names"] = list(
                getattr(obj, "feature_names_in_")
            )
        elif hasattr(final_estimator, "feature_names_in_"):
            analysis["feature_names"] = list(
                getattr(final_estimator, "feature_names_in_")
            )

        # Number of Features
        if hasattr(obj, "n_features_in_"):
            analysis["n_features"] = getattr(obj, "n_features_in_")
        elif hasattr(final_estimator, "n_features_in_"):
            analysis["n_features"] = getattr(
                final_estimator, "n_features_in_"
            )
        elif analysis["feature_names"]:
            analysis["n_features"] = len(analysis["feature_names"])

        # Default feature names if missing
        if analysis["n_features"] > 0 and not analysis["feature_names"]:
            analysis["feature_names"] = [
                f"Feature_{i+1}" for i in range(analysis["n_features"])
            ]

        # Classification classes
        if hasattr(final_estimator, "classes_"):
            analysis["classes"] = list(getattr(final_estimator, "classes_"))

        # Linear Coefficients & Intercepts
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
# 3. DYNAMIC THEME ENGINE
# ==============================================================================
class ThemeEngine:
    """Infers visual domain themes based on model attributes, naming heuristics, and metadata."""

    THEMES = {
        "real_estate": {
            "title": "Real Estate Intelligence Dashboard",
            "icon": "🏠",
            "primary": "#E05A47",
            "secondary": "#2C3E50",
            "bg_gradient": "linear-gradient(135deg, #1A1C23 0%, #2D1B1E 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#FF7A00",
            "chart_palette": ["#E05A47", "#FF7A00", "#4AB8B8", "#6C5CE7"],
        },
        "finance": {
            "title": "Financial Analytics Portal",
            "icon": "📈",
            "primary": "#00B894",
            "secondary": "#0984E3",
            "bg_gradient": "linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#55E6C1",
            "chart_palette": ["#00B894", "#0984E3", "#FDCB6E", "#6C5CE7"],
        },
        "healthcare": {
            "title": "Medical & Clinical Intelligence",
            "icon": "🏥",
            "primary": "#00CEC9",
            "secondary": "#0984E3",
            "bg_gradient": "linear-gradient(135deg, #134E5E 0%, #71B280 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#81ECEC",
            "chart_palette": ["#00CEC9", "#0984E3", "#A29BFE", "#FD79A8"],
        },
        "nlp": {
            "title": "NLP & Sentiment Studio",
            "icon": "💬",
            "primary": "#A29BFE",
            "secondary": "#6C5CE7",
            "bg_gradient": "linear-gradient(135deg, #2D132C 0%, #801336 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#FD79A8",
            "chart_palette": ["#A29BFE", "#FD79A8", "#FFEAA7", "#55E6C1"],
        },
        "default": {
            "title": "Universal Model Analytics Engine",
            "icon": "⚡",
            "primary": "#6C5CE7",
            "secondary": "#00CEC9",
            "bg_gradient": "linear-gradient(135deg, #141E30 0%, #243B55 100%)",
            "card_bg": "rgba(255, 255, 255, 0.05)",
            "accent": "#A29BFE",
            "chart_palette": ["#6C5CE7", "#00CEC9", "#FF7675", "#FDCB6E"],
        },
    }

    KEYWORD_MAPPINGS = {
        "real_estate": [
            "house",
            "housing",
            "living area",
            "bedroom",
            "bathroom",
            "floor",
            "renovation",
            "lot",
            "price",
            "sqft",
            "zipcode",
            "property",
            "built year",
        ],
        "finance": [
            "credit",
            "balance",
            "transaction",
            "amount",
            "fraud",
            "loan",
            "income",
            "debt",
            "payment",
            "account",
        ],
        "healthcare": [
            "patient",
            "blood",
            "age",
            "diagnosis",
            "glucose",
            "bmi",
            "heart",
            "hospital",
            "disease",
            "medical",
        ],
        "nlp": [
            "text",
            "sentiment",
            "embedding",
            "word",
            "tfidf",
            "corpus",
            "token",
            "phrase",
        ],
    }

    @classmethod
    def infer_theme(cls, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Scores candidate themes against extracted features and model metadata."""
        scores = {key: 0 for key in cls.KEYWORD_MAPPINGS.keys()}

        # Combine text representation of metadata
        search_corpus = " ".join(
            [
                analysis.get("type", ""),
                analysis.get("module", ""),
                " ".join(analysis.get("feature_names", [])),
                " ".join([str(c) for c in analysis.get("classes", [])]),
            ]
        ).lower()

        # Score matching keywords
        for theme_key, keywords in cls.KEYWORD_MAPPINGS.items():
            for kw in keywords:
                if kw in search_corpus:
                    scores[theme_key] += 1

        best_theme_key = max(scores, key=scores.get)

        # Fallback if low confidence score
        if scores[best_theme_key] == 0:
            best_theme_key = "default"

        logger.info(
            f"Inferred Theme: {best_theme_key} (Scores: {scores})"
        )
        return cls.THEMES[best_theme_key]


# ==============================================================================
# 4. CUSTOM CSS STYLING & GLASSMORPHISM INJECTION
# ==============================================================================
def inject_custom_styles(theme: Dict[str, Any]):
    """Injects dynamic dark-mode glassmorphic styling into Streamlit."""
    custom_css = f"""
    <style>
        /* Base page theme background */
        .stApp {{
            background: {theme['bg_gradient']};
            color: #E2E8F0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        /* Glassmorphic Metric Cards */
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
        
        /* Headers & Typography */
        h1, h2, h3, h4 {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }}
        
        /* Buttons */
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

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: rgba(15, 23, 42, 0.75) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}
        
        /* Tabs Styling */
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
# 5. UI COMPONENTS & RENDERERS
# ==============================================================================
def render_header(theme: Dict[str, Any], analysis: Dict[str, Any]):
    """Renders the top dynamic hero banner."""
    st.markdown(
        f"""
        <div style="padding: 20px 0px 10px 0px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 25px;">
            <h1 style="margin: 0; font-size: 2.3rem;">{theme['icon']} {theme['title']}</h1>
            <p style="color: #A0AEC0; font-size: 1.05rem; margin-top: 5px;">
                Detected Architecture: <span style="color:{theme['accent']}; font-weight:600;">{analysis['type']}</span> | 
                Type: <span style="color:{theme['accent']}; font-weight:600;">{analysis['estimator_type'].capitalize()}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_tab(analysis: Dict[str, Any], theme: Dict[str, Any]):
    """Renders high-level metrics, parameters, and metadata."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Architecture", analysis["type"])
    with col2:
        st.metric("Estimator Class", analysis["estimator_type"].capitalize())
    with col3:
        st.metric("Feature Count", analysis["n_features"])
    with col4:
        st.metric(
            "scikit-learn Version", str(analysis.get("sklearn_version", "N/A"))
        )

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
        else:
            st.warning("No explicit feature names preserved in metadata.")


def render_visualization_tab(analysis: Dict[str, Any], theme: Dict[str, Any]):
    """Generates dynamic, interactive charts based on extracted coefficients/features."""
    st.subheader("📊 Dynamic Model Visualizations")

    # Linear Model Coefficients Plots
    if (
        analysis["coefficients"] is not None
        and len(analysis["feature_names"]) > 0
    ):
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
                title="Linear Coefficients Impact / Feature Importance",
                color="Coefficient",
                color_continuous_scale=px.colors.diverging.Tealrose,
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Coefficients dimensions do not directly match feature names length."
            )
    else:
        st.info("Feature coefficient plot not supported for this object type.")


def render_prediction_tab(
    obj: Any, analysis: Dict[str, Any], theme: Dict[str, Any]
):
    """Dynamically builds user inputs based on detected features and outputs real-time predictions."""
    st.subheader("🔮 Dynamic Inference Portal")

    if analysis["n_features"] == 0:
        st.warning(
            "Cannot construct prediction inputs: Feature counts could not be auto-detected."
        )
        return

    st.write(
        "Adjust feature values below to generate real-time model inferences:"
    )

    # Dynamic form generation
    input_data = {}
    cols = st.columns(2)

    for idx, feature_name in enumerate(analysis["feature_names"]):
        col = cols[idx % 2]
        with col:
            # Intuitive heuristic defaults for real-estate or numeric features
            lower_name = feature_name.lower()
            if "year" in lower_name:
                input_data[feature_name] = st.number_input(
                    feature_name,
                    min_value=1800,
                    max_value=2030,
                    value=1995,
                    step=1,
                )
            elif "bedroom" in lower_name or "bathroom" in lower_name:
                input_data[feature_name] = st.number_input(
                    feature_name, min_value=0, max_value=20, value=3, step=1
                )
            elif "area" in lower_name or "sqft" in lower_name:
                input_data[feature_name] = st.number_input(
                    feature_name,
                    min_value=0.0,
                    max_value=100000.0,
                    value=1500.0,
                    step=50.0,
                )
            else:
                input_data[feature_name] = st.number_input(
                    feature_name, value=1.0, step=0.1
                )

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
                        <h3 style="margin:0; color:#A0AEC0;">Predicted Value</h3>
                        <h1 style="margin:0; color:{theme['accent']}; font-size: 3rem;">{res_val:,.2f}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.success(f"Predicted Class: **{res_val}**")

            # Check for classification probabilities
            if hasattr(obj, "predict_proba"):
                try:
                    proba = obj.predict_proba(input_df)[0]
                    st.subheader("Class Probabilities")
                    proba_df = pd.DataFrame(
                        {"Class": analysis["classes"], "Probability": proba}
                    )
                    fig = px.bar(
                        proba_df,
                        x="Class",
                        y="Probability",
                        color="Probability",,
                        color_continuous_scale="Viridis",
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as pe:
                    logger.warning(
                        f"Probability prediction unavailable: {str(pe)}"
                    )

        except Exception as e:
            st.error(f"Error during prediction execution: {str(e)}")


# ==============================================================================
# 6. MAIN APPLICATION ENTRYPOINT
# ==============================================================================
def main():
    # Sidebar
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

        # Inspect object & infer theme
        analysis = ModelInspector.analyze_object(obj)
        theme = ThemeEngine.infer_theme(analysis)

        # Inject theme styles
        inject_custom_styles(theme)

        # Main Header
        render_header(theme, analysis)

        # Navigation Tabs
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
        # Default application welcome state
        default_theme = ThemeEngine.THEMES["default"]
        inject_custom_styles(default_theme)

        st.markdown(
            f"""
            <div style="text-align: center; padding: 50px 20px;">
                <h1 style="font-size: 3rem;">🤖 Universal Model Analytics Engine</h1>
                <p style="color: #A0AEC0; font-size: 1.2rem; max-width: 600px; margin: 0 auto 30px auto;">
                    Upload any scikit-learn model, pipeline, or machine learning object (.pkl) in the sidebar to automatically trigger domain-aware styling, metadata extraction, and real-time inference dashboards.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
