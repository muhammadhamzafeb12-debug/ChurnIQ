import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("churn_pipeline.pkl")
DATA_PATH = Path("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

st.markdown("""
<style>
.stApp {
    background: #F3F5F4;
    color: #263746;
}

[data-testid="stSidebar"] {
    background: #263746;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: #263746;
    padding: 28px 30px;
    border-radius: 16px;
    color: #FFFFFF;
    margin-bottom: 22px;
    border-left: 4px solid #56877F;
}

.hero h1 {
    color: #FFFFFF;
    margin: 0;
    font-size: 30px;
    font-weight: 700;
}

.hero p {
    color: #DCE5E2;
    margin: 8px 0 0 0;
}

.kpi {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #DCE5E2;
    min-height: 112px;
    box-shadow: 0 3px 12px rgba(38, 55, 70, 0.045);
}

.kpi-label {
    color: #667783;
    font-size: 13px;
}

.kpi-value {
    color: #263746;
    font-size: 27px;
    font-weight: 700;
    margin-top: 7px;
}

.section {
    color: #263746;
    font-size: 20px;
    font-weight: 700;
    margin: 12px 0 5px 0;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #56877F;
    color: #FFFFFF;
    border: 1px solid #56877F;
    border-radius: 9px;
    min-height: 44px;
    font-weight: 600;
    transition: background 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #426F68;
    color: #FFFFFF;
    border-color: #426F68;
}

div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    border: 1px solid #DCE5E2;
    border-radius: 10px;
}

input, textarea, [data-baseweb="select"] > div {
    border-color: #DCE5E2 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


try:
    model = load_model()
except Exception as exc:
    st.error(f"Model load failed: {exc}")
    st.stop()


FEATURES = list(model.feature_names_in_)


CHURN_THRESHOLD = 0.30


@st.cache_data
def evaluate_model():
    """Evaluate a fresh copy of the pipeline on a stratified holdout set."""
    dataset = load_dataset()

    if dataset is None:
        raise FileNotFoundError("Churn dataset CSV file was not found.")

    frame = dataset.copy()
    frame.columns = frame.columns.str.lower()

    if "churn" not in frame.columns:
        raise ValueError("Dataset must contain a Churn target column.")

    X = frame[FEATURES].copy()
    X["totalcharges"] = X["totalcharges"].astype(str)

    y = frame["churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Churn column contains unexpected or missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    evaluation_model = clone(model)
    evaluation_model.fit(X_train, y_train)

    classifier = evaluation_model.named_steps["classifier"]
    probabilities = evaluation_model.predict_proba(X_test)
    churn_index = list(classifier.classes_).index(1)
    churn_probabilities = probabilities[:, churn_index]

    threshold_results = []

    for threshold in [0.30, 0.40, 0.50]:
        predictions = (churn_probabilities >= threshold).astype(int)

        threshold_results.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0)
        })

    final_predictions = (churn_probabilities >= CHURN_THRESHOLD).astype(int)

    matrix = confusion_matrix(
        y_test,
        final_predictions,
        labels=[0, 1]
    )

    return {
        "metrics": threshold_results[0],
        "thresholds": threshold_results,
        "confusion_matrix": matrix,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }


def predict_churn(data):
    frame = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
    frame = frame[FEATURES].copy()

    # Keep TotalCharges compatible with the trained pipeline.
    frame["totalcharges"] = frame["totalcharges"].astype(str)

    probabilities = model.predict_proba(frame)

    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    churn_index = classes.index(1)

    churn_probabilities = probabilities[:, churn_index]

    # Apply the configured churn decision threshold.
    predictions = (churn_probabilities >= CHURN_THRESHOLD).astype(int)

    return pd.DataFrame({
        "prediction": predictions,
        "prediction_label": np.where(predictions == 1, "Churn", "No Churn"),
        "churn_probability": churn_probabilities
    })


def risk_level(probability):
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def metric_card(label, value, note=""):
    st.markdown(
        f"""<div class="kpi">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{note}</div>
        </div>""",
        unsafe_allow_html=True
    )


def customer_form():
    st.markdown('<div class="section">Customer information</div>', unsafe_allow_html=True)
    st.caption("Complete the details below to estimate the customer's churn risk.")

    with st.form("customer_churn_form"):
        st.markdown("**01 · Personal information**")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            seniorcitizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with c2:
            partner = st.selectbox("Has partner?", ["No", "Yes"])
            dependents = st.selectbox("Has dependents?", ["No", "Yes"])
        with c3:
            tenure = st.number_input("Tenure (months)", 0, 100, 12)
            phoneservice = st.selectbox("Phone service", ["Yes", "No"])

        st.divider()
        st.markdown("**02 · Services and subscription**")
        c1, c2, c3 = st.columns(3)
        with c1:
            multiplelines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internetservice = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            onlinesecurity = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        with c2:
            onlinebackup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            deviceprotection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            techsupport = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        with c3:
            streamingtv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streamingmovies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        st.divider()
        st.markdown("**03 · Billing information**")
        c1, c2, c3 = st.columns(3)
        with c1:
            paperlessbilling = st.selectbox("Paperless billing", ["Yes", "No"])
        with c2:
            paymentmethod = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
        with c3:
            monthlycharges = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=1.0)
            totalcharges = st.number_input("Total charges", min_value=0.0, value=840.0, step=1.0)

        submitted = st.form_submit_button("Run churn prediction", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender,
            "seniorcitizen": seniorcitizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "phoneservice": phoneservice,
            "multiplelines": multiplelines,
            "internetservice": internetservice,
            "onlinesecurity": onlinesecurity,
            "onlinebackup": onlinebackup,
            "deviceprotection": deviceprotection,
            "techsupport": techsupport,
            "streamingtv": streamingtv,
            "streamingmovies": streamingmovies,
            "contract": contract,
            "paperlessbilling": paperlessbilling,
            "paymentmethod": paymentmethod,
            "monthlycharges": monthlycharges,
            "totalcharges": str(totalcharges)
        }

        try:
            result = predict_churn(customer).iloc[0]
            probability = float(result["churn_probability"])
            prediction = result["prediction_label"]
            risk = risk_level(probability)

            st.divider()
            st.markdown('<div class="section">Prediction result</div>', unsafe_allow_html=True)

            a, b, c = st.columns(3)
            with a:
                metric_card("Predicted outcome", prediction)
            with b:
                metric_card("Churn probability", f"{probability:.1%}")
            with c:
                metric_card("Risk category", risk)

            st.progress(probability, text=f"Estimated churn probability: {probability:.1%}")

            st.markdown("### Customer risk assessment")

            if risk == "High":
                st.error("HIGH RISK — This customer needs priority attention.")
            elif risk == "Medium":
                st.warning("MEDIUM RISK — Proactive engagement is recommended.")
            else:
                st.success("LOW RISK — Customer retention outlook is positive.")

            st.markdown("### Recommended retention action plan")

            recommendations = []

            if contract == "Month-to-month":
                recommendations.append(
                    "Offer a suitable long-term contract with clear benefits."
                )

            if tenure <= 12:
                recommendations.append(
                    "Contact this relatively new customer and improve onboarding support."
                )

            if paymentmethod == "Electronic check":
                recommendations.append(
                    "Offer convenient automatic payment options if suitable."
                )

            if techsupport == "No" and internetservice != "No":
                recommendations.append(
                    "Explain available technical support options to the customer."
                )

            if onlinesecurity == "No" and internetservice != "No":
                recommendations.append(
                    "Explain available online security plans and their benefits."
                )

            if monthlycharges >= 80:
                recommendations.append(
                    "Review the customer's monthly bill and offer suitable value options."
                )

            if risk == "High":
                recommendations.insert(
                    0, "Prioritize a personal customer-retention call."
                )
            elif risk == "Medium":
                recommendations.insert(
                    0, "Schedule a proactive customer satisfaction follow-up."
                )
            else:
                recommendations.insert(
                    0, "Maintain service quality and continue regular engagement."
                )

            for number, recommendation in enumerate(recommendations, 1):
                st.markdown(f"**{number}.** {recommendation}")

            st.caption(
                "Recommendations are based on customer information and business rules. "
                "Churn probability is a model estimate, not a guarantee of future behaviour."
            )

            st.caption("This is a machine-learning estimate, not a guarantee of future customer behaviour.")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# Sidebar navigation
with st.sidebar:
    st.markdown("## 📊 ChurnIQ")
    st.caption("Customer intelligence platform")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Customer Prediction", "Bulk Prediction", "Model Insights", "Admin"],
        label_visibility="visible"
    )
    st.divider()
    st.caption("Machine Learning · Customer Retention")


if page == "Overview":
    st.markdown("""
    <div class="hero">
      <h1>Customer Analytics Overview</h1>
      <p>Understand churn patterns and make smarter customer-retention decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()

    if df is not None and "Churn" in df.columns:
        total = len(df)
        churned = int((df["Churn"].astype(str).str.lower() == "yes").sum())
        retained = total - churned
        rate = churned / total if total else 0

        a, b, c, d = st.columns(4)
        with a:
            metric_card("Total customers", f"{total:,}", "Dataset records")
        with b:
            metric_card("Churned customers", f"{churned:,}", "Historical records")
        with c:
            metric_card("Retained customers", f"{retained:,}", "Historical records")
        with d:
            metric_card("Churn rate", f"{rate:.1%}", "Historical dataset")

        st.markdown('<div class="section">Customer churn distribution</div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            counts = df["Churn"].value_counts().rename_axis("Customer status").reset_index(name="Customers")
            st.bar_chart(counts.set_index("Customer status"))
        with right:
            st.markdown("**Churn share**")
            st.dataframe(
                pd.DataFrame({
                    "Status": ["No Churn", "Churn"],
                    "Customers": [retained, churned],
                    "Share": [f"{retained/total:.1%}", f"{rate:.1%}"]
                }),
                use_container_width=True,
                hide_index=True
            )
        st.caption("Overview figures are calculated from the supplied historical dataset, not live customer submissions.")
    else:
        st.warning("Dataset not found. Customer prediction can still be used.")


elif page == "Customer Prediction":
    st.markdown("""
    <div class="hero">
      <h1>Customer Churn Assessment</h1>
      <p>Enter customer details and receive an instant churn-risk estimate.</p>
    </div>
    """, unsafe_allow_html=True)
    customer_form()


elif page == "Bulk Prediction":
    st.markdown("""
    <div class="hero">
      <h1>Bulk Customer Prediction</h1>
      <p>Analyse multiple customers, identify churn risks and generate actionable business insights.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Upload a CSV containing the same 19 feature columns used by the model. Do not include the Churn target column.")

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            file_token = (uploaded.name, uploaded.size)

            if st.session_state.get("bulk_file_token") != file_token:
                st.session_state.pop("bulk_output", None)
                st.session_state["bulk_file_token"] = file_token

            missing = [col for col in FEATURES if col not in batch.columns]

            if batch.empty:
                st.error("The uploaded CSV is empty.")
            elif missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.markdown("### Uploaded customer data")
                st.caption(f"File: {uploaded.name} · {len(batch):,} customer records")

                with st.expander("Preview uploaded customers"):
                    st.dataframe(batch.head(10), use_container_width=True)

                if st.button("Predict all customers", type="primary"):
                    predictions = predict_churn(batch[FEATURES])

                    output = batch.copy()
                    output["prediction"] = predictions["prediction_label"]
                    output["churn_probability"] = predictions["churn_probability"].round(4)
                    output["risk_level"] = predictions["churn_probability"].apply(risk_level)

                    st.session_state["bulk_output"] = output

                output = st.session_state.get("bulk_output")

                if output is not None:
                    total = len(output)
                    churned = int((output["prediction"] == "Churn").sum())
                    retained = total - churned
                    average_probability = output["churn_probability"].mean()

                    high = int((output["risk_level"] == "High").sum())
                    medium = int((output["risk_level"] == "Medium").sum())
                    low = int((output["risk_level"] == "Low").sum())

                    st.success(f"Predictions completed for {total:,} customers.")

                    st.markdown("### Business analytics overview")

                    a, b, c, d = st.columns(4)

                    with a:
                        metric_card("Total customers", f"{total:,}", "Records analysed")

                    with b:
                        metric_card("Predicted churn", f"{churned:,}", f"{churned / total:.1%} of customers")

                    with c:
                        metric_card("Customers retained", f"{retained:,}", f"{retained / total:.1%} of customers")

                    with d:
                        metric_card("Average churn probability", f"{average_probability:.1%}", "Model-estimated average")

                    st.markdown("### Customer risk distribution")

                    r1, r2, r3 = st.columns(3)

                    with r1:
                        metric_card("High risk", f"{high:,}", "Immediate attention")

                    with r2:
                        metric_card("Medium risk", f"{medium:,}", "Monitor closely")

                    with r3:
                        metric_card("Low risk", f"{low:,}", "Regular engagement")

                    left, right = st.columns(2)

                    with left:
                        st.markdown("**Predicted customer status**")
                        status_chart = pd.DataFrame({
                            "Customer status": ["Churn", "No Churn"],
                            "Customers": [churned, retained]
                        })
                        st.bar_chart(status_chart.set_index("Customer status"))

                    with right:
                        st.markdown("**Risk level breakdown**")
                        risk_chart = pd.DataFrame({
                            "Risk level": ["High", "Medium", "Low"],
                            "Customers": [high, medium, low]
                        })
                        st.bar_chart(risk_chart.set_index("Risk level"))

                    st.markdown("### Customer prediction report")

                    risk_filter = st.selectbox(
                        "Filter customers by risk level",
                        ["All customers", "High", "Medium", "Low"]
                    )

                    if risk_filter == "All customers":
                        filtered_output = output
                    else:
                        filtered_output = output[
                            output["risk_level"] == risk_filter
                        ]

                    st.caption(f"Showing {len(filtered_output):,} of {total:,} customers")

                    st.dataframe(
                        filtered_output,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "Download complete prediction report",
                        output.to_csv(index=False).encode("utf-8"),
                        file_name="ChurnIQ_Business_Prediction_Report.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        "Churn probabilities are machine-learning estimates, not guarantees. "
                        "Risk categories are based on the configured risk thresholds."
                    )

        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")


elif page == "Model Insights":
    st.markdown("""
    <div class="hero">
      <h1>Model Insights</h1>
      <p>Understand model performance, churn detection quality and the features influencing predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Evaluating model on holdout data..."):
            evaluation = evaluate_model()

        metrics = evaluation["metrics"]

        st.markdown("### Model Performance Overview")
        st.caption(
            "Holdout evaluation: fresh copy of the pipeline, "
            "80/20 stratified split, random_state=42."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Accuracy", f'{metrics["Accuracy"]:.1%}', "Overall correct predictions")

        with col2:
            metric_card("Precision", f'{metrics["Precision"]:.1%}', "Correct positive predictions")

        with col3:
            metric_card("Recall", f'{metrics["Recall"]:.1%}', "Actual churners detected")

        with col4:
            metric_card("F1 Score", f'{metrics["F1 Score"]:.1%}', "Precision and recall balance")

        st.caption(
            f'Training records: {evaluation["train_size"]:,} | '
            f'Test records: {evaluation["test_size"]:,} | '
            f'Production decision threshold: {CHURN_THRESHOLD:.2f}'
        )

        st.divider()

        st.markdown("### Confusion Matrix")
        st.write(
            "This matrix compares actual customer outcomes with the model's "
            "predictions using the production threshold of 0.30."
        )

        matrix_df = pd.DataFrame(
            evaluation["confusion_matrix"],
            index=["Actual: No Churn", "Actual: Churn"],
            columns=["Predicted: No Churn", "Predicted: Churn"]
        )

        st.dataframe(matrix_df, use_container_width=True)

        tn, fp, fn, tp = evaluation["confusion_matrix"].ravel()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("True Negatives", f"{tn:,}", "Correctly identified retained customers")
        with c2:
            metric_card("False Positives", f"{fp:,}", "Retained customers flagged as churn")
        with c3:
            metric_card("False Negatives", f"{fn:,}", "Churners missed by the model")
        with c4:
            metric_card("True Positives", f"{tp:,}", "Correctly detected churners")

        st.divider()

        st.markdown("### Business Insights")
        st.write(
            "Translate model evaluation results into practical customer "
            "retention decisions."
        )

        churn_total = tp + fn
        retained_total = tn + fp

        detection_rate = tp / churn_total if churn_total else 0
        missed_rate = fn / churn_total if churn_total else 0
        false_alarm_rate = fp / retained_total if retained_total else 0

        b1, b2, b3 = st.columns(3)

        with b1:
            metric_card(
                "Churners Detected",
                f"{detection_rate:.1%}",
                f"{tp:,} out of {churn_total:,} actual churners"
            )

        with b2:
            metric_card(
                "Churners Missed",
                f"{missed_rate:.1%}",
                f"{fn:,} actual churners not detected"
            )

        with b3:
            metric_card(
                "False Alarm Rate",
                f"{false_alarm_rate:.1%}",
                f"{fp:,} retained customers flagged"
            )

        st.markdown("#### Recommended Business Actions")

        st.markdown(
            "- **Prioritize high-risk customers:** Review customers flagged "
            "by the model and offer suitable retention support.\n"
            "- **Verify before offering discounts:** Some flagged customers "
            "may stay, so review their account and engagement first.\n"
            "- **Investigate missed churners:** Improve retention monitoring "
            "because the model does not detect every customer who leaves.\n"
            "- **Track results:** Monitor retention outcomes and periodically "
            "evaluate model performance."
        )

        st.info(
            "These are decision-support recommendations based on holdout "
            "evaluation results. Predictions do not guarantee that a customer "
            "will leave or stay."
        )

        st.divider()

        st.markdown("### Decision Threshold Comparison")
        st.write(
            "A lower threshold identifies more potential churners, "
            "but can also increase false alarms."
        )

        threshold_df = pd.DataFrame(evaluation["thresholds"])
        display_thresholds = threshold_df.copy()

        for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            display_thresholds[column] = display_thresholds[column].map(
                lambda value: f"{value:.1%}"
            )

        st.dataframe(display_thresholds, use_container_width=True, hide_index=True)

        chart_df = threshold_df.copy()
        chart_df["Threshold"] = chart_df["Threshold"].map(
            lambda value: f"{value:.2f}"
        )
        chart_df = chart_df.set_index("Threshold")[
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        ]

        st.line_chart(chart_df)

        st.info(
            "The current production threshold remains 0.30. "
            "It prioritizes detecting churners (recall) rather than simply "
            "maximizing overall accuracy. The comparison does not change "
            "your deployed model settings."
        )

        st.divider()

        st.markdown("### Feature Importance")

        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]

        names = preprocessor.get_feature_names_out()
        importance = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(15)

        def readable_feature_name(name):
            name = str(name)
            name = name.replace("num__", "").replace("cat__", "")

            if "_" in name:
                feature, category = name.split("_", 1)
                return f"{feature.replace('_', ' ').title()}: {category}"

            return name.replace("_", " ").title()

        importance_df["Feature"] = importance_df["Feature"].map(
            readable_feature_name
        )

        st.bar_chart(importance_df.set_index("Feature"))

        display_importance = importance_df.copy()
        display_importance["Importance"] = (
            display_importance["Importance"] * 100
        ).map(lambda value: f"{value:.2f}%")

        st.dataframe(
            display_importance.rename(
                columns={"Feature": "Feature Name", "Importance": "Importance (%)"}
            ),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Feature importance describes how the fitted Random Forest uses "
            "transformed features. It does not prove that a feature causes churn."
        )

        st.divider()

        st.markdown("### Model Configuration")
        config1, config2, config3 = st.columns(3)

        with config1:
            metric_card("Algorithm", "Random Forest", "Classification model")

        with config2:
            metric_card("Input Features", str(len(FEATURES)), "Customer information fields")

        with config3:
            metric_card("Decision Threshold", f"{CHURN_THRESHOLD:.2f}", "Current production setting")

        st.divider()
        st.markdown("### Model Limitations")

        with st.expander("Evaluation Scope and Limitations"):
            st.markdown("""
            **1. Dataset Dependency**

            Results are based on the available Telco Customer Churn dataset.
            Performance may differ on other companies and customer populations.

            **2. Holdout Evaluation**

            A fresh pipeline copy is trained on 80% of the data and evaluated
            on a stratified 20% holdout dataset.

            **3. Decision Threshold**

            The production threshold is 0.30. It prioritizes churn detection
            but may generate false positive predictions.

            **4. Feature Importance**

            Feature importance describes model usage, not cause-and-effect
            relationships.

            **5. Real-World Validation**

            Validate on recent customer data and monitor performance regularly
            before using predictions for business decisions.
            """)

    except Exception as exc:
        st.error(f"Could not generate model insights: {exc}")




elif page == "Admin":
    st.markdown("""
    <div class="hero">
      <h1>Admin Area</h1>
      <p>Restricted access for application administrators.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        admin_password = st.secrets.get("ADMIN_PASSWORD", "")
    except Exception:
        admin_password = ""

    if not admin_password:
        st.warning("Admin password is not configured yet. Add ADMIN_PASSWORD to Streamlit secrets before enabling admin access.")
    else:
        entered = st.text_input("Admin password", type="password")
        if st.button("Sign in"):
            if entered == admin_password:
                st.session_state["admin_authenticated"] = True
            else:
                st.session_state["admin_authenticated"] = False
                st.error("Incorrect password.")

        if st.session_state.get("admin_authenticated", False):
            st.success("Admin access granted.")
            st.write("Admin dashboard access is active.")
            st.info("Customer submissions are not stored by this version of the application.")
            df = load_dataset()
            if df is not None:
                st.write("Historical dataset summary")
                st.dataframe(df.describe(include="all").T, use_container_width=True)
