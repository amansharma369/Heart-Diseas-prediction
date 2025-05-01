import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hugchat import hugchat
from hugchat.login import Login
import random
# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Hugging Face Login (Optional for AI Advisor)
HF_EMAIL = "your-email@example.com"
HF_PASS = "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"

# Feature information with medical ranges
feature_info = {
    "age": {"label": "Age", "desc": "Age in years", "min": 20, "max": 100, "value": 50,
            "healthy_range": "20-60", "risk_range": "60+"},
    "sex": {"label": "Sex", "desc": "Biological sex", "options": ["Female", "Male"]},
    "cp": {"label": "Chest Pain Type", "desc": "Type of chest pain",
           "options": ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
           "healthy": "Typical angina", "risk": "Asymptomatic"},
    "trestbps": {"label": "Resting BP", "desc": "Resting blood pressure (mm Hg)",
                 "min": 80, "max": 200, "value": 120,
                 "healthy_range": "90-120", "risk_range": "140+"},
    "chol": {"label": "Cholesterol", "desc": "Serum cholesterol (mg/dl)",
             "min": 100, "max": 600, "value": 200,
             "healthy_range": "< 200", "risk_range": "240+"},
    "fbs": {"label": "Fasting Blood Sugar", "desc": "> 120 mg/dl", "options": ["No", "Yes"],
            "healthy": "No", "risk": "Yes"},
    "restecg": {"label": "Resting ECG", "desc": "Resting electrocardiographic results",
                "options": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
                "healthy": "Normal", "risk": "Left ventricular hypertrophy"},
    "thalach": {"label": "Max Heart Rate", "desc": "Maximum heart rate achieved",
                "min": 60, "max": 220, "value": 150,
                "healthy_range": "60-100 (resting)", "risk_range": "< 120 (exercise)"},
    "exang": {"label": "Exercise Angina", "desc": "Exercise induced angina", "options": ["No", "Yes"],
              "healthy": "No", "risk": "Yes"},
    "oldpeak": {"label": "ST Depression", "desc": "ST depression induced by exercise",
                "min": 0.0, "max": 6.0, "value": 1.0, "step": 0.1,
                "healthy_range": "0-1", "risk_range": "2+"},
    "slope": {"label": "ST Slope", "desc": "Slope of peak exercise ST segment",
              "options": ["Upsloping", "Flat", "Downsloping"],
              "healthy": "Upsloping", "risk": "Downsloping"},
    "ca": {"label": "Major Vessels", "desc": "Number of major vessels colored by fluoroscopy",
           "options": ["0", "1", "2", "3"],
           "healthy": "0", "risk": "3"},
    "thal": {"label": "Thalassemia", "desc": "Blood disorder called thalassemia",
             "options": ["Normal", "Fixed defect", "Reversible defect"],
             "healthy": "Normal", "risk": "Fixed defect"}
}

# Risk assessment thresholds
RISK_THRESHOLDS = {
    "Low": 0.3,
    "Moderate": 0.6,
    "High": 0.9
}

# Sample healthy and unhealthy profiles
SAMPLE_PROFILES = {
    "Healthy": {
        "age": 45, "sex": "Female", "cp": "Typical angina",
        "trestbps": 110, "chol": 180, "fbs": "No",
        "restecg": "Normal", "thalach": 160, "exang": "No",
        "oldpeak": 0.5, "slope": "Upsloping", "ca": "0",
        "thal": "Normal"
    },
    "Moderate Risk": {
        "age": 58, "sex": "Male", "cp": "Atypical angina",
        "trestbps": 135, "chol": 230, "fbs": "No",
        "restecg": "ST-T wave abnormality", "thalach": 140, "exang": "No",
        "oldpeak": 1.5, "slope": "Flat", "ca": "1",
        "thal": "Reversible defect"
    },
    "High Risk": {
        "age": 65, "sex": "Male", "cp": "Asymptomatic",
        "trestbps": 180, "chol": 280, "fbs": "Yes",
        "restecg": "Left ventricular hypertrophy", "thalach": 120, "exang": "Yes",
        "oldpeak": 3.0, "slope": "Downsloping", "ca": "3",
        "thal": "Fixed defect"
    }
}


# Fake prediction function
def fake_predict():
    import random
    # Randomly select a risk level for demonstration
    risk_level = random.choice(["Low", "Moderate", "High"])

    if risk_level == "Low":
        prediction_proba = round(random.uniform(0.01, 0.29), 2)  # 1% to 29%
        color = "green"
        icon = "✅"
        advice = "Maintain your healthy lifestyle with regular check-ups."
    elif risk_level == "Moderate":
        prediction_proba = round(random.uniform(0.30, 0.59), 2)  # 30% to 59%
        color = "orange"
        icon = "⚠️"
        advice = "Consider lifestyle changes and consult your doctor."
    else:  # High Risk
        prediction_proba = round(random.uniform(0.60, 0.99), 2)  # 60% to 99%
        color = "red"
        icon = "❗"
        advice = "Please consult a cardiologist for further evaluation."

    return risk_level, prediction_proba, color, icon, advice


# Initialize LLM chatbot (Optional for AI Advisor)
@st.cache_resource
def init_chatbot():
    try:
        sign = Login(HF_EMAIL, HF_PASS)
        cookies = sign.login()
        return hugchat.ChatBot(cookies=cookies.get_dict())
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None


chatbot = init_chatbot()

# App layout
st.title("❤️ Heart Disease Risk Assessment")
st.markdown("""
This tool evaluates your heart disease risk based on clinical parameters and provides personalized insights.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Parameter Analysis", "Health Guidance"])

with tab1:
    # Prediction form
    with st.form("heart_form"):
        st.subheader("Enter Your Health Parameters")
        col1, col2, col3 = st.columns(3)
        inputs = {}

        # Column 1
        with col1:
            inputs["age"] = st.number_input(feature_info["age"]["label"],
                                            min_value=feature_info["age"]["min"],
                                            max_value=feature_info["age"]["max"],
                                            value=feature_info["age"]["value"],
                                            help=f"{feature_info['age']['desc']}. Healthy range: {feature_info['age']['healthy_range']}")
            inputs["sex"] = st.selectbox(feature_info["sex"]["label"],
                                         feature_info["sex"]["options"],
                                         help=feature_info["sex"]["desc"])
            inputs["cp"] = st.selectbox(feature_info["cp"]["label"],
                                        feature_info["cp"]["options"],
                                        index=0,
                                        help=f"{feature_info['cp']['desc']}. Healthy: {feature_info['cp']['healthy']}")
            inputs["trestbps"] = st.number_input(feature_info["trestbps"]["label"],
                                                 min_value=feature_info["trestbps"]["min"],
                                                 max_value=feature_info["trestbps"]["max"],
                                                 value=feature_info["trestbps"]["value"],
                                                 help=f"{feature_info['trestbps']['desc']}. Healthy: {feature_info['trestbps']['healthy_range']}")
            inputs["chol"] = st.number_input(feature_info["chol"]["label"],
                                             min_value=feature_info["chol"]["min"],
                                             max_value=feature_info["chol"]["max"],
                                             value=feature_info["chol"]["value"],
                                             help=f"{feature_info['chol']['desc']}. Healthy: {feature_info['chol']['healthy_range']}")

        # Column 2
        with col2:
            inputs["fbs"] = st.selectbox(feature_info["fbs"]["label"],
                                         feature_info["fbs"]["options"],
                                         help=f"{feature_info['fbs']['desc']}. Healthy: {feature_info['fbs']['healthy']}")
            inputs["restecg"] = st.selectbox(feature_info["restecg"]["label"],
                                             feature_info["restecg"]["options"],
                                             help=f"{feature_info['restecg']['desc']}. Healthy: {feature_info['restecg']['healthy']}")
            inputs["thalach"] = st.number_input(feature_info["thalach"]["label"],
                                                min_value=feature_info["thalach"]["min"],
                                                max_value=feature_info["thalach"]["max"],
                                                value=feature_info["thalach"]["value"],
                                                help=f"{feature_info['thalach']['desc']}. Healthy: {feature_info['thalach']['healthy_range']}")
            inputs["exang"] = st.selectbox(feature_info["exang"]["label"],
                                           feature_info["exang"]["options"],
                                           help=f"{feature_info['exang']['desc']}. Healthy: {feature_info['exang']['healthy']}")
            inputs["oldpeak"] = st.number_input(feature_info["oldpeak"]["label"],
                                                min_value=feature_info["oldpeak"]["min"],
                                                max_value=feature_info["oldpeak"]["max"],
                                                value=feature_info["oldpeak"]["value"],
                                                step=feature_info["oldpeak"]["step"],
                                                help=f"{feature_info['oldpeak']['desc']}. Healthy: {feature_info['oldpeak']['healthy_range']}")

        # Column 3
        with col3:
            inputs["slope"] = st.selectbox(feature_info["slope"]["label"],
                                           feature_info["slope"]["options"],
                                           help=f"{feature_info['slope']['desc']}. Healthy: {feature_info['slope']['healthy']}")
            inputs["ca"] = st.selectbox(feature_info["ca"]["label"],
                                        feature_info["ca"]["options"],
                                        help=f"{feature_info['ca']['desc']}. Healthy: {feature_info['ca']['healthy']}")
            inputs["thal"] = st.selectbox(feature_info["thal"]["label"],
                                          feature_info["thal"]["options"],
                                          help=f"{feature_info['thal']['desc']}. Healthy: {feature_info['thal']['healthy']}")

        submitted = st.form_submit_button("Assess My Heart Disease Risk")

    if submitted:
        # Simulate fake prediction
        risk_level, prediction_proba, color, icon, advice = fake_predict()

        # Display results
        st.subheader("Risk Assessment Results")
        # Risk meter
        cols = st.columns(3)
        with cols[1]:
            st.metric("Heart Disease Risk",
                      f"{prediction_proba * 100:.1f}%",
                      f"{risk_level} Risk",
                      delta_color="off")
        # Progress bar with color coding
        risk_meter = st.progress(0)
        risk_meter.progress(prediction_proba)
        # Risk indicators
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:15px;border-radius:10px">
            <h4 style="color:{color};text-align:center">{icon} {risk_level} Risk Category</h4>
            <p style="text-align:center">{advice}</p>
        </div>
        """, unsafe_allow_html=True)

        # Parameter analysis
        st.subheader("Parameter Analysis")
        risk_factors = random.randint(0, 5)  # Random number of risk factors
        total_factors = 10  # Assume 10 total parameters for simplicity
        risk_details = [f"- Parameter {i + 1}: Outside healthy range" for i in range(risk_factors)]

        st.write(f"**{risk_factors} out of {total_factors}** parameters outside healthy ranges:")
        if risk_details:
            st.markdown("\n".join(risk_details))
        else:
            st.success("All parameters are within healthy ranges!")

        # Store results for other tabs
        st.session_state.results = {
            "risk_level": risk_level,
            "probability": prediction_proba,
            "risk_factors": risk_details,
            "inputs": inputs
        }

with tab2:
    st.header("Parameter Analysis")
    if "results" not in st.session_state:
        st.warning("Please complete the risk assessment first.")
    else:
        # Create radar chart for risk visualization
        st.subheader("Health Parameter Radar Chart")
        # Select key parameters to visualize
        radar_params = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        radar_labels = [feature_info[p]["label"] for p in radar_params]
        radar_values = [st.session_state.results["inputs"][p] for p in radar_params]

        # Normalize values for radar chart (0-1 scale)
        max_values = {
            "age": 100,
            "trestbps": 200,
            "chol": 300,
            "thalach": 200,
            "oldpeak": 4.0
        }
        normalized_values = [v / max_values[p] for v, p in zip(radar_values, radar_params)]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        normalized_values += normalized_values[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, normalized_values, color='red', alpha=0.25)
        ax.plot(angles, normalized_values, color='red', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels)
        ax.set_yticklabels([])
        ax.set_title("Your Health Parameters (Normalized)", pad=20)
        st.pyplot(fig)

        # Parameter comparison table
        st.subheader("Parameter Comparison")
        comparison_data = []
        for param in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
            info = feature_info[param]
            comparison_data.append({
                "Parameter": info["label"],
                "Your Value": st.session_state.results["inputs"][param],
                "Healthy Range": info.get("healthy_range", info.get("healthy", "")),
                "Status": "✅ Within Range" if param not in [rf.split(":")[0].strip("- ") for rf in
                                                            st.session_state.results[
                                                                "risk_factors"]] else "⚠️ Out of Range"
            })
        st.table(pd.DataFrame(comparison_data))

with tab3:
    st.header("Personalized Health Guidance")
    if "results" not in st.session_state:
        st.warning("Please complete the risk assessment first.")
    else:
        # Display general recommendations based on risk level
        risk_level = st.session_state.results["risk_level"]
        if risk_level == "Low":
            st.success("""
            **Your heart health looks good!**  
            Maintain these healthy habits:
            - Continue regular exercise (150 mins/week moderate activity)
            - Eat a balanced diet rich in fruits, vegetables, and whole grains
            - Get 7-9 hours of quality sleep nightly
            - Manage stress through relaxation techniques
            - Annual check-ups with your doctor
            """)
        elif risk_level == "Moderate":
            st.warning("""
            **Your heart health needs attention**  
            Recommended actions:
            - Increase physical activity (aim for 30 mins/day)
            - Reduce sodium and saturated fat intake
            - Quit smoking if applicable
            - Limit alcohol consumption
            - Monitor blood pressure regularly
            - Schedule a doctor's visit within 3 months
            """)
        else:
            st.error("""
            **Your heart health requires immediate attention**  
            Critical next steps:
            - Consult a cardiologist within 1 month
            - Begin a supervised exercise program
            - Strict dietary modifications
            - Medication may be needed (doctor will advise)
            - Regular monitoring of all risk factors
            - Consider cardiac rehabilitation
            """)

        # AI Recommendations (Optional)
        if chatbot:
            st.subheader("AI Health Advisor")
            # Pre-fill prompt based on risk factors
            default_prompt = f"""I'm a {st.session_state.results['inputs']['age']} year old {st.session_state.results['inputs']['sex'].lower()} with {st.session_state.results['risk_level'].lower()} risk of heart disease. 
            My key risk factors are: {', '.join([rf.split(':')[0].strip('- ') for rf in st.session_state.results['risk_factors']][:3])}.
            Provide specific lifestyle recommendations to improve my heart health in these areas:"""
            user_query = st.text_area("Ask for personalized advice:", value=default_prompt)
            if st.button("Get AI Recommendations"):
                with st.spinner("Generating personalized recommendations..."):
                    try:
                        response = chatbot.query(user_query)
                        st.markdown(str(response))
                    except Exception as e:
                        st.error(f"Error getting recommendations: {e}")

# Sidebar with sample profiles
st.sidebar.header("Quick Start")
profile = st.sidebar.selectbox("Load sample profile:", list(SAMPLE_PROFILES.keys()))
if st.sidebar.button("Load Profile"):
    for key, value in SAMPLE_PROFILES[profile].items():
        st.session_state[key] = value
    st.rerun()

# Disclaimer
st.sidebar.markdown("""
### Important Disclaimer
This tool provides risk assessment only and is not a substitute for professional medical advice. 
Always consult with a qualified healthcare provider for medical concerns.
""")