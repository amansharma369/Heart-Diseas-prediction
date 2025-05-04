import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hugchat import hugchat
from hugchat.login import Login

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
    "High": 1.0
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

def calculate_risk(inputs):
    """Calculate heart disease risk based on clinical parameters"""
    risk_factors = 0
    total_possible = 13  # Total parameters being evaluated
    risk_details = []
    risk_params = []

    if inputs['age'] > 60:
        risk_factors += 1
        risk_details.append(f"- Age: {inputs['age']} (Risk: 60+)")
        risk_params.append('age')
    if inputs['sex'] == 'Male':
        risk_factors += 1
        risk_details.append("- Sex: Male (Higher risk)")
        risk_params.append('sex')
    if inputs['cp'] == 'Asymptomatic':
        risk_factors += 1
        risk_details.append("- Chest Pain: Asymptomatic (Highest risk)")
        risk_params.append('cp')
    elif inputs['cp'] in ['Atypical angina', 'Non-anginal pain']:
        risk_factors += 0.5
        risk_details.append(f"- Chest Pain: {inputs['cp']} (Moderate risk)")
    if inputs['trestbps'] >= 140:
        risk_factors += 1
        risk_details.append(f"- Blood Pressure: {inputs['trestbps']} mmHg (Stage 2 Hypertension)")
        risk_params.append('trestbps')
    elif inputs['trestbps'] >= 130:
        risk_factors += 0.5
        risk_details.append(f"- Blood Pressure: {inputs['trestbps']} mmHg (Elevated)")
    if inputs['chol'] >= 240:
        risk_factors += 1
        risk_details.append(f"- Cholesterol: {inputs['chol']} mg/dL (High)")
        risk_params.append('chol')
    elif inputs['chol'] >= 200:
        risk_factors += 0.5
        risk_details.append(f"- Cholesterol: {inputs['chol']} mg/dL (Borderline High)")
    if inputs['fbs'] == 'Yes':
        risk_factors += 1
        risk_details.append("- Fasting Blood Sugar > 120 mg/dL")
        risk_params.append('fbs')
    if inputs['restecg'] == 'Left ventricular hypertrophy':
        risk_factors += 1
        risk_details.append("- ECG: Left Ventricular Hypertrophy")
        risk_params.append('restecg')
    elif inputs['restecg'] == 'ST-T wave abnormality':
        risk_factors += 0.5
        risk_details.append("- ECG: ST-T Wave Abnormality")
    if inputs['thalach'] < 120:
        risk_factors += 1
        risk_details.append(f"- Max Heart Rate: {inputs['thalach']} (Low)")
        risk_params.append('thalach')
    if inputs['exang'] == 'Yes':
        risk_factors += 1
        risk_details.append("- Exercise Induced Angina: Yes")
        risk_params.append('exang')
    if inputs['oldpeak'] >= 2.0:
        risk_factors += 1
        risk_details.append(f"- ST Depression: {inputs['oldpeak']} (High)")
        risk_params.append('oldpeak')
    elif inputs['oldpeak'] >= 1.0:
        risk_factors += 0.5
        risk_details.append(f"- ST Depression: {inputs['oldpeak']} (Moderate)")
    if inputs['slope'] == 'Downsloping':
        risk_factors += 1
        risk_details.append("- ST Slope: Downsloping (Highest risk)")
        risk_params.append('slope')
    elif inputs['slope'] == 'Flat':
        risk_factors += 0.5
        risk_details.append("- ST Slope: Flat (Moderate risk)")
    if inputs['ca'] == '3':
        risk_factors += 1
        risk_details.append("- Major Vessels: 3 (Highest risk)")
        risk_params.append('ca')
    elif inputs['ca'] in ['1', '2']:
        risk_factors += 0.5
        risk_details.append(f"- Major Vessels: {inputs['ca']} (Moderate risk)")
    if inputs['thal'] == 'Fixed defect':
        risk_factors += 1
        risk_details.append("- Thalassemia: Fixed Defect")
        risk_params.append('thal')
    elif inputs['thal'] == 'Reversible defect':
        risk_factors += 0.5
        risk_details.append("- Thalassemia: Reversible Defect")

    probability = min(risk_factors / total_possible, 0.99)  # Cap at 99%

    if probability <= RISK_THRESHOLDS['Low']:
        risk_level = 'Low'
        color = "green"
        icon = "✅"
        advice = "Maintain your healthy lifestyle with regular check-ups."
    elif probability <= RISK_THRESHOLDS['Moderate']:
        risk_level = 'Moderate'
        color = "orange"
        icon = "⚠️"
        advice = "Consider lifestyle changes and consult your doctor."
    else:
        risk_level = 'High'
        color = "red"
        icon = "❗"
        advice = "Please consult a cardiologist immediately."

    return risk_level, probability, color, icon, advice, risk_details, risk_params

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
    with st.form("heart_form"):
        st.subheader("Enter Your Health Parameters")
        col1, col2, col3 = st.columns(3)
        inputs = {}

        # Column 1
        with col1:
            inputs["age"] = st.number_input(
                feature_info["age"]["label"],
                min_value=feature_info["age"]["min"],
                max_value=feature_info["age"]["max"],
                value=st.session_state.get("age", feature_info["age"]["value"]),
                help=f"{feature_info['age']['desc']}. Healthy range: {feature_info['age']['healthy_range']}"
            )
            inputs["sex"] = st.selectbox(
                feature_info["sex"]["label"],
                feature_info["sex"]["options"],
                index=feature_info["sex"]["options"].index(st.session_state.get("sex", feature_info["sex"]["options"][0])),
                help=feature_info["sex"]["desc"]
            )
            inputs["cp"] = st.selectbox(
                feature_info["cp"]["label"],
                feature_info["cp"]["options"],
                index=feature_info["cp"]["options"].index(st.session_state.get("cp", feature_info["cp"]["options"][0])),
                help=f"{feature_info['cp']['desc']}. Healthy: {feature_info['cp']['healthy']}"
            )
            inputs["trestbps"] = st.number_input(
                feature_info["trestbps"]["label"],
                min_value=feature_info["trestbps"]["min"],
                max_value=feature_info["trestbps"]["max"],
                value=st.session_state.get("trestbps", feature_info["trestbps"]["value"]),
                help=f"{feature_info['trestbps']['desc']}. Healthy: {feature_info['trestbps']['healthy_range']}"
            )
            inputs["chol"] = st.number_input(
                feature_info["chol"]["label"],
                min_value=feature_info["chol"]["min"],
                max_value=feature_info["chol"]["max"],
                value=st.session_state.get("chol", feature_info["chol"]["value"]),
                help=f"{feature_info['chol']['desc']}. Healthy: {feature_info['chol']['healthy_range']}"
            )

        # Column 2
        with col2:
            inputs["fbs"] = st.selectbox(
                feature_info["fbs"]["label"],
                feature_info["fbs"]["options"],
                index=feature_info["fbs"]["options"].index(st.session_state.get("fbs", feature_info["fbs"]["options"][0])),
                help=f"{feature_info['fbs']['desc']}. Healthy: {feature_info['fbs']['healthy']}"
            )
            inputs["restecg"] = st.selectbox(
                feature_info["restecg"]["label"],
                feature_info["restecg"]["options"],
                index=feature_info["restecg"]["options"].index(st.session_state.get("restecg", feature_info["restecg"]["options"][0])),
                help=f"{feature_info['restecg']['desc']}. Healthy: {feature_info['restecg']['healthy']}"
            )
            inputs["thalach"] = st.number_input(
                feature_info["thalach"]["label"],
                min_value=feature_info["thalach"]["min"],
                max_value=feature_info["thalach"]["max"],
                value=st.session_state.get("thalach", feature_info["thalach"]["value"]),
                help=f"{feature_info['thalach']['desc']}. Healthy: {feature_info['thalach']['healthy_range']}"
            )
            inputs["exang"] = st.selectbox(
                feature_info["exang"]["label"],
                feature_info["exang"]["options"],
                index=feature_info["exang"]["options"].index(st.session_state.get("exang", feature_info["exang"]["options"][0])),
                help=f"{feature_info['exang']['desc']}. Healthy: {feature_info['exang']['healthy']}"
            )
            inputs["oldpeak"] = st.number_input(
                feature_info["oldpeak"]["label"],
                min_value=feature_info["oldpeak"]["min"],
                max_value=feature_info["oldpeak"]["max"],
                value=st.session_state.get("oldpeak", feature_info["oldpeak"]["value"]),
                step=feature_info["oldpeak"]["step"],
                help=f"{feature_info['oldpeak']['desc']}. Healthy: {feature_info['oldpeak']['healthy_range']}"
            )

        # Column 3
        with col3:
            inputs["slope"] = st.selectbox(
                feature_info["slope"]["label"],
                feature_info["slope"]["options"],
                index=feature_info["slope"]["options"].index(st.session_state.get("slope", feature_info["slope"]["options"][0])),
                help=f"{feature_info['slope']['desc']}. Healthy: {feature_info['slope']['healthy']}"
            )
            inputs["ca"] = st.selectbox(
                feature_info["ca"]["label"],
                feature_info["ca"]["options"],
                index=feature_info["ca"]["options"].index(st.session_state.get("ca", feature_info["ca"]["options"][0])),
                help=f"{feature_info['ca']['desc']}. Healthy: {feature_info['ca']['healthy']}"
            )
            inputs["thal"] = st.selectbox(
                feature_info["thal"]["label"],
                feature_info["thal"]["options"],
                index=feature_info["thal"]["options"].index(st.session_state.get("thal", feature_info["thal"]["options"][0])),
                help=f"{feature_info['thal']['desc']}. Healthy: {feature_info['thal']['healthy']}"
            )

        submitted = st.form_submit_button("Assess My Heart Disease Risk")

    if submitted:
        risk_level, prediction_proba, color, icon, advice, risk_details, risk_params = calculate_risk(inputs)

        st.subheader("Risk Assessment Results")
        cols = st.columns(3)
        with cols[1]:
            st.metric("Heart Disease Risk", f"{prediction_proba * 100:.1f}%", f"{risk_level} Risk", delta_color="off")
        risk_meter = st.progress(prediction_proba)
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:15px;border-radius:10px">
            <h4 style="color:{color};text-align:center">{icon} {risk_level} Risk Category</h4>
            <p style="text-align:center">{advice}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Risk Factor Analysis")
        total_factors = 13
        risk_factors = len(risk_details)
        st.write(f"**{risk_factors} out of {total_factors}** parameters show elevated risk:")
        if risk_details:
            st.markdown("\n".join(risk_details))
        else:
            st.success("All parameters are within healthy ranges!")

        st.session_state.results = {
            "risk_level": risk_level,
            "probability": prediction_proba,
            "risk_factors": risk_details,
            "risk_params": risk_params,
            "inputs": inputs
        }

with tab2:
    st.header("Parameter Analysis")
    if "results" not in st.session_state:
        st.warning("Please complete the risk assessment first.")
    else:
        st.subheader("Health Parameter Radar Chart")
        radar_params = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        radar_labels = [feature_info[p]["label"] for p in radar_params]
        radar_values = [st.session_state.results["inputs"][p] for p in radar_params]

        max_values = {
            "age": 100,
            "trestbps": 200,
            "chol": 600,
            "thalach": 220,
            "oldpeak": 6.0
        }
        normalized_values = [v / max_values[p] for v, p in zip(radar_values, radar_params)]
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        angles += angles[:1]
        normalized_values += normalized_values[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, normalized_values, color='red', alpha=0.25)
        ax.plot(angles, normalized_values, color='red', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels)
        ax.set_yticklabels([])
        ax.set_title("Your Health Parameters (Normalized)", pad=20)
        st.pyplot(fig)

        st.subheader("Parameter Comparison")
        comparison_data = []
        for param in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
            info = feature_info[param]
            status = "⚠️ Out of Range" if param in st.session_state.results["risk_params"] else "✅ Within Range"
            comparison_data.append({
                "Parameter": info["label"],
                "Your Value": st.session_state.results["inputs"][param],
                "Healthy Range": info.get("healthy_range", info.get("healthy", "")),
                "Status": status
            })
        st.table(pd.DataFrame(comparison_data))

with tab3:
    st.header("Personalized Health Guidance")
    if "results" not in st.session_state:
        st.warning("Please complete the risk assessment first.")
    else:
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
            - Reduce sodium (<1500mg/day) and saturated fat intake
            - Quit smoking if applicable
            - Limit alcohol to 1 drink/day (women) or 2/day (men)
            - Monitor blood pressure regularly
            - Schedule a doctor's visit within 3 months
            - Consider cholesterol screening
            """)
        else:
            st.error("""
            **Your heart health requires immediate attention**  
            Critical next steps:
            - Consult a cardiologist within 1 month
            - Begin a supervised exercise program
            - Strict dietary modifications (Mediterranean diet recommended)
            - Medication may be needed (doctor will advise)
            - Regular monitoring of blood pressure and cholesterol
            - Consider cardiac rehabilitation program
            - Emergency care if chest pain or shortness of breath occurs
            """)

# Sidebar with sample profiles
st.sidebar.header("Quick Start")
profile = st.sidebar.selectbox("Load sample profile:", list(SAMPLE_PROFILES.keys()))
if st.sidebar.button("Load Profile"):
    for key in list(st.session_state.keys()):
        if key in feature_info:
            del st.session_state[key]
    for key, value in SAMPLE_PROFILES[profile].items():
        st.session_state[key] = value
    st.rerun()

# Disclaimer
st.sidebar.markdown("""
### Important Disclaimer
This tool provides risk assessment only and is not a substitute for professional medical advice. 
Always consult with a qualified healthcare provider for medical concerns.
""")
