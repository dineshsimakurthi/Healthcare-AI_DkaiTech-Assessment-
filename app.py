"""
PART 3 - SIMPLE INTERFACE (Streamlit)
Patient info -> ML risk prediction -> RAG explanation with document sources.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import predict_risk          # noqa: E402
from rag_chain import ask, build_rag_chain  # noqa: E402

st.set_page_config(page_title="Diabetes Risk + RAG Assistant", page_icon="🩺")
st.title("🩺 Diabetes Risk Prediction + RAG Assistant")
st.caption("Demonstration prototype - not medical advice.")


@st.cache_resource            # build the chain once, not on every click
def get_chain():
    return build_rag_chain()


# ---------------- patient form
with st.form("patient"):
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)

    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 1, 120, 45)
    pregnancies = c2.number_input(
        "Pregnancies",
        0,
        20,
        0 if sex == "Male" else 1,
        disabled=sex == "Male",
    )
    glucose = c1.number_input("Glucose (mg/dL)", 0, 300, 140)
    bp = c2.number_input("Blood pressure (diastolic, mmHg)", 0, 200, 80)
    bmi = c1.number_input("BMI", 0.0, 70.0, 30.0)
    insulin = c2.number_input("Insulin (mu U/ml, 0 = unknown)", 0, 900, 0)
    skin = c1.number_input("Skin thickness (mm, 0 = unknown)", 0, 100, 0)
    dpf = c2.number_input("Diabetes pedigree function", 0.0, 2.5, 0.5)
    submitted = st.form_submit_button("Predict risk")

if submitted:
    patient = {"Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
               "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
               "DiabetesPedigreeFunction": dpf, "Age": age}

    # 1) ML prediction
    result = predict_risk(patient)
    st.subheader("ML prediction")
    st.caption(f"Patient sex: {sex}")
    st.metric("Diabetes probability", f"{result['probability']:.0%}")
    st.write(f"**Risk level:** {result['risk_level']}  |  **Prediction:** {result['prediction']}")

    if result["risk_level"] == "High":
        st.error("High diabetes risk: please consult a doctor immediately. Seek urgent medical care for severe thirst, frequent urination, vomiting, abdominal pain, confusion, or difficulty breathing.")

    st.caption("The current ML model does not use sex as an input, so this probability is the same for male and female patients.")

    # 2) RAG explanation
    sex_considerations = (
        "For a female patient, consider pregnancy-related history, gestational diabetes history, and sex-specific screening or warning signs only when supported by the documents."
        if sex == "Female"
        else "For a male patient, consider sex-specific screening or warning signs only when supported by the documents."
    )
    question = (f"A {age}-year-old {sex.lower()} patient has {result['risk_level'].lower()} diabetes risk "
                f"(glucose {glucose} mg/dL, BMI {bmi}). {sex_considerations} "
                "What lifestyle changes are recommended and when should they consult a doctor?")
    with st.spinner("Searching documents..."):
        rag = ask(get_chain(), question)
    st.subheader("Explanation (from documents)")
    st.write(rag["answer"])
    if rag["sources"]:
        st.caption("Sources: " + ", ".join(rag["sources"]))

# ---------------- free question box
st.divider()
q = st.text_input("Or ask any question, e.g. 'What are the symptoms of diabetes?'")
if q:
    rag = ask(get_chain(), q)
    st.write(rag["answer"])
    st.caption("Sources: " + (", ".join(rag["sources"]) or "none"))
