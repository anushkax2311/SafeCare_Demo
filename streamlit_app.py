import streamlit as st
import google.generativeai as genai
import json

st.set_page_config(page_title="SafeCare", page_icon="🏥", layout="wide")
st.markdown("# 🏥 SafeCare\n### Cancer Detection + Abuse Identification for Rural Women")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = """You are SafeCare, an AI health assistant for rural women in India.
You identify early cancer symptoms AND detect signs of abuse/control that prevent women from seeking care.
Always respond in valid JSON only, no extra text:
{
  "severity": "high/medium/low",
  "symptom_analysis": "simple explanation",
  "abuse_risk_flags": ["list or empty"],
  "action_steps": ["step1", "step2", "step3"],
  "safe_resources": ["Mahila Helpline: 181", "NCW: 7827170170"]
}"""

tab1, tab2, tab3 = st.tabs([
    "👩 Woman's Symptom Checker",
    "💼 Health Worker Dashboard",
    "ℹ️ About SafeCare"
])

with tab1:
    st.markdown("## अपने लक्षणों को बताएं")
    symptom = st.text_area(
        "आप कैसा महसूस कर रही हैं?",
        placeholder="Mujhe 3 mahine se bleeding ho rahi hai, pet mein dard hai...",
        height=120
    )
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region", ["Bihar","Jharkhand","UP","West Bengal","Rajasthan","Haryana"])
    with col2:
        barrier = st.selectbox("Barrier?", ["None","Husband won't allow","No money","Far from doctor","Social stigma"])

    if st.button("📋 विश्लेषण करें", type="primary"):
        if symptom.strip():
            with st.spinner("Analyzing..."):
                prompt = f"""{SYSTEM_PROMPT}

महिला ने कहा: "{symptom}"
Region: {region}
Barrier: {barrier}"""

                response = model.generate_content(prompt)
                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                try:
                    data = json.loads(raw)
                    sev = data.get("severity","medium")
                    color = {"high":"🔴","medium":"🟡","low":"🟢"}.get(sev,"🟡")

                    st.markdown(f"### {color} Severity: {sev.upper()}")
                    st.info(f"**Analysis:** {data.get('symptom_analysis','')}")

                    if data.get("abuse_risk_flags"):
                        st.warning(f"⚠️ Safety concern: {', '.join(data['abuse_risk_flags'])}")

                    st.markdown("### Action Steps")
                    for i, step in enumerate(data.get("action_steps",[]), 1):
                        st.markdown(f"**{i}.** {step}")

                    st.markdown("### 📞 Help")
                    for r in data.get("safe_resources",[]):
                        st.markdown(f"- {r}")

                except json.JSONDecodeError:
                    st.markdown(raw)
        else:
            st.warning("Please describe your symptoms.")

with tab2:
    st.markdown("## Health Worker Dashboard")
    observation = st.text_area(
        "Patient observations",
        placeholder="Woman missed 3 appointments. Husband always present. Visible bruises.",
        height=120
    )

    if st.button("🔍 Assess Risk", type="primary"):
        if observation.strip():
            with st.spinner("Assessing..."):
                prompt = f"""{SYSTEM_PROMPT}

Health worker observation: {observation}
Assess both health AND abuse risk."""

                response = model.generate_content(prompt)
                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]

                try:
                    data = json.loads(raw)
                    sev = data.get("severity","medium")
                    color = {"high":"🔴","medium":"🟡","low":"🟢"}.get(sev,"🟡")

                    st.markdown(f"### {color} Risk Level: {sev.upper()}")
                    st.info(data.get("symptom_analysis",""))

                    if data.get("abuse_risk_flags"):
                        st.error(f"🚨 Flags: {', '.join(data['abuse_risk_flags'])}")

                    st.markdown("### Recommended Actions")
                    for i, step in enumerate(data.get("action_steps",[]), 1):
                        st.markdown(f"**{i}.** {step}")

                    st.markdown("### Resources")
                    for r in data.get("safe_resources",[]):
                        st.markdown(f"- {r}")

                except json.JSONDecodeError:
                    st.markdown(raw)

with tab3:
    st.markdown("""
    ## About SafeCare

    A woman died of stage 3 cancer while being subjected to marital rape. Her daughter witnessed it.
    SafeCare was built so this doesn't happen again.

    ### What it does
    - Recognizes early cancer symptoms in women's own words (Hindi)
    - Identifies when abuse is preventing care-seeking
    - Gives safe, culturally appropriate action steps
    - Works for 6 regions: Bihar, Jharkhand, UP, West Bengal, Rajasthan, Haryana

    ### Technical
    - Fine-tuned: Gemma 4 E4B via Unsloth + LoRA on 95 real examples
    - Base model: google/gemma-4-E4B-it
    - Training: 3 epochs, r=16, lora_alpha=32
    - Data: Cancer symptoms + abuse indicators from real NGO reports

    ### Help Resources
    - **Mahila Helpline: 181** (free, 24/7)
    - **NCW Helpline: 7827170170**
    - **Police: 112**
    """)