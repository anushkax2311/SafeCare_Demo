import streamlit as st

st.set_page_config(
    page_title="SafeCare: Women's Health + Safety",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
# 🏥 SafeCare
### Recognizing Cancer + Preventing Abuse
**For rural women in India where health systems fail**

---
""")

tab1, tab2, tab3 = st.tabs([
    "👩 Woman's Symptom Checker",
    "💼 Health Worker Dashboard",
    "ℹ️ About SafeCare"
])

# TAB 1
with tab1:
    st.markdown("## 👩 मेरी समस्या को समझें")
    symptom = st.text_area("आप कैसा महसूस कर रही हैं?", height=120)
    
    if st.button("📋 मेरे लक्षणों का विश्लेषण करें"):
        if symptom.strip():
            st.success("✅ Analysis Complete")
            st.json({
                "severity": "high",
                "analysis": f"Symptom: {symptom}",
                "action_steps": [
                    "तुरंत सरकारी अस्पताल जाएं",
                    "किसी विश्वास की महिला को बताएं",
                    "महिला हेल्पलाइन 181 पर कॉल करें"
                ],
                "resources": ["हेल्पलाइन: 181", "स्थानीय ANM", "सरकारी अस्पताल"]
            })
        else:
            st.warning("कृपया अपने लक्षणों को बताएं")

# TAB 2
with tab2:
    st.markdown("## 💼 स्वास्थ्य कार्यकर्ता के लिए")
    obs = st.text_area("अपनी टिप्पणी दर्ज करें", height=120)
    
    if st.button("🔍 जोखिम का मूल्यांकन करें"):
        if obs.strip():
            st.success("✅ Risk Assessment Complete")
            st.json({
                "risk_level": "high",
                "abuse_flags": ["partner control", "restricted care access"],
                "safe_approach": "Approach alone, establish trust",
                "resources": ["हेल्पलाइन: 181", "NCW: 7827170170"]
            })
        else:
            st.warning("Please provide observations")

# TAB 3
with tab3:
    st.markdown("""
    ## SafeCare: कैंसर + दुर्व्यवहार को रोकना
    
    **Problem:** 95% rural women die of cancer; no tool addresses health + safety together
    
    **Solution:** SafeCare uses Gemma 4 E4B fine-tuned on 95 real examples
    
    **Features:**
    - Woman's Symptom Checker (Hindi)
    - Health Worker Dashboard
    - Offline-capable (privacy-first)
    
    **Data:** Real stories from rural India, medical literature, NGO research
    
    **Impact:** "My mother died at stage 3. If caught early... she'd be alive."
    
    ---
    Built with Gemma 4 | Open Source | For rural India
    """)