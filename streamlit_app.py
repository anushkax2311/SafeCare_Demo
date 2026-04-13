import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# Page config
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

# Load model (cached)
@st.cache_resource
def load_model():
    st.info("🔄 Loading fine-tuned SafeCare model...")
    
    try:
        # Load base model
        model_name = "google/gemma-4-E4B-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter (your fine-tuned weights)
        model = PeftModel.from_pretrained(model, "./final-model")
        
        st.success("✅ Model loaded!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.warning("Make sure `final-model/` folder exists with your trained weights")
        return None, None

# Load model
model, tokenizer = load_model()

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "👩 Woman's Symptom Checker",
    "💼 Health Worker Dashboard",
    "ℹ️ About SafeCare"
])

# TAB 1: Woman's Interface
with tab1:
    st.markdown("""
    ## 👩 मेरी समस्या को समझें
    **अपने लक्षणों को हिंदी में बताएं**
    """)
    
    symptom = st.text_area(
        "आप कैसा महसूस कर रही हैं?",
        placeholder="उदाहरण: मुझे 3 महीने से bleeding हो रही है, पेट में दर्द है...",
        height=120
    )
    
    if st.button("📋 मेरे लक्षणों का विश्लेषण करें"):
        if symptom.strip() and model:
            with st.spinner("🔄 विश्लेषण किया जा रहा है..."):
                prompt = f"""महिला ने कहा: "{symptom}"

Respond in JSON:
{{"severity": "high/medium/low", "analysis": "analysis in hindi", "actions": ["step1", "step2"]}}"""
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=500,
                        temperature=0.7,
                        do_sample=True
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.success("✅ Analysis Complete")
                st.write(response)
                
                # Resources
                st.markdown("""
                ### 📞 मदद के लिए संपर्क करें
                - **महिला हेल्पलाइन: 181** (24/7, गोपनीय)
                - **पुलिस: 112** (तुरंत खतरे में)
                - **स्थानीय ANM/ASHA**
                """)
        else:
            st.warning("कृपया अपने लक्षणों को विस्तार से बताएं")

# TAB 2: Health Worker Interface
with tab2:
    st.markdown("""
    ## 💼 स्वास्थ्य कार्यकर्ता के लिए
    **एक महिला को पहचानें, सुरक्षित रूप से संपर्क करें**
    """)
    
    observation = st.text_area(
        "अपनी टिप्पणी दर्ज करें",
        placeholder="महिला के बारे में आपने क्या देखा? (लक्षण, व्यवहार, परिवार की गतिविधि)",
        height=120
    )
    
    if st.button("🔍 जोखिम का मूल्यांकन करें"):
        if observation.strip() and model:
            with st.spinner("🔄 मूल्यांकन किया जा रहा है..."):
                prompt = f"""स्वास्थ्य कार्यकर्ता की टिप्पणी: {observation}

Risk assessment in JSON:
{{"risk_level": "high/medium/low", "abuse_flags": [], "safe_approach": "strategy"}}"""
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=500,
                        temperature=0.7,
                        do_sample=True
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.success("✅ Assessment Complete")
                st.write(response)
                
                st.markdown("""
                ### 📞 Resources to Share
                - **Women's Helpline: 181**
                - **NCW: 7827170170**
                - **Nearest Government Hospital**
                """)
        else:
            st.warning("Please provide detailed observations")

# TAB 3: About
with tab3:
    st.markdown("""
    ## SafeCare: कैंसर + दुर्व्यवहार को रोकना
    
    ### समस्या
    - 95% अस्पतालों में मरने वाली महिलाएं गरीब और ग्रामीण होती हैं
    - शुरुआती लक्षण नहीं पहचाने जाते (शर्म, साक्षरता की कमी)
    - दुर्व्यवहार में महिलाएं मदद नहीं ले सकतीं (पति का नियंत्रण)
    
    ### समाधान
    **SafeCare = Gemma 4 AI जो:**
    - ✓ कैंसर के शुरुआती लक्षण पहचानता है
    - ✓ दुर्व्यवहार के संकेत चिन्हित करता है
    - ✓ सुरक्षित कदम बताता है
    - ✓ पूरी तरह गोपनीय है (आपके फोन में)
    
    ### तकनीकी विवरण
    - **मॉडल:** Gemma 4 E4B (4B parameters)
    - **प्रशिक्षण:** 95 वास्तविक उदाहरणों पर
    - **विधि:** LoRA (0.15% trainable)
    - **भाषा:** Hindi
    """)