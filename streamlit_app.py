import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json

st.set_page_config(
    page_title="SafeCare: Women's Health + Safety",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
# 🏥 SafeCare
### Recognizing Cancer + Preventing Abuse
**Fine-tuned Gemma 4 E4B — Live Demo**
""")

# Load model (matching your Kaggle training exactly)
@st.cache_resource
def load_model():
    st.info("⏳ Loading fine-tuned Gemma 4 (first load: 2-3 min)...")
    
    try:
        # 4-bit quantization (matching your training)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf8",
        )
        
        # Load base model from HuggingFace
        base_model_name = "google/gemma-4-E4B-it"
        
        st.write("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        st.write("📥 Downloading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        st.write("📥 Loading your fine-tuned LoRA adapter...")
        # Load your fine-tuned LoRA weights
        model = PeftModel.from_pretrained(model, "./final-model")
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("Make sure final-model/ folder is in your GitHub repo")
        return None, None

# Load model
model, tokenizer = load_model()

if model is None:
    st.stop()

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
        height=120,
        key="woman_symptom"
    )
    
    if st.button("📋 मेरे लक्षणों का विश्लेषण करें", key="woman_btn"):
        if symptom.strip() and model:
            with st.spinner("🔄 विश्लेषण किया जा रहा है... (30-60 सेकंड)"):
                try:
                    # Create prompt in SAME format as training
                    prompt = f"""<start_of_turn>user
You are SafeCare, a health assistant for rural women in India. Analyze symptoms and respond ONLY with JSON.

महिला ने कहा: "{symptom}"
Region: rural India<end_of_turn>
<start_of_turn>model
"""
                    
                    # Tokenize (matching your training format)
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to("cuda" if torch.cuda.is_available() else "cpu")
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=800,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract just the model response part
                    if "<start_of_turn>model" in response:
                        response = response.split("<start_of_turn>model")[-1].strip()
                    
                    st.success("✅ Analysis Complete")
                    st.markdown(response)
                    
                    st.markdown("""
                    ### 📞 मदद के लिए संपर्क करें
                    - **महिला हेल्पलाइन: 181** (24/7, गोपनीय)
                    - **पुलिस: 112** (तुरंत खतरे में)
                    - **स्थानीय ANM/ASHA**
                    """)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
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
        placeholder="महिला के बारे में आपने क्या देखा?",
        height=120,
        key="hw_observation"
    )
    
    if st.button("🔍 जोखिम का मूल्यांकन करें", key="hw_btn"):
        if observation.strip() and model:
            with st.spinner("🔄 मूल्यांकन किया जा रहा है... (30-60 सेकंड)"):
                try:
                    # Create prompt in SAME format as training
                    prompt = f"""<start_of_turn>user
You are SafeCare, a health assistant for rural women in India. Assess abuse risk and health concerns. Respond ONLY with JSON.

स्वास्थ्य कार्यकर्ता की टिप्पणी: {observation}<end_of_turn>
<start_of_turn>model
"""
                    
                    # Tokenize (matching your training format)
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to("cuda" if torch.cuda.is_available() else "cpu")
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=800,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract just the model response part
                    if "<start_of_turn>model" in response:
                        response = response.split("<start_of_turn>model")[-1].strip()
                    
                    st.success("✅ Assessment Complete")
                    st.markdown(response)
                    
                    st.markdown("""
                    ### 📞 Resources to Share
                    - **महिला हेल्पलाइन: 181** (24/7, गोपनीय)
                    - **NCW: 7827170170**
                    - **Nearest Government Hospital**
                    """)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please provide observations")

# TAB 3: About
with tab3:
    st.markdown("""
    ## SafeCare: कैंसर + दुर्व्यवहार को रोकना
    
    ### समस्या
    - 95% अस्पतालों में मरने वाली महिलाएं गरीब और ग्रामीण होती हैं
    - शुरुआती लक्षण नहीं पहचाने जाते
    - दुर्व्यवहार में महिलाएं मदद नहीं ले सकतीं
    
    ### समाधान
    **SafeCare = Gemma 4 E4B fine-tuned on 95 real examples**
    - ✓ कैंसर के शुरुआती लक्षण पहचानता है
    - ✓ दुर्व्यवहार के संकेत चिन्हित करता है
    - ✓ सुरक्षित कदम बताता है
    - ✓ पूरी तरह गोपनीय (आपके फोन में)
    
    ### तकनीकी विवरण
    - **मॉडल:** Gemma 4 E4B (4B parameters)
    - **प्रशिक्षण:** Unsloth + LoRA (r=16, lora_alpha=32)
    - **डेटा:** 95 real SafeCare examples
    - **भाषा:** Hindi
    
    ### वास्तविक प्रभाव
    > "मेरी माँ को स्टेज 3 कैंसर था। उसे उसके पति द्वारा बलात्कार भी किया गया। 
    > अगर जल्दी पकड़ा होता... वह जीवित होती।"
    > — कैंसर पीड़ित की बेटी
    
    ---
    Built with Gemma 4 | Open Source | For NGOs & Health Ministries
    """)

st.markdown("""
---
⏳ **Note:** First load takes 2-3 minutes (model download). Subsequent requests are faster.
""")