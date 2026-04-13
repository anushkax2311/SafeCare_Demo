# SafeCare: Women's Health + Safety AI

Fine-tuned Gemma 4 model for cancer detection + abuse identification in rural India.

## Two Interfaces

### Woman's Symptom Checker
- Enter symptoms in Hindi
- Get severity assessment
- Receive safe action steps
- Access emergency resources

### Health Worker Dashboard
- Note patient observations
- Get risk assessment
- Receive safe conversation scripts
- Find intervention resources

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Visit: http://localhost:8501

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Connect to Streamlit Cloud
3. Get public URL

## Model

Fine-tuned on 95 real examples from SafeCare dataset.
- Base: google/gemma-4-E4B-it
- Adapter: LoRA (0.15% trainable params)