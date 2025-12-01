import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import pathlib
import sys
import tempfile
import os
from groq import Groq

# --- CONFIG ---
st.set_page_config(page_title="AgroAI", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

# --- PATHS ---
current_dir = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = current_dir.parent
src_dir = PROJECT_ROOT / 'src'
sys.path.append(str(src_dir))
sys.path.append(str(src_dir / 'risk_predictor'))


# --- HELPER ---
def get_latest_model_path():
    runs_dir = src_dir / "runs"
    if not runs_dir.exists(): return None, None
    candidates = [f for f in runs_dir.iterdir() if f.is_dir() and f.name.startswith("LeafDisease")]
    if not candidates: return None, None
    latest_run = max(candidates, key=lambda x: x.stat().st_mtime)
    return latest_run / "checkpoints" / "best_model.pth", latest_run / "class_mapping.json"


MODEL_PATH, CLASS_MAP_PATH = get_latest_model_path()

# --- IMPORTS ---
try:
    from cnn_model import LeafDiseaseResNet18
    from risk_model import RiskPredictionModel
except ImportError:
    st.error("⚠️ Setup Error: Check 'src' folder.")
    st.stop()


# --- LOADERS ---
@st.cache_resource
def load_models():
    if not MODEL_PATH or not MODEL_PATH.exists(): return None, None, None
    device = torch.device('cpu')
    try:
        cnn = LeafDiseaseResNet18(num_classes=16, pretrained=False)
        ckpt = torch.load(MODEL_PATH, map_location=device)
        cnn.load_state_dict(ckpt['model_state_dict'])
        cnn.eval()
        with open(CLASS_MAP_PATH, 'r') as f:
            classes = json.load(f)
        risk_path = src_dir / "runs" / "risk_model"
        risk = RiskPredictionModel(str(risk_path)) if risk_path.exists() else None
        return cnn, classes, risk
    except:
        return None, None, None


cnn_model, class_names, risk_model = load_models()


def get_groq_client():
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except:
        return None


def get_ai_plan(condition, score, plan_type="disease"):
    client = get_groq_client()
    if not client: return "⚠️ AI Doctor unavailable."
    clean_condition = condition.replace("RISK", "").strip()
    if plan_type == "risk":
        prompt = f"Act as expert agronomist. Plant has **{score:.0f}% risk** of **{clean_condition}**. Provide preventive strategy (Analysis, Actions, Monitoring)."
    else:
        prompt = f"Act as plant doctor. Diagnosis: **{condition}** (**{score:.0f}%**). Provide treatment plan (Insight, Organic, Chemical, Prevention)."
    try:
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600
        ).choices[0].message.content
    except:
        return "⚠️ AI Service Error."


def chat_with_ai_doctor(context, user_question):
    client = get_groq_client()
    if not client: return "Connection failed."
    try:
        return client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Context: {context}. User: {user_question}. Answer briefly."}],
            max_tokens=200
        ).choices[0].message.content
    except:
        return "Thinking error."


def preprocess_image(image):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(image).unsqueeze(0)


# --- MODAL ---
@st.dialog("📋 AI Health Report")
def show_treatment_plan(content):
    st.markdown(content)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: st.download_button("💾 Save", content, "report.md", "text/markdown", use_container_width=True)
    with col2:
        if st.button("❌ Close", use_container_width=True): st.rerun()


# --- MAIN UI ---
st.title("🌿 AgroAI Plant Health")
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Detector", "🔮 Risk", "📚 Guide", "ℹ️ About"])

# --- FLOATING CHATBOT (Fixed) ---
# Inject CSS to style the native st.popover button
st.markdown("""
<style>
    /* Target the popover button */
    [data-testid="stPopover"] > button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #2E7D32;
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s;
    }
    [data-testid="stPopover"] > button:hover {
        transform: scale(1.1);
        background-color: #1b5e20;
    }
    /* Add Chat Icon */
    [data-testid="stPopover"] > button::before {
        content: "💬";
        font-size: 30px;
    }
    /* Hide default text if any */
    [data-testid="stPopover"] > button > div {
        display: none;
    }
    /* Styling the popover window itself */
    [data-testid="stPopoverBody"] {
        width: 350px !important;
        max-height: 500px !important;
        border-radius: 15px;
        border: 1px solid #ddd;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    /* Remove extra padding inside */
    .stChatMessage {
        padding: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Create the popover (This automatically handles the button and window state)
chat_popover = st.popover("💬")  # Text is hidden by CSS above, but required

with chat_popover:
    st.markdown("### 🤖 AI Doctor")
    st.markdown("---")

    # Chat History Container
    chat_container = st.container(height=400)

    # Initialize History
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help?"}]

    # Display Messages
    with chat_container:
        for msg in st.session_state.messages:
            avatar = "🤖" if msg["role"] == "assistant" else "👨‍🌾"
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

    # Input
    if prompt := st.chat_input("Type here...", key="popover_chat"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            st.chat_message("user", avatar="👨‍🌾").write(prompt)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    ctx = st.session_state.get("detected_disease", "general")
                    response = chat_with_ai_doctor(ctx, prompt)
                    st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- DETECTOR TAB ---
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📸 Input")
        with st.container(border=True):
            with st.form("detect_form", clear_on_submit=False):
                img_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png"], label_visibility="collapsed")
                submitted = st.form_submit_button("🔍 Diagnose", use_container_width=True, type="primary")

            if img_file:
                st.image(img_file, use_container_width=True)
            else:
                # Empty state placeholder to keep layout stable
                st.markdown("<br><br><br>", unsafe_allow_html=True)

    with col2:
        st.subheader("🔍 Results")

        if submitted and img_file and cnn_model:
            with st.spinner("Analyzing..."):
                img = Image.open(img_file).convert('RGB')
                with torch.no_grad():
                    out = cnn_model(preprocess_image(img))
                    prob = torch.softmax(out, 1)
                    score, idx = torch.max(prob, 1)
                st.session_state.detected_disease = class_names.get(str(idx.item()), "Unknown").replace('_', ' ')
                st.session_state.detection_score = float(score.item() * 100)

        # Display Results
        if 'detected_disease' in st.session_state and img_file:
            d_name = st.session_state.detected_disease
            d_score = st.session_state.detection_score

            with st.container(border=True):
                st.markdown(f"### **{d_name}**")
                st.progress(d_score / 100, text=f"Confidence: {d_score:.1f}%")

                if d_score > 75:
                    st.success("Status: **High Confidence Detection**")
                else:
                    st.warning("Status: **Potential Match**")

                st.markdown("---")
                if st.button("📋 View Treatment Plan", use_container_width=True):
                    with st.spinner("Generating..."):
                        plan = get_ai_plan(d_name, d_score, "disease")
                        show_treatment_plan(plan)
        else:
            # Custom Empty State
            st.markdown("""
            <div style="
                border: 2px dashed #e0e0e0; 
                border-radius: 12px; 
                padding: 40px; 
                text-align: center; 
                background-color: #f8f9fa;
                color: #6c757d;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            ">
                <div style="font-size: 40px; margin-bottom: 10px;">👈</div>
                <h3 style="margin: 0; color: #343a40;">Start Diagnosis</h3>
                <p style="margin-top: 10px; font-size: 16px;">
                    Upload a leaf image on the left to begin analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)

# --- RISK TAB ---
with tab2:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📸 Input")
        with st.container(border=True):
            with st.form("risk_form", clear_on_submit=False):
                risk_file = st.file_uploader("Upload Healthy Leaf", type=["jpg", "png"], key="risk_up",
                                             label_visibility="collapsed")
                risk_sub = st.form_submit_button("🔮 Predict Risk", use_container_width=True, type="primary")

            if risk_file:
                st.image(risk_file, use_container_width=True)
            else:
                st.markdown("<br><br><br>", unsafe_allow_html=True)

    with col2:
        st.subheader("🔮 Forecast")

        if risk_sub and risk_file and risk_model:
            with st.spinner("Calculating..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(risk_file.getbuffer())
                    tpath = tmp.name
                try:
                    score, level, _ = risk_model.predict_risk(tpath)
                    clean = level.lower().replace("risk", "").strip().upper()
                    st.session_state.risk_level = f"{clean} RISK"
                    st.session_state.risk_clean = clean
                    st.session_state.risk_score = float(score)
                finally:
                    os.unlink(tpath)

        if 'risk_level' in st.session_state and risk_file:
            r_level = st.session_state.risk_level
            r_score = st.session_state.risk_score

            with st.container(border=True):
                st.markdown(f"### **{r_level}**")
                st.progress(r_score / 100, text=f"Risk Score: {r_score:.1f}%")

                color = "green" if r_score < 30 else "orange" if r_score < 70 else "red"
                st.markdown(f"Status: <b style='color:{color}'>{r_level}</b>", unsafe_allow_html=True)

                st.markdown("---")
                if st.button("🛡️ View Strategy", use_container_width=True):
                    with st.spinner("Generating..."):
                        plan = get_ai_plan(st.session_state.risk_clean, r_score, "risk")
                        show_treatment_plan(plan)
        else:
            st.markdown("""
            <div style="
                border: 2px dashed #e0e0e0; 
                border-radius: 12px; 
                padding: 40px; 
                text-align: center; 
                background-color: #f8f9fa;
                color: #6c757d;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            ">
                <div style="font-size: 40px; margin-bottom: 10px;">👈</div>
                <h3 style="margin: 0; color: #343a40;">Check Risk</h3>
                <p style="margin-top: 10px; font-size: 16px;">
                    Upload a healthy leaf on the left to predict future risks.
                </p>
            </div>
            """, unsafe_allow_html=True)

# GUIDE TAB
with tab3:
    st.header("Reference Guide")
    st.markdown("Learn about common crop diseases.")
    st.markdown("---")

    st.markdown("### Early Blight (Potato/Tomato)")
    with st.expander("🦠 Click to expand"):
        st.markdown("*Symptoms:* Dark brown spots with concentric rings on older leaves.")
        st.markdown("*Cause:* Fungus Alternaria solani.")
        st.markdown("*Management:* Crop rotation, remove infected debris, use fungicides.")

    st.markdown("### Late Blight (Potato/Tomato)")
    with st.expander("🦠 Click to expand"):
        st.markdown("*Symptoms:* Large water-soaked spots, white fungal growth on undersides.")
        st.markdown("*Cause:* Oomycete Phytophthora infestans.")
        st.markdown("*Management:* Avoid overhead watering, use resistant varieties, copper fungicides.")

    st.markdown("### Bacterial Spot (Pepper/Tomato)")
    with st.expander("🦠 Click to expand"):
        st.markdown("*Symptoms:* Small water-soaked spots turning brown/black.")
        st.markdown("*Cause:* Bacteria Xanthomonas species.")
        st.markdown("*Management:* Pathogen-free seeds, copper sprays, avoid wet fields.")

    st.markdown("### Tomato Leaf Mold")
    with st.expander("🦠 Click to expand"):
        st.markdown("*Symptoms:* Yellow spots on upper surface, mold on underside.")
        st.markdown("*Cause:* Fungus Passalora fulva.")
        st.markdown("*Management:* Improve air circulation, reduce humidity.")

    st.info("✨ More diseases coming soon...")


# --- ABOUT TAB ---
with tab4:
    st.header("About AgroAI")

    with st.container(border=True):
        st.subheader("🌟 Project Overview")
        st.markdown("""
        *AgroAI* is an advanced deep learning system designed to assist farmers and agronomists in early plant disease detection. 
        By leveraging *Computer Vision (CNNs)* and *Machine Learning (XGBoost)*, it analyzes leaf images to identify 
        diseases with *98.2% accuracy* and predicts future infection risks based on subtle visual cues.

        Integrated with a *Groq-powered AI Doctor*, it provides real-time, actionable treatment plans and preventive strategies.
        """)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):
            st.subheader("🛠 Tech Stack & Models")
            st.markdown("""
            *1. Disease Detection Model (CNN)*
            *   *Architecture:* ResNet18 (Pre-trained on ImageNet)
            *   *Framework:* PyTorch
            *   *Performance:* 98.2% Accuracy on Test Set
            *   *Technique:* Transfer Learning with Fine-tuning

            *2. Risk Prediction Model (ML)*
            *   *Algorithm:* XGBoost Ensemble Classifier
            *   *Features:* GLCM (Texture), HSV (Color), LBP (Patterns)
            *   *Purpose:* Early risk assessment for "healthy-looking" leaves

            *3. AI Doctor (LLM)*
            *   *Model:* Llama-3.3-70b-versatile (via Groq API)
            *   *Role:* Generates instant treatment & prevention advice

            *4. Frontend & Deployment*
            *   *Framework:* Streamlit (Python)
            *   *Interface:* Reactive UI with Floating Chatbot
            """)

    with col2:
        with st.container(border=True):
            st.subheader("🚀 How It Works")
            st.markdown("""
            1.  *Image Upload:* User uploads a leaf image.
            2.  *Preprocessing:* Image is resized (224x224) and normalized.
            3.  *Feature Extraction:*
                *   CNN extracts deep visual features for classification.
                *   Texture Analysis extracts statistical patterns for risk scoring.
            4.  *Inference:* Models predict the disease class or risk level.
            5.  *AI Consultation:* The diagnosis is sent to the LLM to generate a custom care plan.
            """)

    with st.container(border=True):
        st.subheader("🚧 Challenges & Solutions")
        st.markdown("""
        *   *Challenge:* Distinguishing between similar early-stage symptoms (e.g., Early Blight vs. Late Blight).
            *   Solution: Implemented *GLCM texture analysis* to capture micro-patterns invisible to the naked eye.
        *   *Challenge:* High latency in LLM responses.
            *   Solution: Integrated *Groq's LPU inference engine* for near-instant text generation.
        *   *Challenge:* Model overfitting on limited data.
            *   Solution: Used *Data Augmentation* (rotations, flips) and *Dropout layers* in the CNN.
        """)

    st.markdown("---")
    st.caption("© 2025 AgroAI Project | Developed by *Kunal Jha* | Data Science Major | Final Year Project")