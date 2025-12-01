# 🌿 AgroAI: Leaf Disease Detection & Risk Assessment

An advanced deep learning solution for early plant disease detection and predictive risk assessment. This project leverages **ResNet18 with CBAM Attention** for classification and **XGBoost** for analyzing subtle leaf texture changes to predict disease risk before it becomes visible.

## 🚀 Features
- **Deep Learning Detection**: Identifies 16+ specific plant diseases using a custom CNN architecture (ResNet18 + CBAM).
- **Early Risk Prediction**: Uses texture analysis (GLCM, LBP, HSV) and ML (XGBoost) to detect "at-risk" plants before symptoms appear.
- **AI Agronomist**: Integrated **Groq (LLaMA-3)** chatbot that provides real-time treatment plans and agricultural advice.
- **Interactive Dashboard**: A user-friendly web interface built with **Streamlit**.

## 🛠️ Tech Stack
- **Deep Learning**: PyTorch, Torchvision, ResNet18, CBAM Attention
- **Machine Learning**: XGBoost, Scikit-learn
- **Computer Vision**: OpenCV, Scikit-image (GLCM, LBP)
- **Web Framework**: Streamlit
- **LLM Integration**: Groq API (LLaMA-3-70b)

## 📂 Project Structure

*   **LeafDisease-CNN/** (Main Project Folder)
    *   📂 **.streamlit/** - Contains `secrets.toml` (API keys)
    *   📂 **data/** - Your Kaggle and PlantVillage datasets
    *   📂 **notebooks/** - Jupyter notebooks for experiments
    *   📂 **plantdisease/** - **CNN Module** (ResNet18 model & training scripts)
    *   📂 **src/** - **Risk Module** (Feature extraction & XGBoost)
    *   📂 **webapp/** - **Frontend** (Streamlit website code)
    *   📂 **runs/** - Where model checkpoints and logs are saved
    *   📄 **requirements.txt** - List of libraries to install
    *   📄 **README.md** - This documentation file



## ⚙️ Installation

1. **Clone the repository**
git clone https://github.com/your-username/AgroAI.git
cd AgroAI


2. **Install dependencies**
pip install -r requirements.txt


3. **Setup API Keys**
Create a file named `secrets.toml` inside the `.streamlit` folder:
.streamlit/secrets.toml
GROQ_API_KEY = "your_actual_api_key_here"


## 🏃 Usage

**1. Run the Web Application:**
streamlit run webapp/app.py


**2. Train the CNN Model:**
python plantdisease/train.py


## 📄 License
This project is for academic and educational purposes.
