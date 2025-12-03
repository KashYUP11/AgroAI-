# 🌿 AgroAI: Leaf Disease Detection & Risk Assessment

An advanced deep learning solution for early plant disease detection and predictive risk assessment. This project leverages **ResNet18 with CBAM Attention** for classification and **XGBoost** for analyzing subtle leaf texture changes to predict disease risk before it becomes visible.

## 🚀 Features

- **Deep Learning Detection**: Identifies 16+ specific plant diseases using a custom CNN architecture (ResNet18 + CBAM).
- **Early Risk Prediction**: Uses texture analysis (GLCM, LBP, HSV) and ML (XGBoost) to detect "at-risk" plants before symptoms appear.
- **AI Agronomist**: Integrated **Groq (LLaMA-3)** chatbot that provides real-time treatment plans and agricultural advice.
- **Interactive Dashboard**: A user-friendly web interface built with **Streamlit**.

## 🎥 Project Demo
Watch the full working demo of the project:

[![Watch the video](https://drive.google.com/file/d/1yNZxF1x3YQpuujKg63E2XjMxUunlvwlL/view?usp=drive_link)](https://drive.google.com/file/d/1XKnGsfV2LfLO5jFg8QwcIfAzK_x-sMxh/view?usp=drive_link)

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Torchvision, ResNet18, CBAM Attention
- **Machine Learning**: XGBoost, Scikit-learn
- **Computer Vision**: OpenCV, Scikit-image (GLCM, LBP)
- **Web Framework**: Streamlit
- **LLM Integration**: Groq API (LLaMA-3-70b)

## 📂 Project Structure

```
AgroAI-/
├── src/
│   ├── cnn_model.py          # ResNet18 with CBAM attention
│   ├── plant_dataset.py      # Data loading & augmentation
│   ├── train.py              # Training script
│   ├── fine_tune.py          # Fine-tuning script
│   ├── test_model.py         # Model testing
│   ├── test_loader.py        # Data loader tests
│   ├── build_risk_dataset.py # Risk dataset builder
│   └── risk_predictor/       # XGBoost risk model
├── webapp/
│   └── app.py                # Streamlit web interface
├── data/                     # PlantVillage & Kaggle datasets (not in repo)
├── runs/                     # Training checkpoints (not in repo)
├── .streamlit/
│   └── secrets.toml          # API keys (not in repo)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```


## ⚙️ Installation

### 1. Clone the repository
git clone https://github.com/KashYUP11/AgroAI-.git

cd AgroAI-

### 2. Install dependencies
pip install -r requirements.txt


### 3. Setup API Keys

Create a file named `secrets.toml` inside the `.streamlit` folder:

**`.streamlit/secrets.toml`:**
GROQ_API_KEY = "your_actual_api_key_here"


### 4. Download Dataset

Place the PlantVillage dataset in `data/PlantVillage/`

For the risk detection model:
Data is taken from - "https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification"

## 🔄 Project Workflow

The development of AgroAI followed a **Model-First Architecture**, prioritizing robust predictive capabilities before interface development:

1.  **Model Training & Validation**:
    *   The CNN (ResNet18 + CBAM) was trained first on the PlantVillage dataset to ensure high classification accuracy.
    *   Simultaneously, the XGBoost risk prediction model was trained on texture features (GLCM, HSV, LBP) to distinguish between healthy and "at-risk" leaves.
    *   Rigorous testing and hyperparameter tuning were conducted to finalize model weights (`best_model.pth` and risk model checkpoints).

2.  **Backend Integration**:
    *   Inference scripts were developed to load trained models and process live inputs.
    *   The Groq API was integrated to provide LLM-based agricultural advice.

3.  **User Interface Construction**:
    *   Once the core AI components were fully functional, the Streamlit web interface was built to wrap these models into an accessible tool for end-users.

## 📊 Model Performance

### 1. Disease Detection (CNN)
*   **Architecture**: ResNet18 + CBAM Attention
*   **Test Accuracy**: **98.2%**
*   **Weighted F1-Score**: **98.6%**
*   **Inference Time**: ~30ms (GPU), ~120ms (CPU)

### 2. Risk Prediction (XGBoost)
*   **Model**: XGBoost Classifier trained on GLCM, LBP, and HSV texture features
*   **Test Set Accuracy**: **94.64%**
*   **Recall**: **94.69%**
*   **F1-Score**: **94.58%**
*   **ROC-AUC Score**: **0.9905**
*   **Top Predictive Features**: Texture contrast and homogeneity (Feature 133, 199)

## 🏃 Usage

### Run the Web Application

streamlit run webapp/app.py

Then open: http://localhost:8501

### Train the CNN Model
python src/train.py

### Fine-tune the Model
python src/fine_tune.py

## 📄 License

This project is for academic and educational purposes.

## 👨‍💻 Author

**Kunal Jha**  
University of Petroleum & Energy Studies (UPES)

## 🙏 Acknowledgments

- PlantVillage for the disease dataset
- Kaggle for the CSAFRIT dataset
- Groq for the LLM API integration
- PyTorch and scikit-learn communities

---

**⭐ Star this repo if you find it useful!**
