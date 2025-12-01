import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
from pathlib import Path
from feature_extractor import AdvancedFeatureExtractor


class RiskPredictionModel:
    """
    XGBoost-based disease risk predictor
    Predicts if a leaf will develop disease (0-100% risk score)
    """

    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.extractor = AdvancedFeatureExtractor()
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def prepare_dataset(self, healthy_images_dir, at_risk_images_dir, test_size=0.2):
        """
        Prepare training dataset

        Args:
            healthy_images_dir: Path to folder with healthy leaf images
            at_risk_images_dir: Path to folder with early-stage diseased images
            test_size: Train-test split ratio
        """
        print("=" * 60)
        print("PREPARING RISK PREDICTION DATASET")
        print("=" * 60)

        X = []
        y = []

        # Load healthy images (Label: 0 = Low Risk)
        print(f"\n📂 Loading healthy images from: {healthy_images_dir}")
        healthy_files = list(Path(healthy_images_dir).glob('*.jpg')) + \
                        list(Path(healthy_images_dir).glob('*.png'))

        for i, img_path in enumerate(healthy_files):
            try:
                features = self.extractor.extract_all_features(str(img_path))
                X.append(features)
                y.append(0)  # Low risk
                if (i + 1) % 100 == 0:
                    print(f"  ✓ Processed {i + 1} healthy images")
            except Exception as e:
                print(f"  ✗ Error processing {img_path}: {e}")

        print(f"✅ Loaded {len([x for x in y if x == 0])} healthy images")

        # Load at-risk images (Label: 1 = High Risk)
        print(f"\n📂 Loading at-risk images from: {at_risk_images_dir}")
        at_risk_files = list(Path(at_risk_images_dir).glob('*.jpg')) + \
                        list(Path(at_risk_images_dir).glob('*.png'))

        for i, img_path in enumerate(at_risk_files):
            try:
                features = self.extractor.extract_all_features(str(img_path))
                X.append(features)
                y.append(1)  # High risk
                if (i + 1) % 100 == 0:
                    print(f"  ✓ Processed {i + 1} at-risk images")
            except Exception as e:
                print(f"  ✗ Error processing {img_path}: {e}")

        print(f"✅ Loaded {len([x for x in y if x == 1])} at-risk images")

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print(f"\n📊 Dataset Summary:")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Class distribution: {np.bincount(y)}")

        # Normalize features
        print(f"\n⚙️ Normalizing features...")
        X = self.scaler.fit_transform(X)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\n✅ Train set: {X_train.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost model for risk prediction
        """
        print("\n" + "=" * 60)
        print("TRAINING XGBOOST RISK PREDICTION MODEL")
        print("=" * 60)

        # XGBoost configuration (optimized for imbalanced data)
        params = {
            'max_depth': 7,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'random_state': 42,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'auc'
        }

        self.model = xgb.XGBClassifier(**params)

        print(f"\n🚀 Training model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=20
        )

        # Evaluate
        print(f"\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print(f"\n📊 Test Set Results:")
        print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
        print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
        print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

        # Feature importance
        print(f"\n🎯 Top 10 Most Important Features:")
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[-10:][::-1]
        for i, idx in enumerate(indices):
            print(f"  {i + 1}. Feature {idx}: {importance[idx]:.4f}")

    def predict_risk(self, image_path):
        """
        Predict disease risk for a single image

        Returns: risk_score (0-100%), risk_level, confidence
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded!")

        # Extract features
        features = self.extractor.extract_all_features(image_path)
        features = features.reshape(1, -1)
        features = self.scaler.transform(features)

        # Predict
        risk_score = self.model.predict_proba(features)[0][1] * 100
        confidence = max(self.model.predict_proba(features)[0]) * 100

        # Categorize risk
        if risk_score < 30:
            risk_level = "🟢 LOW RISK"
        elif risk_score < 70:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🔴 HIGH RISK"

        return risk_score, risk_level, confidence

    def save_model(self, save_path):
        """Save trained model"""
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_path, 'xgboost_risk_model.pkl'))
        joblib.dump(self.scaler, os.path.join(save_path, 'scaler.pkl'))
        print(f"✅ Model saved to: {save_path}")

    def load_model(self, model_path):
        """Load trained model"""
        self.model = joblib.load(os.path.join(model_path, 'xgboost_risk_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        print(f"✅ Model loaded from: {model_path}")


# Training script
if __name__ == "__main__":
    model = RiskPredictionModel()

    # ===== CONFIGURE YOUR DATA PATHS =====
    HEALTHY_DIR = r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\data\data_risk_kaggle\healthy"  # UPDATE THIS
    AT_RISK_DIR = r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\data\data_risk_kaggle\at_risk"  # UPDATE THIS

    # Prepare dataset
    X_train, X_test, y_train, y_test = model.prepare_dataset(
        HEALTHY_DIR,
        AT_RISK_DIR
    )

    # Train
    model.train(X_train, y_train, X_test, y_test)

    # Save
    model.save_model(r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\src\runs\risk_model")
