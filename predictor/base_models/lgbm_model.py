# predictor/base_models/lgbm_model.py

import lightgbm as lgb
import joblib
import os

class LGBMPredictor:
    def __init__(self, label_name, model_dir="models/lgbm"):
        self.label_name = label_name
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, f"lgbm_{label_name}.pkl")
        self.model = None

    def train(self, X_train, y_train):
        self.model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        print(f"✅ 模型保存: {self.model_path}")

    def predict(self, X):
        if self.model is None:
            self.model = joblib.load(self.model_path)
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            self.model = joblib.load(self.model_path)
        return self.model.predict_proba(X)
