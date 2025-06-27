# predictor/trainer.py

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(__file__))

from base_models.lgbm_model import LGBMPredictor

import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ 依旧保留路径与特征列
FEATURES_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_features_v2.csv")
LABELS = ["sim_bai", "sim_shi", "sim_ge"]
MODEL_DIR = os.path.join(BASE_DIR, "models", "lgbm")

FEATURE_COLUMNS = [
    'sim_sum_val', 'sim_span',
    'open_sum_val', 'open_span',
    'match_count', 'match_pos_count',
    'sim_pattern_组三', 'sim_pattern_组六', 'sim_pattern_豹子'
]

def main():
    df = pd.read_csv(FEATURES_PATH).dropna()
    X = df[FEATURE_COLUMNS]

    for label_name in LABELS:
        y = df[label_name]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"🚀 开始训练 {label_name} ...")
        model = LGBMPredictor(label_name=label_name, model_dir=MODEL_DIR)
        model.train(X_train, y_train)

        preds = model.predict(X_val)
        acc = (preds == y_val).mean()
        print(f"✅ {label_name} 验证集准确率: {acc:.4f}")

if __name__ == "__main__":
    main()
