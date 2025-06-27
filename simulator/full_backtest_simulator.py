# simulator/full_backtest_simulator.py

import os
import sys
import pandas as pd

# 路径准备
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "predictor"))

from feature_engineering.feature_generator_v2 import FeatureGeneratorV2
from predictor.base_models.lgbm_model import LGBMPredictor

# 自动训练导入
from predictor import trainer

# 路径
HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")
HISTORY_ALL_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history_all.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_features_v2.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "lgbm")
RESULT_CSV = os.path.join(BASE_DIR, "result", "full_backtest_simulator.csv")

os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)

FEATURE_COLUMNS = [
    'sim_sum_val', 'sim_span',
    'open_sum_val', 'open_span',
    'match_count', 'match_pos_count',
    'sim_pattern_组三', 'sim_pattern_组六', 'sim_pattern_豹子'
]

def check_models():
    """
    检查3个模型是否存在，不存在则自动调用训练
    """
    bai = os.path.join(MODEL_DIR, "lgbm_sim_bai.pkl")
    shi = os.path.join(MODEL_DIR, "lgbm_sim_shi.pkl")
    ge = os.path.join(MODEL_DIR, "lgbm_sim_ge.pkl")

    if not (os.path.exists(bai) and os.path.exists(shi) and os.path.exists(ge)):
        print("⚠️ 检测到模型文件不存在，自动启动训练...")
        trainer.main()
    else:
        print("✅ 检测到模型文件已存在，无需重新训练。")

def main():
    check_models()

    df_all = pd.read_csv(HISTORY_ALL_PATH, dtype=str)
    correct_count = 0
    total_count = 0
    results = []

    while True:
        if not os.path.exists(HISTORY_PATH):
            print(f"❌ 未找到 {HISTORY_PATH}")
            break

        df_current = pd.read_csv(HISTORY_PATH, dtype=str)
        last_issue = df_current['issue'].max()

        df_next = df_all[df_all['issue'] > last_issue].sort_values('issue').head(1)
        if df_next.empty:
            print("✅ 所有可采集数据已完成模拟。")
            break

        next_row = df_next.iloc[0]
        df_current = pd.concat([df_current, df_next], ignore_index=True)
        df_current.to_csv(HISTORY_PATH, index=False)
        print(f"✅ 已模拟采集新一期: {next_row['issue']}")

        # 特征提取
        fg = FeatureGeneratorV2(HISTORY_PATH)
        fg.generate_and_save(FEATURES_PATH)

        df_features = pd.read_csv(FEATURES_PATH, dtype=str)
        last_row = df_features.iloc[-1]
        X_new = pd.DataFrame([last_row[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS).astype(float)

        bai_model = LGBMPredictor(label_name="sim_bai", model_dir=MODEL_DIR)
        shi_model = LGBMPredictor(label_name="sim_shi", model_dir=MODEL_DIR)
        ge_model = LGBMPredictor(label_name="sim_ge", model_dir=MODEL_DIR)

        pred_bai = bai_model.predict(X_new)[0]
        pred_shi = shi_model.predict(X_new)[0]
        pred_ge = ge_model.predict(X_new)[0]

        pred_code = f"{pred_bai}{pred_shi}{pred_ge}"
        true_code = str(next_row['sim_test_code']).zfill(3)

        is_hit = int(pred_code == true_code)
        correct_count += is_hit
        total_count += 1

        print(f"[{next_row['issue']}] 真:{true_code} | 预测:{pred_code} | {'✅ 命中' if is_hit else '❌ 未中'}")

        results.append({
            "issue": next_row['issue'],
            "pred_code": pred_code,
            "true_code": true_code,
            "hit": is_hit
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULT_CSV, index=False)

    print("\n📊 全流程模拟完成:")
    print(f"👉 模拟期数: {total_count}")
    print(f"👉 命中期数: {correct_count}")

    if total_count > 0:
        print(f"👉 命中率: {correct_count / total_count:.2%}")
    else:
        print("👉 没有新数据需要模拟，无需计算命中率。")

    print(f"✅ 结果已保存: {RESULT_CSV}")


if __name__ == "__main__":
    main()
