# predictor/batch_validator.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# 保证路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(__file__))

from base_models.lgbm_model import LGBMPredictor

FEATURES_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_features_v2.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "lgbm")
RESULT_PATH = os.path.join(BASE_DIR, "data", "batch_backtest.csv")

FEATURE_COLUMNS = [
    'sim_sum_val', 'sim_span',
    'open_sum_val', 'open_span',
    'match_count', 'match_pos_count',
    'sim_pattern_组三', 'sim_pattern_组六', 'sim_pattern_豹子'
]

os.makedirs(os.path.join(BASE_DIR, "result"), exist_ok=True)

def main(backtest_count=200):
    df = pd.read_csv(FEATURES_PATH).dropna()

    bai_model = LGBMPredictor(label_name="sim_bai", model_dir=MODEL_DIR)
    shi_model = LGBMPredictor(label_name="sim_shi", model_dir=MODEL_DIR)
    ge_model = LGBMPredictor(label_name="sim_ge", model_dir=MODEL_DIR)

    results = []

    total = min(backtest_count, len(df) - 1)

    for idx in range(len(df) - total, len(df)):
        row = df.iloc[idx]
        X = pd.DataFrame([row[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS)

        pred_bai = bai_model.predict(X)[0]
        pred_shi = shi_model.predict(X)[0]
        pred_ge = ge_model.predict(X)[0]

        true_bai = row['sim_bai']
        true_shi = row['sim_shi']
        true_ge = row['sim_ge']

        full_match = (pred_bai == true_bai) and (pred_shi == true_shi) and (pred_ge == true_ge)

        results.append({
            "issue": row['issue'],
            "true_code": f"{true_bai}{true_shi}{true_ge}",
            "pred_code": f"{pred_bai}{pred_shi}{pred_ge}",
            "hit_bai": int(pred_bai == true_bai),
            "hit_shi": int(pred_shi == true_shi),
            "hit_ge": int(pred_ge == true_ge),
            "hit_all": int(full_match)
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULT_PATH, index=False)
    print(f"✅ 回测结果已保存: {RESULT_PATH}")

    bai_acc = result_df['hit_bai'].mean()
    shi_acc = result_df['hit_shi'].mean()
    ge_acc = result_df['hit_ge'].mean()
    all_acc = result_df['hit_all'].mean()

    print("\n✅ 回测汇总：")
    print(f"百位命中率: {bai_acc:.2%}")
    print(f"十位命中率: {shi_acc:.2%}")
    print(f"个位命中率: {ge_acc:.2%}")
    print(f"组三位全中命中率: {all_acc:.2%}")

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(result_df['issue'], result_df['hit_bai'], label='百位')
    plt.plot(result_df['issue'], result_df['hit_shi'], label='十位')
    plt.plot(result_df['issue'], result_df['hit_ge'], label='个位')
    plt.plot(result_df['issue'], result_df['hit_all'], label='全中')
    plt.xlabel('期号')
    plt.ylabel('命中 (1=中)')
    plt.title(f'最近 {total} 期命中趋势')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(BASE_DIR, "result", "batch_backtest.png")
    plt.savefig(plot_path)
    print(f"✅ 命中趋势图已保存: {plot_path}")

    plt.show()

if __name__ == "__main__":
    main(backtest_count=200)
