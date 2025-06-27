# predictor/predictor.py

import os
import sys
import pandas as pd
from datetime import datetime

# ✅ 保证直接执行能找到依赖
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(__file__))

from base_models.lgbm_model import LGBMPredictor
from utils.wechat_notify import send_wechat_template

FEATURES_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_features_v2.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "lgbm")
RESULT_PATH = os.path.join(BASE_DIR, "data", "next_predict_result.csv")

FEATURE_COLUMNS = [
    'sim_sum_val', 'sim_span',
    'open_sum_val', 'open_span',
    'match_count', 'match_pos_count',
    'sim_pattern_组三', 'sim_pattern_组六', 'sim_pattern_豹子'
]

def main():
    df = pd.read_csv(FEATURES_PATH, dtype=str).dropna()
    last_row = df.iloc[-1]

    current_issue = int(last_row['issue'])
    next_issue = current_issue + 1

    X_new = pd.DataFrame(
        [last_row[FEATURE_COLUMNS].values],
        columns=FEATURE_COLUMNS
    ).astype(float)

    bai_model = LGBMPredictor(label_name="sim_bai", model_dir=MODEL_DIR)
    shi_model = LGBMPredictor(label_name="sim_shi", model_dir=MODEL_DIR)
    ge_model = LGBMPredictor(label_name="sim_ge", model_dir=MODEL_DIR)

    bai = bai_model.predict(X_new)[0]
    shi = shi_model.predict(X_new)[0]
    ge = ge_model.predict(X_new)[0]

    sim_test_code_pred = f"{bai}{shi}{ge}"

    print(f"📌 当前最新期号: {current_issue}")
    print(f"🎯 推荐拼搏风格模拟试机号 (预测下一期 {next_issue}): {sim_test_code_pred}")

    # 从历史中提取最新真值
    df_his = pd.read_csv(HISTORY_PATH, dtype=str).dropna()
    last_real_row = df_his.iloc[-1]
    last_real_issue = last_real_row['issue']
    last_real_sim_test_code = str(last_real_row['sim_test_code']).zfill(3)

    print(f"📌 最新已开奖期号: {last_real_issue}")
    print(f"📌 最新已开奖模拟试机号: {last_real_sim_test_code}")

    # ✅ 写到CSV
    record = pd.DataFrame([{
        "predict_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predict_issue": next_issue,
        "predict_sim_test_code": sim_test_code_pred,
        "last_real_issue": last_real_issue,
        "last_real_sim_test_code": last_real_sim_test_code
    }])
    if os.path.exists(RESULT_PATH):
        record.to_csv(RESULT_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        record.to_csv(RESULT_PATH, mode="w", header=True, index=False, encoding="utf-8-sig")

    print(f"✅ 预测结果已记录到: {RESULT_PATH}")

    # ✅ 发送微信提醒
    send_wechat_template(
        to_users = [
            "oXUv66MibUi7VInLBf7AHqMIY438",
            "oXUv66DvDIoQG39Vnspwj97QVLn4",
            "oXUv66HUVNyZ0Hd8RWKmkVV1dkAs"
        ],
        title=f"拼搏3D 试机号预测提醒-v2",
        content1=f"预测期号：{next_issue}",
        content2=f"预测试机号：{sim_test_code_pred}",
        content3=f"上期试机号：{last_real_sim_test_code}",
        remark="请关注开奖走势，理性参考！"
    )

if __name__ == "__main__":
    main()
