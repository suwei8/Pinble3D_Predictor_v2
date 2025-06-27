# feature_engineering/feature_generator_v2.py

import pandas as pd
from collections import Counter
import os

# ✅ 自动定位项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")
DEFAULT_SAVE_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_features_v2.csv")


class FeatureGeneratorV2:
    def __init__(self, history_path: str):
        self.history_path = history_path

    @staticmethod
    def parse_digits(code_str):
        code_str = str(code_str).zfill(3)
        return [int(d) for d in code_str]

    @staticmethod
    def get_pattern_type(digits):
        count = Counter(digits)
        if len(count) == 1:
            return '豹子'
        elif len(count) == 2:
            return '组三'
        else:
            return '组六'

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_rows = []

        for _, row in df.iterrows():
            # ⚡ 保证期号、号码是字符串且补 0
            issue = str(row["issue"]).zfill(7)  # 期号可保持7位或不补，看你需要
            sim_test_code = str(row["sim_test_code"]).zfill(3)
            open_code = str(row["open_code"]).zfill(3)

            sim_digits = self.parse_digits(sim_test_code)
            open_digits = self.parse_digits(open_code)

            sim_sum = sum(sim_digits)
            sim_span = max(sim_digits) - min(sim_digits)

            open_sum = sum(open_digits)
            open_span = max(open_digits) - min(open_digits)

            match_count = len(set(sim_digits) & set(open_digits))
            match_pos_count = sum([1 if s == o else 0 for s, o in zip(sim_digits, open_digits)])

            pattern = self.get_pattern_type(sim_digits)

            feature_rows.append({
                "issue": issue,
                "sim_test_code": sim_test_code,
                "open_code": open_code,
                "sim_sum_val": sim_sum,
                "sim_span": sim_span,
                "open_sum_val": open_sum,
                "open_span": open_span,
                "match_count": match_count,
                "match_pos_count": match_pos_count,
                "sim_pattern_组三": int(pattern == "组三"),
                "sim_pattern_组六": int(pattern == "组六"),
                "sim_pattern_豹子": int(pattern == "豹子"),
                "sim_bai": sim_digits[0],
                "sim_shi": sim_digits[1],
                "sim_ge": sim_digits[2]
            })

        return pd.DataFrame(feature_rows)

    def generate_and_save(self, save_path: str):
        if not os.path.exists(self.history_path):
            print(f"❌ 未找到历史数据文件: {self.history_path}")
            return

        df = pd.read_csv(self.history_path, dtype=str).dropna()  # ⚡ 强制按字符串读入
        feature_df = self.extract_features(df)
        feature_df.to_csv(save_path, index=False)
        print(f"✅ 特征提取完成，共 {len(feature_df)} 条，已保存至: {save_path}")


if __name__ == "__main__":
    generator = FeatureGeneratorV2(history_path=DEFAULT_HISTORY_PATH)
    generator.generate_and_save(save_path=DEFAULT_SAVE_PATH)
