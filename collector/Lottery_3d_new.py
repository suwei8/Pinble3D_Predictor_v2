# collector/Lottery_3d.py

import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESENTINFO_PATH = os.path.join(BASE_DIR, "data", "pinble3d_presentinfo.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "3d_shijihao_history.csv")

load_dotenv()  # 本地调试时加载 .env，Actions 上忽略

HOST = os.getenv("PINBLE_PRESENTINFO_HOST")
if not HOST:
    raise ValueError("❌ 缺少环境变量 PINBLE_PRESENTINFO_HOST")

URL = f"http://{HOST}"


def fetch_presentinfo():
    resp = requests.get(URL, timeout=120)
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "xml")
    html_text = soup.find("string").text
    html = BeautifulSoup(html_text, "html.parser")

    tables = html.find_all("table")
    for i in range(len(tables)):
        if "福彩3D" in tables[i].text:
            date_tag = tables[i].find("span", class_="kaiTime")
            date = date_tag.get_text(strip=True) if date_tag else None

            next_table = tables[i + 1] if i + 1 < len(tables) else None
            if next_table:
                text = next_table.get_text()

                def extract_digits(pattern):
                    m = re.search(pattern, text)
                    return ''.join(re.findall(r"\d", m.group(1))) if m else None

                def extract_focus(text):
                    m = re.search(r"关注码.*?：(.*?)金码", text)
                    return ''.join(re.findall(r"\d", m.group(1))) if m else None

                def extract_gold(text):
                    m = re.search(r"金码.*?：.*?(\d+)", text)
                    return m.group(1) if m else None

                sim_code = extract_digits(r"模拟试机号：\[(.*?)\]")
                open_code = extract_digits(r"开奖号：\[(.*?)\]")
                focus_code = extract_focus(text)
                gold_code = extract_gold(text)

                issue_tag = tables[i].find("a")
                issue = re.search(r"(\d{7})期", issue_tag.text).group(1) if issue_tag else None

                return {
                    "issue": issue,
                    "date": date,
                    "sim_code": sim_code,
                    "focus_code": focus_code,
                    "gold_code": gold_code,
                    "open_code": open_code
                }

    return None


def issue_exists(issue):
    exists = False
    if os.path.exists(PRESENTINFO_PATH):
        df = pd.read_csv(PRESENTINFO_PATH, usecols=["issue"], dtype={"issue": str})
        if issue in df["issue"].values:
            exists = True
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH, usecols=["issue"], dtype={"issue": str})
        if issue in df["issue"].values:
            exists = True
    return exists


def save_presentinfo(info):
    df = pd.DataFrame([{
        "issue": info["issue"],
        "date": info["date"],
        "sim_code": info["sim_code"],
        "focus_code": info["focus_code"],
        "gold_code": info["gold_code"],
        "open_code": info["open_code"]
    }])
    if os.path.exists(PRESENTINFO_PATH):
        df.to_csv(PRESENTINFO_PATH, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df.to_csv(PRESENTINFO_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存至 {PRESENTINFO_PATH}")


def append_history(info):
    df = pd.DataFrame([{
        "date": info["date"],
        "issue": info["issue"],
        "sim_test_code": info["sim_code"],
        "open_code": info["open_code"]
    }])
    if os.path.exists(HISTORY_PATH):
        df.to_csv(HISTORY_PATH, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    print(f"📦 已追加至 {HISTORY_PATH}")


if __name__ == "__main__":
    info = fetch_presentinfo()
    if info:
        print(f"🎯 获取成功：{info}")
        if issue_exists(info["issue"]):
            print(f"⏩ 已存在 issue={info['issue']}，跳过采集")
        else:
            save_presentinfo(info)
            append_history(info)
    else:
        print("❌ 未能提取到福彩3D PresentInformation 内容")
