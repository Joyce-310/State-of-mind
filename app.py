
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import json
import re
import hmac
import uuid
import ast

from datetime import datetime, date
from supabase import create_client


# =========================================================
# 1. 系統設定
# =========================================================

st.set_page_config(
    page_title="AI 情緒智慧分析系統",
    page_icon="💗",
    layout="wide",
)

EMOTIONS = [
    "開心", "悲傷", "憤怒",
    "恐懼", "平靜", "焦慮"
]

EMOTION_INFO = {
    "開心": {
        "emoji": "😊",
        "color": "#FFD166",
        "animation": "happy",
        "advice": "很棒！可以把這份好心情分享給身邊的人，也記得好好享受當下。"
    },
    "悲傷": {
        "emoji": "🥺",
        "color": "#8EACD8",
        "animation": "sad",
        "advice": "先給自己一點時間，不需要急著讓自己好起來。可以找信任的人聊聊。"
    },
    "憤怒": {
        "emoji": "😠",
        "color": "#F28B82",
        "animation": "angry",
        "advice": "先深呼吸，讓自己冷靜一下。等情緒穩定後，再思考怎麼表達自己的感受。"
    },
    "恐懼": {
        "emoji": "😨",
        "color": "#B39DDB",
        "animation": "fear",
        "advice": "害怕是正常的，可以先把問題拆成小步驟，一次處理一件事。"
    },
    "平靜": {
        "emoji": "😌",
        "color": "#81C995",
        "animation": "calm",
        "advice": "保持現在穩定的步調，給自己一些休息與整理思緒的時間。"
    },
    "焦慮": {
        "emoji": "😟",
        "color": "#F6B26B",
        "animation": "anxious",
        "advice": "試著把擔心的事情寫下來，先處理目前能控制的部分，不用一次解決所有問題。"
    }
}


# =========================================================
# 2. Session State
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "custom_avatars" not in st.session_state:
    st.session_state.custom_avatars = {}

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False


# =========================================================
# 3. Secrets 設定
# =========================================================

def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, default)

        if value is None:
            return default

        return str(value).strip()

    except Exception:
        return default


INFERENCE_API_URL = get_secret(
    "INFERENCE_API_URL"
).rstrip("/")

INFERENCE_API_TOKEN = get_secret(
    "INFERENCE_API_TOKEN"
)

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")


# =========================================================
# 4. Supabase 資料庫
# =========================================================

@st.cache_resource(show_spinner=False)
def get_supabase():

    try:
        if not SUPABASE_URL or not SUPABASE_KEY:
            return None

        if not SUPABASE_URL.startswith("https://"):
            return None

        return create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

    except Exception as exc:
        print("Supabase 連線錯誤：", exc)
        return None


def save_record(record):

    client = get_supabase()

    if client is None:
        return False, "尚未設定 Supabase Secrets。"

    row = {
        "session_id": st.session_state.session_id,
        "input_text": record["input_text"],
        "emotion": record["emotion"],
        "text_type": record["text_type"],
        "reason": record["reason"],
        "intensity": record["intensity"],
        "advice": record["advice"],
        "created_at": datetime.now().astimezone().isoformat()
    }

    try:
        client.table("emotion_records").insert(row).execute()
        return True, "已儲存"

    except Exception as exc:
        return False, str(exc)


def fetch_my_records():

    client = get_supabase()

    if client is None:
        return []

    try:
        response = (
            client.table("emotion_records")
            .select("*")
            .eq("session_id", st.session_state.session_id)
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        )

        return response.data or []

    except Exception as exc:
        st.error(f"讀取個人紀錄失敗：{exc}")
        return []


def fetch_all_records():

    client = get_supabase()

    if client is None:
        raise RuntimeError("Supabase 尚未設定。")

    rows = []
    start = 0
    page_size = 500

    while True:

        response = (
            client.table("emotion_records")
            .select("*")
            .order("created_at", desc=True)
            .range(start, start + page_size - 1)
            .execute()
        )

        data = response.data or []
        rows.extend(data)

        if len(data) < page_size:
            break

        start += page_size

    return rows


# =========================================================
# 5. Mac Qwen + LoRA API
# =========================================================

def generate_text(prompt, max_new_tokens=400):

    if not INFERENCE_API_URL:
        raise RuntimeError(
            "INFERENCE_API_URL 尚未設定。"
        )

    if not INFERENCE_API_TOKEN:
        raise RuntimeError(
            "INFERENCE_API_TOKEN 尚未設定。"
        )

    try:

        response = requests.post(
            f"{INFERENCE_API_URL}/analyze",
            json={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens
            },
            headers={
                "Authorization": f"Bearer {INFERENCE_API_TOKEN}"
            },
            timeout=(15, 240)
        )

        response.raise_for_status()

        try:
            result = response.json()

        except ValueError:
            raise RuntimeError(
                "API 沒有回傳有效 JSON。\n\n"
                + response.text[:2000]
            )

        generated_text = None

        if isinstance(result, str):
            generated_text = result

        elif isinstance(result, dict):

            possible_keys = [
                "text",
                "generated_text",
                "response",
                "output",
                "answer",
                "content"
            ]

            for key in possible_keys:

                value = result.get(key)

                if isinstance(value, str) and value.strip():
                    generated_text = value
                    break

            if generated_text is None:

                nested = result.get("result")

                if isinstance(nested, dict):

                    for key in possible_keys:

                        value = nested.get(key)

                        if isinstance(value, str) and value.strip():
                            generated_text = value
                            break

                elif isinstance(nested, str) and nested.strip():
                    generated_text = nested

        elif isinstance(result, list) and result:

            first = result[0]

            if isinstance(first, dict):

                generated_text = (
                    first.get("generated_text")
                    or first.get("text")
                )

        if not isinstance(generated_text, str):

            raise RuntimeError(
                "API 找不到模型輸出文字。\n\n"
                "API 原始回覆：\n"
                + repr(result)[:2500]
            )

        generated_text = generated_text.strip()

        if not generated_text:
            raise RuntimeError("Qwen 回傳空白內容。")

        return generated_text

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "模型回應逾時，請確認 Mac 推論服務是否正常。"
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "無法連接 Mac 推論服務，請確認 API 和 Tunnel 是否啟動。"
        )

    except requests.exceptions.HTTPError as exc:

        detail = ""

        if exc.response is not None:
            detail = exc.response.text[:2000]

        raise RuntimeError(
            f"模型 API HTTP 錯誤：{exc}\n\n"
            f"API 回應：\n{detail}"
        )


# =========================================================
# 6. JSON 解析與修復
# =========================================================

def extract_json(text):

    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = text.strip()

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace("```", "").strip()

    cleaned = (
        cleaned
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    def parse_candidate(candidate):

        candidate = candidate.strip()

        if not candidate:
            return None

        try:
            result = json.loads(candidate)

            if isinstance(result, dict):
                return result

        except (json.JSONDecodeError, TypeError):
            pass

        repaired = re.sub(
            r",\s*([}\]])",
            r"\1",
            candidate
        )

        try:
            result = json.loads(repaired)

            if isinstance(result, dict):
                return result

        except (json.JSONDecodeError, TypeError):
            pass

        try:
            result = ast.literal_eval(candidate)

            if isinstance(result, dict):
                return result

        except (ValueError, SyntaxError, TypeError):
            pass

        return None

    result = parse_candidate(cleaned)

    if result is not None:
        return result

    decoder = json.JSONDecoder()

    for start, char in enumerate(cleaned):

        if char != "{":
            continue

        candidate = cleaned[start:]

        try:
            result, _ = decoder.raw_decode(candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

        repaired = re.sub(
            r",\s*([}\]])",
            r"\1",
            candidate
        )

        try:
            result, _ = decoder.raw_decode(repaired)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    return None


# =========================================================
# 7. AI 情緒分析
# =========================================================

def analyze_emotion(user_text):

    prompt = f"""
你是一個專業的繁體中文 AI 情緒分析助手。

請完整理解使用者輸入的內容，
根據上下文、語意、語氣及事件進行情緒判斷。

不要只根據單一關鍵字分類。

情緒只能選擇以下六種：
開心、悲傷、憤怒、恐懼、平靜、焦慮。

請分析以下五個項目：

1. emotion：主要情緒
2. text_type：文字類型
3. reason：情緒原因分析
4. intensity：情緒強度，必須是 1 到 5 的整數
5. advice：溫暖且具體的關懷建議

嚴格遵守：
- 只輸出 JSON。
- 不要輸出 Markdown。
- 不要輸出其他解釋。
- 欄位名稱使用雙引號。
- 五個欄位都不可缺少。
- emotion 必須是指定的六種情緒之一。
- intensity 必須是 1 到 5 的整數。
- 請使用繁體中文。

輸出格式：

{{
    "emotion": "開心",
    "text_type": "日常生活",
    "reason": "情緒原因分析",
    "intensity": 3,
    "advice": "溫暖且具體的建議"
}}

使用者文字：

{user_text}

請直接輸出 JSON：
"""

    response = generate_text(
        prompt,
        max_new_tokens=400
    )

    data = extract_json(response)

    # 第一次失敗時，讓原模型重新整理格式
    if not isinstance(data, dict):

        repair_prompt = f"""
請將以下模型回答整理成合法 JSON。

只能輸出以下五個欄位：
emotion、text_type、reason、intensity、advice。

emotion 只能是：
開心、悲傷、憤怒、恐懼、平靜、焦慮。

intensity 必須是 1 到 5 的整數。

不要增加其他欄位，不要輸出解釋。

原始回答：
{response}

請輸出 JSON：
"""

        repair_response = generate_text(
            repair_prompt,
            max_new_tokens=300
        )

        data = extract_json(repair_response)

        if not isinstance(data, dict):

            raise ValueError(
                "AI JSON 格式錯誤，無法解析模型回覆。\n\n"
                "第一次原始回覆：\n"
                + repr(response)[:1800]
                + "\n\n第二次修復回覆：\n"
                + repr(repair_response)[:1200]
            )

    # 情緒分類
    emotion = str(
        data.get("emotion", "")
    ).strip()

    emotion = re.sub(
        r"^(主要情緒|情緒分類|情緒)\s*[:：]\s*",
        "",
        emotion
    ).strip()

    if emotion not in EMOTIONS:

        raise ValueError(
            f"情緒分類錯誤：{emotion}\n\n"
            "模型原始回覆：\n"
            + repr(response)[:1500]
        )

    # 情緒強度
    try:

        raw_intensity = data.get("intensity", 3)

        if isinstance(raw_intensity, str):

            match = re.search(
                r"[1-5]",
                raw_intensity
            )

            intensity = int(match.group()) if match else 3

        else:
            intensity = int(float(raw_intensity))

    except (ValueError, TypeError, OverflowError):
        intensity = 3

    intensity = max(1, min(5, intensity))

    text_type = str(
        data.get("text_type") or "日常對話"
    ).strip()

    reason = str(
        data.get("reason") or "AI 正在理解你的文字內容。"
    ).strip()

    advice = str(
        data.get("advice")
        or EMOTION_INFO[emotion]["advice"]
    ).strip()

    return {
        "emotion": emotion,
        "text_type": text_type,
        "reason": reason,
        "intensity": intensity,
        "advice": advice
    }


# =========================================================
# 8. Q 版動態角色
# =========================================================

def avatar_svg(emotion):

    info = EMOTION_INFO[emotion]

    emoji = info["emoji"]
    color = info["color"]
    animation = info["animation"]

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
    body {{
        margin: 0;
        padding: 10px;
        background: transparent;
        font-family: Arial;
    }}

    .avatar-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 220px;
    }}

    .avatar {{
        width: 150px;
        height: 150px;
        border-radius: 45%;
        background: {color};
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 85px;
        box-shadow: 0 8px 20px rgba(0,0,0,.12);
        animation: {animation} 1.5s ease-in-out infinite;
    }}

    @keyframes happy {{
        0%,100% {{transform:translateY(0)}}
        50% {{transform:translateY(-15px)}}
    }}

    @keyframes sad {{
        0%,100% {{transform:translateY(0)}}
        50% {{transform:translateY(8px)}}
    }}

    @keyframes angry {{
        0%,100% {{transform:translateX(0)}}
        50% {{transform:translateX(7px)}}
    }}

    @keyframes fear {{
        0%,100% {{transform:scale(1)}}
        50% {{transform:scale(.92)}}
    }}

    @keyframes calm {{
        0%,100% {{transform:translateY(0)}}
        50% {{transform:translateY(-8px)}}
    }}

    @keyframes anxious {{
        0%,100% {{transform:rotate(0)}}
        50% {{transform:rotate(5deg)}}
    }}
    </style>
    </head>

    <body>
    <div class="avatar-container">
        <div class="avatar">{emoji}</div>
    </div>

    <h3 style="text-align:center;color:{color}">
        {emotion}
    </h3>
    </body>
    </html>
    """


def show_avatar(emotion):

    custom_image = st.session_state.custom_avatars.get(emotion)

    if custom_image:

        st.image(custom_image, width=180)

        st.markdown(
            f"<h4 style='text-align:center'>{emotion}</h4>",
            unsafe_allow_html=True
        )

    else:

        components.html(
            avatar_svg(emotion),
            height=250,
            scrolling=False
        )


# =========================================================
# 9. 側邊欄
# =========================================================

with st.sidebar:

    st.header("⚙️ 系統設定")

    st.markdown("### 🤖 AI 模型")

    st.write("Qwen2.5-0.5B-Instruct")
    st.write("原本 LoRA 情緒辨識模型")
    st.write("Mac 本機推論 API")

    st.divider()

    st.markdown("### 🔌 服務連線狀態")

    st.write(
        "模型 API：",
        "✅ 已設定" if INFERENCE_API_URL else "❌ 未設定"
    )

    st.write(
        "API Token：",
        "✅ 已設定" if INFERENCE_API_TOKEN else "❌ 未設定"
    )

    st.write(
        "Supabase：",
        "✅ 已設定"
        if SUPABASE_URL and SUPABASE_KEY
        else "❌ 未設定"
    )

    st.divider()

    st.markdown("### 🎨 自訂情緒角色")

    for emotion in EMOTIONS:

        with st.expander(
            f"{EMOTION_INFO[emotion]['emoji']} {emotion}"
        ):

            uploaded = st.file_uploader(
                f"上傳{emotion}圖片",
                type=["png", "jpg", "jpeg"],
                key=f"avatar_{emotion}"
            )

            if uploaded:

                st.session_state.custom_avatars[emotion] = (
                    uploaded.getvalue()
                )

            if emotion in st.session_state.custom_avatars:

                st.image(
                    st.session_state.custom_avatars[emotion],
                    width=120
                )

                if st.button(
                    "恢復預設角色",
                    key=f"reset_{emotion}"
                ):

                    del st.session_state.custom_avatars[emotion]
                    st.rerun()

    st.divider()

    st.markdown("### 📊 本次連線紀錄")

    st.write(
        f"目前分析：{len(st.session_state.history)} 筆"
    )

    if st.button(
        "清除本次畫面紀錄",
        use_container_width=True
    ):

        st.session_state.history = []
        st.session_state.analysis_result = None
        st.rerun()


# =========================================================
# 10. 主頁面
# =========================================================

st.title("💗 AI 情緒智慧分析系統")

st.markdown(
    "透過 Qwen2.5-0.5B 與原本 LoRA，"
    "理解文字中的情緒，並提供情緒原因分析與主動關懷。"
)

st.info(
    "隱私告知：為提供情緒紀錄與管理功能，"
    "您送出的文字及分析結果會儲存在資料庫。"
    "系統管理者可查看這些紀錄。"
    "請勿輸入密碼、身分證字號或其他不必要的敏感個資。"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "💗 情緒主動分析",
    "📊 情緒趨勢分析",
    "📖 系統介紹",
    "🔐 管理者後台"
])


# =========================================================
# 11. 情緒主動分析
# =========================================================

with tab1:

    st.subheader("💬 輸入你的文字")

    user_text = st.text_area(
        "分享今天的心情、生活事件或想法：",
        height=150,
        placeholder="例如：今天報告終於完成了，真的覺得很開心！"
    )

    consent = st.checkbox(
        "我已閱讀隱私告知，並同意儲存文字及分析結果。"
    )

    analyze_button = st.button(
        "✨ 開始主動分析",
        type="primary",
        use_container_width=True,
        disabled=not consent
    )

    if analyze_button:

        if not user_text.strip():

            st.warning("請先輸入文字。")

        else:

            try:

                with st.spinner(
                    "AI 正在理解你的文字與情緒..."
                ):

                    result = analyze_emotion(
                        user_text.strip()
                    )

                    result["input_text"] = user_text.strip()

                    result["time"] = (
                        datetime.now()
                        .astimezone()
                        .strftime("%Y-%m-%d %H:%M:%S")
                    )

                    st.session_state.analysis_result = result

                    st.session_state.history.append(result)

                    saved, message = save_record(result)

                    if saved:

                        st.success("分析完成，紀錄已儲存。")

                    else:

                        st.warning(
                            f"分析完成，但資料庫儲存失敗：{message}"
                        )

            except Exception as exc:

                st.error(f"AI 分析失敗：{exc}")

                with st.expander(
                    "查看完整除錯資訊",
                    expanded=True
                ):

                    st.code(str(exc), language="text")

    if st.session_state.analysis_result:

        result = st.session_state.analysis_result
        emotion = result["emotion"]
        info = EMOTION_INFO[emotion]

        st.divider()
        st.subheader("🌈 情緒分析結果")

        col1, col2 = st.columns([1, 1.5])

        with col1:
            show_avatar(emotion)

        with col2:

            st.markdown(
                f"### {info['emoji']} 主要情緒：{emotion}"
            )

            st.metric(
                "情緒強度",
                f"{result['intensity']} / 5"
            )

            st.markdown("**📝 文字類型**")
            st.write(result["text_type"])

            st.markdown("**🔍 情緒原因分析**")
            st.write(result["reason"])

        st.divider()
        st.subheader("💌 AI 主動關懷")

        st.info(result["advice"])

        st.caption(
            "以上是 AI 根據文字推測的情緒，不代表專業心理診斷。"
        )

        with st.expander("查看本次分析的原始文字"):
            st.write(result["input_text"])


# =========================================================
# 12. 個人情緒趨勢
# =========================================================

with tab2:

    st.subheader("📊 我的情緒歷史與趨勢")

    db_rows = fetch_my_records()

    if db_rows:

        df = pd.DataFrame(db_rows)

        df["time"] = pd.to_datetime(
            df["created_at"],
            errors="coerce"
        )

    elif st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        df["time"] = pd.to_datetime(
            df["time"],
            errors="coerce"
        )

    else:

        df = pd.DataFrame()

    if df.empty:

        st.info("目前還沒有情緒紀錄。")

    else:

        required_columns = [
            "time",
            "input_text",
            "emotion",
            "text_type",
            "intensity"
        ]

        for column in required_columns:

            if column not in df.columns:
                df[column] = None

        display_df = df[required_columns].copy()

        display_df.columns = [
            "分析時間",
            "輸入文字",
            "情緒",
            "文字類型",
            "情緒強度"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.markdown("### 📈 情緒分布")

        emotion_counts = (
            df["emotion"]
            .value_counts()
            .reindex(EMOTIONS, fill_value=0)
        )

        st.bar_chart(emotion_counts)

        st.divider()
        st.markdown("### 📉 情緒強度變化")

        chart_df = (
            df.dropna(subset=["time"])
            [["time", "intensity"]]
            .copy()
        )

        chart_df["intensity"] = pd.to_numeric(
            chart_df["intensity"],
            errors="coerce"
        )

        chart_df = (
            chart_df.dropna()
            .set_index("time")
            .sort_index()
        )

        if not chart_df.empty:
            st.line_chart(chart_df)

        st.download_button(
            "⬇️ 下載我的情緒紀錄 CSV",
            data=df.to_csv(
                index=False,
                encoding="utf-8-sig"
            ),
            file_name="my_emotion_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# 13. 系統介紹
# =========================================================

with tab3:

    st.subheader("📖 系統功能介紹")

    st.markdown("""
### 💗 AI 情緒智慧分析系統

本系統結合小型語言模型與 LoRA 微調技術，
讓 AI 理解使用者輸入的文字，
分析可能的情緒狀態，並提供適當的關懷與建議。

### 🤖 使用技術

- **Qwen2.5-0.5B-Instruct**：基礎語言模型。
- **LoRA**：低秩適應微調技術。
- **PyTorch**：模型推論。
- **Streamlit**：互動式網頁介面。
- **Apple Silicon MPS / CPU**：Mac 本機模型推論。
- **Supabase**：情緒分析紀錄。

### 🌈 六種情緒分類

| 情緒 | 說明 |
|---|---|
| 😊 開心 | 正向、愉快的情緒 |
| 🥺 悲傷 | 失落、難過的情緒 |
| 😠 憤怒 | 不滿、生氣的情緒 |
| 😨 恐懼 | 害怕、不安全感 |
| 😌 平靜 | 穩定、平和的情緒 |
| 😟 焦慮 | 緊張、擔心的情緒 |

### ✨ 系統特色

1. AI 語意情緒辨識。
2. 自動判斷文字類型。
3. 情緒原因分析。
4. 主動關懷與建議。
5. Q 版動態情緒角色。
6. 自訂角色圖片。
7. 情緒歷史與趨勢分析。
8. CSV 紀錄下載。
9. 管理者後台。

本系統僅提供文字情緒推測，
不代表使用者真實心理狀態，也不是心理診斷工具。
""")


# =========================================================
# 14. 管理者後台
# =========================================================

with tab4:

    st.subheader("🔐 管理者登入")

    if not ADMIN_PASSWORD:

        st.error("尚未設定 ADMIN_PASSWORD。")

    elif not st.session_state.admin_authenticated:

        with st.form("admin_login_form"):

            entered_password = st.text_input(
                "管理者密碼",
                type="password"
            )

            login_clicked = st.form_submit_button(
                "登入管理者後台",
                use_container_width=True
            )

        if login_clicked:

            if hmac.compare_digest(
                entered_password,
                ADMIN_PASSWORD
            ):

                st.session_state.admin_authenticated = True
                st.rerun()

            else:

                st.error("密碼錯誤。")

    else:

        top_left, top_right = st.columns([4, 1])

        with top_left:
            st.success("管理者已登入")

        with top_right:

            if st.button("登出", use_container_width=True):

                st.session_state.admin_authenticated = False
                st.rerun()

        try:

            all_rows = fetch_all_records()

            if not all_rows:

                st.info("目前沒有使用者紀錄。")

            else:

                admin_df = pd.DataFrame(all_rows)

                admin_df["created_at"] = pd.to_datetime(
                    admin_df["created_at"],
                    errors="coerce"
                )

                st.metric(
                    "全部分析紀錄",
                    f"{len(admin_df):,} 筆"
                )

                filter1, filter2 = st.columns(2)

                with filter1:

                    selected_emotions = st.multiselect(
                        "依情緒篩選",
                        EMOTIONS,
                        default=EMOTIONS
                    )

                with filter2:

                    today = date.today()

                    date_range = st.date_input(
                        "依日期篩選",
                        value=(today.replace(day=1), today)
                    )

                filtered = admin_df[
                    admin_df["emotion"].isin(selected_emotions)
                ].copy()

                if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

                    start_date, end_date = date_range

                    filtered = filtered[
                        (filtered["created_at"].dt.date >= start_date)
                        &
                        (filtered["created_at"].dt.date <= end_date)
                    ]

                elif isinstance(date_range, date):

                    filtered = filtered[
                        filtered["created_at"].dt.date == date_range
                    ]

                st.write(
                    f"符合條件：{len(filtered):,} 筆"
                )

                cols = [
                    "created_at",
                    "session_id",
                    "input_text",
                    "emotion",
                    "text_type",
                    "reason",
                    "intensity",
                    "advice"
                ]

                for column in cols:

                    if column not in filtered.columns:
                        filtered[column] = None

                view_df = filtered[cols].copy()

                view_df.columns = [
                    "分析時間",
                    "匿名連線代碼",
                    "使用者輸入",
                    "情緒",
                    "文字類型",
                    "情緒原因",
                    "強度",
                    "AI 建議"
                ]

                st.dataframe(
                    view_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ 下載全部紀錄 CSV",
                    data=filtered.to_csv(
                        index=False,
                        encoding="utf-8-sig"
                    ).encode("utf-8-sig"),
                    file_name=f"emotion_records_{date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as exc:

            st.error(
                f"管理者資料讀取失敗：{exc}"
            )


# =========================================================
# 15. 頁尾
# =========================================================

st.divider()

st.markdown(
    """
    <div style='text-align:center;color:gray;'>
        AI 情緒智慧分析系統 |
        Qwen2.5 + LoRA Mac 本機推論
    </div>
    """,
    unsafe_allow_html=True
)
