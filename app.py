
import streamlit as st
import streamlit.components.v1 as components
import torch
import pandas as pd
import json
import re
import os
import base64
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================================================
# 1. 系統設定
# =========================================================

st.set_page_config(
    page_title="AI 情緒智慧分析系統",
    page_icon="💗",
    layout="wide"
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "outputs/qwen-emotion-lora"

EMOTIONS = ["開心", "悲傷", "憤怒", "恐懼", "平靜", "焦慮"]

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

# =========================================================
# 3. 載入 Qwen + LoRA
# =========================================================

@st.cache_resource(show_spinner="正在載入 AI 情緒分析模型，請稍候...")
def load_model():

    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(
            f"找不到 LoRA 模型資料夾：{LORA_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# =========================================================
# 4. AI 文字生成
# =========================================================

def generate_text(prompt, max_new_tokens=250):

    tokenizer, model, device = load_model()

    messages = [
        {
            "role": "system",
            "content": "你是一位專業、溫暖且善於理解人類情緒的 AI 助理。請使用繁體中文回答。"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    result = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return result.strip()


# =========================================================
# 5. JSON 解析
# =========================================================

def extract_json(text):

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None

    return None


# =========================================================
# 6. 情緒分析
# =========================================================

def analyze_emotion(user_text):

    prompt = f"""
請分析以下使用者輸入的文字。

使用完整語意判斷，不要只依照單一關鍵字分類。

情緒只能選擇以下六種：
開心、悲傷、憤怒、恐懼、平靜、焦慮。

請自動判斷文字類型，例如：
日常對話、心情分享、人際關係、學業壓力、工作壓力、
生活事件、疑問句、其他。

請分析：
1. 主要情緒
2. 情緒原因
3. 文字類型
4. 情緒程度（1到5）
5. 給使用者的關懷建議

請只輸出 JSON，不要輸出其他說明。

格式：
{{
    "emotion": "六種情緒其中之一",
    "text_type": "文字類型",
    "reason": "情緒原因分析",
    "intensity": 3,
    "advice": "溫暖且具體的建議"
}}

使用者文字：
{user_text}
"""

    try:
        response = generate_text(prompt, max_new_tokens=300)
        data = extract_json(response)

        if data:
            emotion = data.get("emotion", "平靜")

            if emotion not in EMOTIONS:
                emotion = "平靜"

            try:
                intensity = int(data.get("intensity", 3))
            except (ValueError, TypeError):
                intensity = 3

            return {
                "emotion": emotion,
                "text_type": str(data.get("text_type", "日常對話")),
                "reason": str(data.get("reason", "AI 正在理解你的文字內容。")),
                "intensity": max(1, min(5, intensity)),
                "advice": str(data.get("advice", EMOTION_INFO[emotion]["advice"]))
            }

    except Exception as e:
        st.warning(f"AI 模型分析暫時無法使用，改以基本分析方式呈現。")
        st.caption(str(e))

    return fallback_analysis(user_text)


# =========================================================
# 7. 基本備援分析
# =========================================================

def fallback_analysis(text):

    keywords = {
        "開心": ["開心", "快樂", "幸福", "成功", "喜歡", "期待", "好棒"],
        "悲傷": ["難過", "傷心", "失望", "孤單", "哭", "失去", "委屈"],
        "憤怒": ["生氣", "討厭", "憤怒", "煩死", "不爽", "過分"],
        "恐懼": ["害怕", "恐怖", "擔心", "不敢", "恐懼", "危險"],
        "焦慮": ["焦慮", "壓力", "緊張", "來不及", "煩惱", "不安"],
        "平靜": ["還好", "普通", "平常", "沒事", "平靜"]
    }

    scores = {
        emotion: sum(word in text for word in words)
        for emotion, words in keywords.items()
    }

    emotion = max(scores, key=scores.get)

    if scores[emotion] == 0:
        emotion = "平靜"

    if any(word in text for word in ["考試", "報告", "作業", "上課", "畢業"]):
        text_type = "學業相關"
    elif any(word in text for word in ["朋友", "家人", "男友", "女友", "同學"]):
        text_type = "人際關係"
    elif any(word in text for word in ["工作", "主管", "同事", "上班"]):
        text_type = "工作相關"
    elif "？" in text or "?" in text:
        text_type = "疑問句"
    else:
        text_type = "日常對話"

    return {
        "emotion": emotion,
        "text_type": text_type,
        "reason": "根據文字內容進行初步判斷，可能反映了目前的情緒狀態。",
        "intensity": 3,
        "advice": EMOTION_INFO[emotion]["advice"]
    }


# =========================================================
# 8. Q 版動態角色 HTML
# =========================================================

def avatar_svg(emotion):

    info = EMOTION_INFO[emotion]

    emoji = info["emoji"]
    color = info["color"]
    animation = info["animation"]

    # CSS 與角色放在同一個 HTML 中
    # 使用 components.html 時不會受到外部 CSS 隔離影響

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">

    <style>

    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        padding: 10px;
        background: transparent;
        font-family: Arial, sans-serif;
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
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        animation: {animation} 1.5s ease-in-out infinite;
        position: relative;
    }}

    .avatar::after {{
        content: "";
        position: absolute;
        bottom: -12px;
        width: 75px;
        height: 15px;
        background: rgba(0,0,0,0.12);
        border-radius: 50%;
        animation: shadow 1.5s ease-in-out infinite;
    }}

    @keyframes happy {{
        0%, 100% {{ transform: translateY(0) rotate(0deg); }}
        25% {{ transform: translateY(-15px) rotate(-8deg); }}
        50% {{ transform: translateY(-5px) rotate(8deg); }}
        75% {{ transform: translateY(-15px) rotate(-5deg); }}
    }}

    @keyframes sad {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(8px); }}
    }}

    @keyframes angry {{
        0%, 100% {{ transform: translateX(0); }}
        25% {{ transform: translateX(-7px); }}
        50% {{ transform: translateX(7px); }}
        75% {{ transform: translateX(-5px); }}
    }}

    @keyframes fear {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(0.92); }}
    }}

    @keyframes calm {{
        0%, 100% {{ transform: translateY(0) scale(1); }}
        50% {{ transform: translateY(-8px) scale(1.03); }}
    }}

    @keyframes anxious {{
        0%, 100% {{ transform: rotate(0deg); }}
        25% {{ transform: rotate(-5deg); }}
        75% {{ transform: rotate(5deg); }}
    }}

    @keyframes shadow {{
        0%, 100% {{ transform: scale(1); opacity: 0.5; }}
        50% {{ transform: scale(0.7); opacity: 0.2; }}
    }}

    .emotion-name {{
        text-align: center;
        color: {color};
        font-size: 22px;
        font-weight: bold;
        margin-top: 0;
    }}

    </style>
    </head>

    <body>

    <div class="avatar-container">
        <div class="avatar">{emoji}</div>
    </div>

    <div class="emotion-name">{emotion}</div>

    </body>
    </html>
    """

    return html


# =========================================================
# 9. 顯示角色
# =========================================================

def show_avatar(emotion):

    # 如果使用者有上傳自訂角色圖片，優先顯示圖片
    custom_image = st.session_state.custom_avatars.get(emotion)

    if custom_image:
        st.image(
            custom_image,
            width=180
        )
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
# 10. 頁面標題
# =========================================================

st.title("💗 AI 情緒智慧分析系統")

st.markdown(
    "透過 Qwen2.5-0.5B 與 LoRA，理解文字中的情緒，"
    "並提供情緒原因分析與主動關懷。"
)

st.divider()


# =========================================================
# 11. 側邊欄
# =========================================================

with st.sidebar:

    st.header("⚙️ 系統設定")

    st.markdown("### 🤖 AI 模型")
    st.write("Qwen2.5-0.5B-Instruct")
    st.write("LoRA 情緒辨識模型")

    st.divider()

    st.markdown("### 🎨 自訂情緒角色")

    st.caption("可以為不同情緒上傳自己的角色圖片。")

    for emotion in EMOTIONS:

        with st.expander(f"{EMOTION_INFO[emotion]['emoji']} {emotion}"):

            uploaded = st.file_uploader(
                f"上傳{emotion}圖片",
                type=["png", "jpg", "jpeg"],
                key=f"avatar_{emotion}"
            )

            if uploaded:
                st.session_state.custom_avatars[emotion] = uploaded.getvalue()

            if emotion in st.session_state.custom_avatars:
                st.image(
                    st.session_state.custom_avatars[emotion],
                    width=120
                )

                if st.button(
                    f"恢復預設角色",
                    key=f"reset_{emotion}"
                ):
                    del st.session_state.custom_avatars[emotion]
                    st.rerun()

    st.divider()

    st.markdown("### 📊 歷史紀錄")

    st.write(f"目前紀錄：{len(st.session_state.history)} 筆")

    if st.button("清除所有歷史紀錄", use_container_width=True):
        st.session_state.history = []
        st.session_state.analysis_result = None
        st.rerun()


# =========================================================
# 12. 分頁
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "💗 情緒主動分析",
    "📊 情緒趨勢分析",
    "📖 系統介紹"
])


# =========================================================
# 13. 情緒分析頁面
# =========================================================

with tab1:

    st.subheader("💬 輸入你的文字")

    user_text = st.text_area(
        "分享今天的心情、生活事件或想法：",
        height=150,
        placeholder="例如：今天報告終於完成了，真的覺得很開心！"
    )

    analyze_button = st.button(
        "✨ 開始主動分析",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not user_text.strip():
            st.warning("請先輸入文字，再進行情緒分析。")

        else:

            with st.spinner("AI 正在理解你的文字與情緒..."):

                result = analyze_emotion(user_text)

                result["input_text"] = user_text
                result["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state.analysis_result = result
                st.session_state.history.append(result)

    # 顯示分析結果
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

            st.markdown(f"### {info['emoji']} 主要情緒：{emotion}")

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
# 14. 情緒趨勢分析
# =========================================================

with tab2:

    st.subheader("📊 情緒歷史與趨勢")

    if len(st.session_state.history) == 0:

        st.info("目前還沒有情緒紀錄，請先到「情緒主動分析」進行分析。")

    else:

        df = pd.DataFrame(st.session_state.history)

        st.markdown("### 📋 歷史紀錄")

        display_df = df[
            ["time", "input_text", "emotion", "text_type", "intensity"]
        ].copy()

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

        df["time"] = pd.to_datetime(df["time"])

        chart_df = df[
            ["time", "intensity"]
        ].set_index("time")

        st.line_chart(chart_df)

        st.divider()

        csv_data = df.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            "⬇️ 下載情緒分析紀錄 CSV",
            data=csv_data,
            file_name="emotion_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# 15. 系統介紹
# =========================================================

with tab3:

    st.subheader("📖 系統功能介紹")

    st.markdown("""
    ### 💗 AI 情緒智慧分析系統

    本系統結合小型語言模型與 LoRA 微調技術，
    讓 AI 能夠理解使用者輸入的文字，分析可能的情緒狀態，
    並提供適當的關懷與建議。

    ### 🤖 使用技術

    - **Qwen2.5-0.5B-Instruct**：作為基礎語言模型。
    - **LoRA**：透過低秩適應技術進行情緒辨識。
    - **Python**：主要程式開發語言。
    - **Streamlit**：建立互動式網頁介面。
    - **PyTorch**：執行深度學習模型。

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

    1. 自動辨識使用者輸入的文字類型。
    2. 分析文字中的主要情緒。
    3. 根據語意推測可能的情緒原因。
    4. 提供主動關懷與建議。
    5. 透過 Q 版動態角色呈現情緒。
    6. 支援自訂情緒角色圖片。
    7. 保存情緒歷史並分析變化趨勢。
    8. 支援 CSV 紀錄下載。
    """)

st.divider()

st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "AI 情緒智慧分析系統 | Qwen2.5 + LoRA"
    "</div>",
    unsafe_allow_html=True
)
