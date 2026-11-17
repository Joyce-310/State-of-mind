import streamlit as st
import streamlit.components.v1 as components
import torch
import pandas as pd
import json
import re
import os
import hmac
import uuid
from datetime import datetime, date, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from supabase import create_client

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
    "開心": {"emoji": "😊", "color": "#FFD166", "animation": "happy", "advice": "很棒！可以把這份好心情分享給身邊的人，也記得好好享受當下。"},
    "悲傷": {"emoji": "🥺", "color": "#8EACD8", "animation": "sad", "advice": "先給自己一點時間，不需要急著讓自己好起來。可以找信任的人聊聊。"},
    "憤怒": {"emoji": "😠", "color": "#F28B82", "animation": "angry", "advice": "先深呼吸，讓自己冷靜一下。等情緒穩定後，再思考怎麼表達自己的感受。"},
    "恐懼": {"emoji": "😨", "color": "#B39DDB", "animation": "fear", "advice": "害怕是正常的，可以先把問題拆成小步驟，一次處理一件事。"},
    "平靜": {"emoji": "😌", "color": "#81C995", "animation": "calm", "advice": "保持現在穩定的步調，給自己一些休息與整理思緒的時間。"},
    "焦慮": {"emoji": "😟", "color": "#F6B26B", "animation": "anxious", "advice": "試著把擔心的事情寫下來，先處理目前能控制的部分，不用一次解決所有問題。"}
}

# =========================================================
# 2. Session State：每個瀏覽器連線產生獨立匿名 ID
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
# 3. Supabase 連線與資料庫操作
# =================================================n@st.cache_resource(show_spinner=False)
def get_supabase():
    """使用 Streamlit Secrets 在伺服器端連線；金鑰不會傳給瀏覽器。"""
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)


def save_record(record):
    """將一筆分析結果寫入 Supabase。"""
    client = get_supabase()
    if client is None:
        return False, "尚未設定 Supabase Secrets。"

    row = {
        "session_id": st.session_state.session_id,
        "input_text": record["input_text"],
        "emotion": record["emotion"],
        "text_type": record.get("text_type", ""),
        "reason": record.get("reason", ""),
        "intensity": int(record.get("intensity", 3)),
        "advice": record.get("advice", ""),
        "created_at": datetime.now().astimezone().isoformat()
    }
    try:
        client.table("emotion_records").insert(row).execute()
        return True, "已儲存"
    except Exception as exc:
        return False, str(exc)


def fetch_my_records():
    """一般使用者只查詢目前 session_id 的資料。"""
    client = get_supabase()
    if client is None:
        return []
    try:
        response = (
            client.table("emotion_records")
            .select("id,session_id,input_text,emotion,text_type,reason,intensity,advice,created_at")
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
    """管理者登入後由伺服器端分頁讀取所有紀錄。"""
    client = get_supabase()
    if client is None:
        raise RuntimeError("尚未設定 Supabase Secrets。")
    all_rows = []
    page_size = 1000
    start = 0
    while True:
        response = (
            client.table("emotion_records")
            .select("id,session_id,input_text,emotion,text_type,reason,intensity,advice,created_at")
            .order("created_at", desc=True)
            .range(start, start + page_size - 1)
            .execute()
        )
        rows = response.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return all_rows

# =========================================================
# 4. 載入 Qwen + LoRA
# =========================================================
@st.cache_resource(show_spinner="正在載入 AI 情緒分析模型，請稍候...")
def load_model():
    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"找不到 LoRA 模型資料夾：{LORA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model, device

# =========================================================
# 5. AI 文字生成與 JSON 解析
# =========================================================
def generate_text(prompt, max_new_tokens=250):
    tokenizer, model, device = load_model()
    messages = [
        {"role": "system", "content": "你是一位專業、溫暖且善於理解人類情緒的 AI 助理。請使用繁體中文回答。"},
        {"role": "user", "content": prompt}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
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
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
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
請分析以下使用者輸入的文字。使用完整語意判斷，不要只依照單一關鍵字分類。
情緒只能選擇：開心、悲傷、憤怒、恐懼、平靜、焦慮。
請自動判斷文字類型，例如日常對話、心情分享、人際關係、學業壓力、工作壓力、生活事件、疑問句或其他。
請分析主要情緒、情緒原因、文字類型、情緒程度（1到5）、關懷建議。
請只輸出 JSON，不要輸出其他說明。格式：
{{"emotion":"六種情緒其中之一","text_type":"文字類型","reason":"情緒原因分析","intensity":3,"advice":"溫暖且具體的建議"}}
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
    except Exception as exc:
        st.warning("AI 模型分析暫時無法使用，改以基本分析方式呈現。")
        st.caption(str(exc))
    return fallback_analysis(user_text)


def fallback_analysis(text):
    keywords = {
        "開心": ["開心", "快樂", "幸福", "成功", "喜歡", "期待", "好棒"],
        "悲傷": ["難過", "傷心", "失望", "孤單", "哭", "失去", "委屈"],
        "憤怒": ["生氣", "討厭", "憤怒", "煩死", "不爽", "過分"],
        "恐懼": ["害怕", "恐怖", "不敢", "恐懼", "危險"],
        "焦慮": ["焦慮", "壓力", "緊張", "來不及", "煩惱", "不安", "擔心"],
        "平靜": ["還好", "普通", "平常", "沒事", "平靜"]
    }
    scores = {emotion: sum(word in text for word in words) for emotion, words in keywords.items()}
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
# 7. Q 版動態角色
# =========================================================
def avatar_svg(emotion):
    info = EMOTION_INFO[emotion]
    emoji, color, animation = info["emoji"], info["color"], info["animation"]
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
    * {{box-sizing:border-box;}} body {{margin:0;padding:10px;background:transparent;font-family:Arial,sans-serif;}}
    .avatar-container {{display:flex;justify-content:center;align-items:center;height:220px;}}
    .avatar {{width:150px;height:150px;border-radius:45%;background:{color};display:flex;justify-content:center;align-items:center;font-size:85px;box-shadow:0 8px 20px rgba(0,0,0,.12);animation:{animation} 1.5s ease-in-out infinite;position:relative;}}
    .avatar:after {{content:"";position:absolute;bottom:-12px;width:75px;height:15px;background:rgba(0,0,0,.12);border-radius:50%;animation:shadow 1.5s ease-in-out infinite;}}
    @keyframes happy {{0%,100%{{transform:translateY(0) rotate(0)}}25%{{transform:translateY(-15px) rotate(-8deg)}}50%{{transform:translateY(-5px) rotate(8deg)}}75%{{transform:translateY(-15px) rotate(-5deg)}}}}
    @keyframes sad {{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(8px)}}}}
    @keyframes angry {{0%,100%{{transform:translateX(0)}}25%{{transform:translateX(-7px)}}50%{{transform:translateX(7px)}}75%{{transform:translateX(-5px)}}}}
    @keyframes fear {{0%,100%{{transform:scale(1)}}50%{{transform:scale(.92)}}}}
    @keyframes calm {{0%,100%{{transform:translateY(0) scale(1)}}50%{{transform:translateY(-8px) scale(1.03)}}}}
    @keyframes anxious {{0%,100%{{transform:rotate(0)}}25%{{transform:rotate(-5deg)}}75%{{transform:rotate(5deg)}}}}
    @keyframes shadow {{0%,100%{{transform:scale(1);opacity:.5}}50%{{transform:scale(.7);opacity:.2}}}}
    .emotion-name {{text-align:center;color:{color};font-size:22px;font-weight:bold;margin-top:0;}}
    </style></head><body><div class="avatar-container"><div class="avatar">{emoji}</div></div><div class="emotion-name">{emotion}</div></body></html>"""


def show_avatar(emotion):
    custom_image = st.session_state.custom_avatars.get(emotion)
    if custom_image:
        st.image(custom_image, width=180)
        st.markdown(f"<h4 style='text-align:center'>{emotion}</h4>", unsafe_allow_html=True)
    else:
        components.html(avatar_svg(emotion), height=250, scrolling=False)

# =========================================================
# 8. 側邊欄
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
            uploaded = st.file_uploader(f"上傳{emotion}圖片", type=["png", "jpg", "jpeg"], key=f"avatar_{emotion}")
            if uploaded:
                st.session_state.custom_avatars[emotion] = uploaded.getvalue()
            if emotion in st.session_state.custom_avatars:
                st.image(st.session_state.custom_avatars[emotion], width=120)
                if st.button("恢復預設角色", key=f"reset_{emotion}"):
                    del st.session_state.custom_avatars[emotion]
                    st.rerun()
    st.divider()
    st.markdown("### 📊 本次連線紀錄")
    st.write(f"目前分析：{len(st.session_state.history)} 筆")
    if st.button("清除本次畫面紀錄", use_container_width=True):
        st.session_state.history = []
        st.session_state.analysis_result = None
        st.rerun()

# =========================================================
# 9. 頁面與分頁
# =========================================================
st.title("💗 AI 情緒智慧分析系統")
st.markdown("透過 Qwen2.5-0.5B 與 LoRA，理解文字中的情緒，並提供情緒原因分析與主動關懷。")
st.info("隱私告知：為提供情緒紀錄與管理功能，您送出的文字及分析結果會儲存在資料庫，系統管理者可查看這些紀錄。請勿輸入密碼、身分證字號或其他不必要的敏感個資。")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["💗 情緒主動分析", "📊 情緒趨勢分析", "📖 系統介紹", "🔐 管理者後台"])

# =========================================================
# 10. 情緒分析
# =========================================================
with tab1:
    st.subheader("💬 輸入你的文字")
    user_text = st.text_area("分享今天的心情、生活事件或想法：", height=150, placeholder="例如：今天報告終於完成了，真的覺得很開心！")
    consent = st.checkbox("我已閱讀上方隱私告知，並同意將輸入文字及分析結果儲存，供系統管理者管理與查看。")
    analyze_button = st.button("✨ 開始主動分析", type="primary", use_container_width=True, disabled=not consent)

    if analyze_button:
        if not user_text.strip():
            st.warning("請先輸入文字，再進行情緒分析。")
        else:
            with st.spinner("AI 正在理解你的文字與情緒..."):
                result = analyze_emotion(user_text.strip())
                result["input_text"] = user_text.strip()
                result["time"] = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.analysis_result = result
                st.session_state.history.append(result)
                saved, message = save_record(result)
                if saved:
                    st.success("本次分析已完成，紀錄已安全儲存。")
                else:
                    st.error(f"分析完成，但資料庫儲存失敗：{message}")

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
            st.metric("情緒強度", f"{result['intensity']} / 5")
            st.markdown("**📝 文字類型**")
            st.write(result["text_type"])
            st.markdown("**🔍 情緒原因分析**")
            st.write(result["reason"])
        st.divider()
        st.subheader("💌 AI 主動關懷")
        st.info(result["advice"])
        st.caption("以上是 AI 根據文字推測的情緒，不代表專業心理診斷。")
        with st.expander("查看本次分析的原始文字"):
            st.write(result["input_text"])

# =========================================================
# 11. 個人趨勢與紀錄
# =========================================================
with tab2:
    st.subheader("📊 我的情緒歷史與趨勢")
    db_rows = fetch_my_records()
    if db_rows:
        df = pd.DataFrame(db_rows)
        df["time"] = pd.to_datetime(df["created_at"], errors="coerce")
    elif st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df = pd.DataFrame()

    if df.empty:
        st.info("目前還沒有情緒紀錄，請先到「情緒主動分析」進行分析。")
    else:
        display_df = df[["time", "input_text", "emotion", "text_type", "intensity"]].copy()
        display_df.columns = ["分析時間", "輸入文字", "情緒", "文字類型", "情緒強度"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("### 📈 情緒分布")
        st.bar_chart(df["emotion"].value_counts().reindex(EMOTIONS, fill_value=0))
        st.divider()
        st.markdown("### 📉 情緒強度變化")
        chart_df = df.dropna(subset=["time"])[["time", "intensity"]].set_index("time").sort_index()
        if not chart_df.empty:
            st.line_chart(chart_df)
        st.download_button("⬇️ 下載我的情緒紀錄 CSV", data=df.to_csv(index=False, encoding="utf-8-sig"), file_name="my_emotion_history.csv", mime="text/csv", use_container_width=True)

# =========================================================
# 12. 系統介紹
# =========================================================
with tab3:
    st.subheader("📖 系統功能介紹")
    st.markdown("""
### 💗 AI 情緒智慧分析系統
本系統結合小型語言模型與 LoRA 微調技術，讓 AI 理解使用者輸入的文字，分析可能的情緒狀態，並提供適當的關懷與建議。

### 🤖 使用技術
- **Qwen2.5-0.5B-Instruct**：基礎語言模型。
- **LoRA**：透過低秩適應技術進行情緒辨識。
- **Python、PyTorch**：模型與程式開發。
- **Streamlit**：互動式網頁介面。
- **Supabase**：儲存情緒分析紀錄。

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
1. 自動辨識文字類型與主要情緒。
2. 根據語意推測情緒原因。
3. 提供主動關懷與建議。
4. 透過 Q 版動態角色呈現情緒。
5. 支援自訂情緒角色圖片。
6. 保存情緒歷史並分析變化趨勢。
7. 支援個人紀錄 CSV 下載及管理者紀錄管理。

本系統僅提供文字情緒推測，不代表使用者真實心理狀態，也不是心理診斷工具。
""")

# =========================================================
# 13. 管理者後台
# =========================================================
with tab4:
    st.subheader("🔐 管理者登入")
    admin_password = st.secrets.get("ADMIN_PASSWORD", "")

    if not admin_password:
        st.error("尚未在 Streamlit Secrets 設定 ADMIN_PASSWORD。")
    elif not st.session_state.admin_authenticated:
        with st.form("admin_login_form"):
            entered_password = st.text_input("管理者密碼", type="password")
            login_clicked = st.form_submit_button("登入管理者後台", use_container_width=True)
        if login_clicked:
            if hmac.compare_digest(entered_password, admin_password):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("密碼錯誤，請再試一次。")
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
                st.info("目前資料庫還沒有使用者紀錄。")
            else:
                admin_df = pd.DataFrame(all_rows)
                admin_df["created_at"] = pd.to_datetime(admin_df["created_at"], errors="coerce")
                st.metric("全部分析紀錄", f"{len(admin_df):,} 筆")
                st.caption("管理者可以查看所有使用者提交的文字及分析結果。請妥善保護這些資料。")

                filter1, filter2 = st.columns(2)
                with filter1:
                    selected_emotions = st.multiselect("依情緒篩選", EMOTIONS, default=EMOTIONS)
                with filter2:
                    today = date.today()
                    date_range = st.date_input("依日期篩選", value=(today.replace(day=1), today))

                filtered = admin_df[admin_df["emotion"].isin(selected_emotions)]
                if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered = filtered[
                        (filtered["created_at"].dt.date >= start_date) &
                        (filtered["created_at"].dt.date <= end_date)
                    ]
                elif isinstance(date_range, date):
                    filtered = filtered[filtered["created_at"].dt.date == date_range]

                st.write(f"符合條件：{len(filtered):,} 筆")
                cols = ["created_at", "session_id", "input_text", "emotion", "text_type", "reason", "intensity", "advice"]
                view_df = filtered[cols].copy()
                view_df.columns = ["分析時間", "匿名連線代碼", "使用者輸入", "情緒", "文字類型", "情緒原因", "強度", "AI 建議"]
                st.dataframe(view_df, use_container_width=True, hide_index=True)

                csv_bytes = filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "⬇️ 下載篩選後的全部紀錄 CSV",
                    data=csv_bytes,
                    file_name=f"all_emotion_records_{date.today().isoformat()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as exc:
            st.error(f"讀取管理者紀錄失敗：{exc}")
            st.caption("請確認 Supabase 資料表欄位與 Streamlit Secrets 設定正確。")

st.divider()
st.markdown("<div style='text-align:center;color:gray;'>AI 情緒智慧分析系統 | Qwen2.5 + LoRA</div>", unsafe_allow_html=True)
