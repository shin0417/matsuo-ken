# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- カスタムCSS (毒々しいレインボー) ---
st.markdown("""
<style>
    @keyframes rainbow-bg {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes rainbow-text {
        0% { color: red; }
        15% { color: orange; }
        30% { color: yellow; }
        45% { color: lime; }
        60% { color: cyan; }
        75% { color: blue; }
        90% { color: magenta; }
        100% { color: red; }
    }

    /* 全体設定 - 動くレインボーグラデーション */
    .stApp {
        background: linear-gradient(120deg, red, orange, yellow, lime, cyan, blue, magenta, red);
        background-size: 1800% 1800%;
        animation: rainbow-bg 18s ease infinite;
    }

    /* サイドバー - 異なるレインボーグラデーション */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, fuchsia, purple, indigo, teal, green, gold);
        padding: 15px;
        border-right: 5px dashed yellow;
    }
    [data-testid="stSidebar"] * {
        color: white !important; /* 白文字強制 */
        text-shadow: 1px 1px 2px black;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: bold;
        font-size: 1.1em;
        background-color: rgba(0,0,0,0.3);
        padding: 5px;
        border-radius: 5px;
    }

    /* メインコンテンツ */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.7); /* 半透明白背景 */
        padding: 2rem 5rem;
        border-radius: 15px;
        box-shadow: 0 0 20px lime;
    }

    /* タイトル - レインボーテキストアニメーション */
    h1 {
        text-align: center;
        padding-bottom: 20px;
        font-size: 3em;
        font-weight: bold;
        animation: rainbow-text 5s linear infinite;
        text-shadow: 2px 2px 4px black;
    }

    /* サブヘッダー - 交互色 */
    h2 {
        color: magenta;
        border-bottom: 3px dotted yellow;
        padding-bottom: 5px;
        margin-top: 30px;
        text-shadow: 1px 1px 2px blue;
    }
    h3 {
        color: cyan;
        border-bottom: 3px dotted red;
        padding-bottom: 5px;
        margin-top: 25px;
        text-shadow: 1px 1px 2px purple;
    }
    h4, h5, h6 {
        color: lime;
        margin-top: 20px;
        text-decoration: underline wavy orange;
    }


    /* ボタン - 点滅レインボーボーダー */
    @keyframes rainbow-border {
        0% { border-color: red; box-shadow: 0 0 10px red; }
        15% { border-color: orange; box-shadow: 0 0 10px orange; }
        30% { border-color: yellow; box-shadow: 0 0 10px yellow; }
        45% { border-color: lime; box-shadow: 0 0 10px lime; }
        60% { border-color: cyan; box-shadow: 0 0 10px cyan; }
        75% { border-color: blue; box-shadow: 0 0 10px blue; }
        90% { border-color: magenta; box-shadow: 0 0 10px magenta; }
        100% { border-color: red; box-shadow: 0 0 10px red; }
    }
    .stButton>button {
        background: linear-gradient(45deg, blue, magenta);
        color: white;
        border: 5px solid red; /* 初期色 */
        padding: 15px 30px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.2em;
        text-shadow: 1px 1px 2px black;
        animation: rainbow-border 3s linear infinite;
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.1);
        background: linear-gradient(45deg, magenta, blue);
    }
    .stButton>button:active {
        transform: scale(0.9);
    }


    /* テキスト入力 */
    .stTextArea textarea {
        background-color: #ffebcd; /* Blanched Almond */
        border: 3px dashed purple;
        border-radius: 8px;
        color: navy;
        font-size: 1.1em;
        box-shadow: inset 0 0 10px cyan;
    }

    /* 回答表示エリア */
    [data-testid="stMarkdownContainer"] p {
         line-height: 1.7;
         font-size: 1.1em;
         background: linear-gradient(to right, #e0ffff, #f0fff0); /* Light Cyan to Honeydew */
         padding: 20px;
         border-radius: 10px;
         border: 3px dotted magenta;
         color: black;
         text-shadow: 1px 1px 1px lightgrey;
         box-shadow: 5px 5px 15px rgba(0,0,255,0.3);
    }


    /* フォーム */
    [data-testid="stForm"] {
        background: linear-gradient(135deg, yellow, lime, cyan);
        padding: 25px;
        border-radius: 15px;
        border: 5px groove red;
        box-shadow: inset 0 0 15px orange;
    }

    /* ラジオボタン（フォーム内） */
    [data-testid="stForm"] .stRadio label {
        color: navy !important; /* Important to override sidebar style */
        background-color: rgba(255,255,0,0.5); /* Semi-transparent Yellow */
        padding: 8px;
        border-radius: 5px;
        text-shadow: none;
        border: 1px solid orange;
    }


    /* エキスパンダー */
    .stExpander {
        border: 3px solid blue;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: rgba(0, 255, 0, 0.1); /* Transparent Green */
    }
    .stExpander header {
        background: linear-gradient(90deg, red, orange, yellow);
        color: black;
        font-weight: bold;
        padding: 10px 15px;
        border-radius: 7px 7px 0 0;
        border-bottom: 3px solid black;
        text-shadow: 1px 1px 1px white;
    }
     .stExpander header:hover {
        background: linear-gradient(90deg, yellow, orange, red);
     }

    /* メトリクス */
    [data-testid="stMetric"] {
        background: linear-gradient(pink, lightcoral);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid purple;
        box-shadow: 0 0 8px magenta;
    }
    [data-testid="stMetric"] label {
        font-weight: bold;
        color: darkred;
        text-shadow: 1px 1px 1px white;
    }
     [data-testid="stMetric"] div:nth-of-type(2) { /* Value */
         font-size: 1.8em;
         color: white;
         text-shadow: 1px 1px 3px black;
     }

    /* データフレームとテーブル */
    .stDataFrame, .stTable {
        border: 4px groove cyan;
        box-shadow: 0 0 10px blue;
    }
     .stDataFrame thead th, .stTable thead th {
         background: linear-gradient(blue, indigo);
         color: yellow;
         font-size: 1.1em;
         text-shadow: 1px 1px 2px black;
     }
     .stDataFrame tbody tr:nth-child(even), .stTable tbody tr:nth-child(even) {
        background-color: #e6e6fa; /* Lavender */
     }
     .stDataFrame tbody tr:nth-child(odd), .stTable tbody tr:nth-child(odd) {
        background-color: #fff0f5; /* Lavender Blush */
     }
     .stDataFrame td, .stTable td {
         border: 1px dashed magenta;
         color: darkblue;
     }

    /* 区切り線 */
    hr {
      border: none;
      height: 5px;
      background: linear-gradient(to right, red, orange, yellow, lime, cyan, blue, magenta);
      margin-top: 40px;
      margin-bottom: 40px;
    }

</style>
""", unsafe_allow_html=True)

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None
pipe = llm.load_model()

# --- Streamlit アプリケーション ---
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)


# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")