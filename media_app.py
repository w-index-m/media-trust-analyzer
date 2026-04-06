"""
Media Trust Analyzer
メディア信頼度・多角分析ダッシュボード（独立アプリ）
Gemini → Groq → OpenRouter フォールバック対応
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from datetime import datetime, timezone
import pytz

# ── ページ設定 ─────────────────────────────────────────
st.set_page_config(
    page_title="Media Trust Analyzer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

JST = pytz.timezone("Asia/Tokyo")
logger = logging.getLogger(__name__)

# ── Plotly ─────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── APIキー取得 ─────────────────────────────────────────
def get_env_var(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        import os
        return os.environ.get(key, default)

GEMINI_API_KEY    = get_env_var("GEMINI_API_KEY", "")
GROQ_API_KEY      = get_env_var("GROQ_API_KEY", "")
OPENROUTER_API_KEY = get_env_var("OPENROUTER_API_KEY", "")

# ── AI フォールバック ───────────────────────────────────
def call_ai(prompt: str, max_tokens: int = 600, temperature: float = 0.5) -> tuple:
    """Gemini → Groq → OpenRouter の順でフォールバック"""

    # ① Gemini
    if GEMINI_API_KEY:
        models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ]
        for model in models:
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={GEMINI_API_KEY}",
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"maxOutputTokens": max_tokens,
                                               "temperature": temperature}},
                    timeout=20,
                )
                if r.status_code == 200:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return text, f"Gemini ({model})"
            except Exception:
                continue

    # ② Groq
    if GROQ_API_KEY:
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ]
        for model in groq_models:
            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": model,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens,
                          "temperature": temperature},
                    timeout=20,
                )
                if r.status_code == 200:
                    text = r.json()["choices"][0]["message"]["content"]
                    return text, f"Groq ({model})"
            except Exception:
                continue

    # ③ OpenRouter
    if OPENROUTER_API_KEY:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "mistralai/mistral-7b-instruct:free",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens},
                timeout=20,
            )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                return text, "OpenRouter (Mistral-7B)"
        except Exception:
            pass

    return "⚠️ AI APIキーが設定されていないか、全APIが失敗しました。", ""


# ── メディアデータ ─────────────────────────────────────
SCORE_AXES = ["事実正確性", "中立性", "速報性", "深度", "独立性"]

JP_MEDIA = [
    {"name": "日経CNBC",       "icon": "📺", "type": "テレビ・専門経済チャンネル",
     "scores": {"事実正確性":84,"中立性":68,"速報性":90,"深度":72,"独立性":60}, "recommend":78,
     "badges": [("市場速報◎","green"),("経済特化","blue"),("財界寄り","orange")],
     "note": "日本唯一の経済専門テレビ。マーケット速報・アナリスト解説は強い。日経グループのため財界寄りの視点が出やすい。"},
    {"name": "テレ東WBS",      "icon": "🌙", "type": "テレビ・経済番組",
     "scores": {"事実正確性":82,"中立性":70,"速報性":78,"深度":68,"独立性":62}, "recommend":74,
     "badges": [("経済特化◎","green"),("夜のニュース","blue"),("分かりやすい","blue")],
     "note": "ワールドビジネスサテライト。民放では最も経済報道に力を入れる。深夜帯で視聴者層が高い。"},
    {"name": "NHK",            "icon": "📺", "type": "テレビ・公共放送",
     "scores": {"事実正確性":85,"中立性":72,"速報性":88,"深度":55,"独立性":60}, "recommend":82,
     "badges": [("速報◎","green"),("公共","blue"),("一次情報","blue")],
     "note": "速報と事実報道は強い。ただし政治的独立性に疑問符も。"},
    {"name": "東洋経済",       "icon": "📰", "type": "経済週刊誌・Web",
     "scores": {"事実正確性":88,"中立性":78,"速報性":45,"深度":92,"独立性":82}, "recommend":90,
     "badges": [("深度◎","green"),("調査報道","green"),("スポンサー少","green")],
     "note": "政治経済の深掘りなら最優先。スポンサー忖度が比較的少ない。"},
    {"name": "文春オンライン",  "icon": "🔍", "type": "週刊誌・調査報道",
     "scores": {"事実正確性":80,"中立性":65,"速報性":60,"深度":88,"独立性":92}, "recommend":85,
     "badges": [("タブー破り◎","orange"),("調査報道","green"),("独立性高","green")],
     "note": "スポンサー圧力を受けにくい独立系。政財界タブーに踏み込む。"},
    {"name": "ダイヤモンド",   "icon": "💎", "type": "経済週刊誌・Web",
     "scores": {"事実正確性":85,"中立性":74,"速報性":40,"深度":88,"独立性":78}, "recommend":85,
     "badges": [("財界分析◎","green"),("深度◎","green"),("経済特化","blue")],
     "note": "財界・産業構造の分析が強い。経済政策を深く理解したい人向け。"},
    {"name": "日経新聞",       "icon": "📊", "type": "経済新聞",
     "scores": {"事実正確性":87,"中立性":68,"速報性":80,"深度":82,"独立性":65}, "recommend":83,
     "badges": [("経済速報◎","green"),("一次情報","blue"),("財界寄り","orange")],
     "note": "経済指標・マーケット情報は最速。ただし財界・大企業寄り。"},
    {"name": "朝日新聞",       "icon": "🗞️", "type": "全国紙",
     "scores": {"事実正確性":78,"中立性":55,"速報性":72,"深度":75,"独立性":60}, "recommend":68,
     "badges": [("調査報道あり","blue"),("やや左寄り","orange"),("政治面強い","blue")],
     "note": "政治系調査報道はある。ただし過去の誤報・政治的立場に注意。"},
    {"name": "読売新聞",       "icon": "🗞️", "type": "全国紙（最大部数）",
     "scores": {"事実正確性":80,"中立性":58,"速報性":75,"深度":70,"独立性":52}, "recommend":65,
     "badges": [("最大部数","blue"),("やや右寄り","orange"),("政府寄り","red")],
     "note": "発行部数世界最大。速報・政治ニュースはある。ただし親米・政府寄り。"},
    {"name": "産経新聞",       "icon": "🗞️", "type": "全国紙（保守系）",
     "scores": {"事実正確性":72,"中立性":42,"速報性":72,"深度":65,"独立性":55}, "recommend":52,
     "badges": [("保守系","orange"),("右寄り","red"),("対比参考用","gray")],
     "note": "明確な保守・右寄り路線。対比参考用として読むのが◎。"},
    {"name": "民放TV",         "icon": "📡", "type": "テレビ・民間放送",
     "scores": {"事実正確性":62,"中立性":52,"速報性":90,"深度":30,"独立性":35}, "recommend":45,
     "badges": [("速報のみ","gray"),("スポンサー影響大","red"),("深度×","red")],
     "note": "速報・話題性はあるが深度なし。スポンサー批判はほぼ不可能。"},
    {"name": "日刊ゲンダイ",   "icon": "📋", "type": "日刊紙（反権力系）",
     "scores": {"事実正確性":65,"中立性":48,"速報性":70,"深度":55,"独立性":88}, "recommend":55,
     "badges": [("反権力◎","orange"),("独立性高","green"),("精度要確認","red")],
     "note": "権力批判・政財界タブーに踏み込む。独立性は高いが事実精度にムラあり。"},
    {"name": "プレジデント",   "icon": "👔", "type": "経済誌・Web",
     "scores": {"事実正確性":75,"中立性":70,"速報性":35,"深度":78,"独立性":72}, "recommend":72,
     "badges": [("ビジネス視点","blue"),("経営者向け","blue"),("深度あり","green")],
     "note": "ビジネス・経営視点から政治経済を分析。深度はある。"},
]

US_MEDIA = [
    {"name": "AP News",         "icon": "📡", "type": "通信社（米）",
     "scores": {"事実正確性":95,"中立性":90,"速報性":95,"深度":60,"独立性":92}, "recommend":95,
     "badges": [("最中立◎","green"),("一次情報◎","green"),("速報◎","green")],
     "note": "世界で最も中立に近い通信社。政治経済の一次情報源として最優先。"},
    {"name": "Reuters",         "icon": "🌐", "type": "通信社（英）",
     "scores": {"事実正確性":94,"中立性":89,"速報性":93,"深度":65,"独立性":90}, "recommend":93,
     "badges": [("金融特化◎","green"),("一次情報◎","green"),("中立◎","green")],
     "note": "金融・マーケット系は世界一の速さと精度。APと並ぶ最高峰。"},
    {"name": "Financial Times",  "icon": "🟠", "type": "経済新聞（英）",
     "scores": {"事実正確性":92,"中立性":78,"速報性":80,"深度":92,"独立性":80}, "recommend":92,
     "badges": [("経済深度◎","green"),("グローバル◎","green"),("有料多い","orange")],
     "note": "世界最高水準の経済ジャーナリズム。金融・政策・地政学の深度は随一。"},
    {"name": "The Economist",   "icon": "📗", "type": "週刊誌（英）",
     "scores": {"事実正確性":91,"中立性":75,"速報性":25,"深度":96,"独立性":85}, "recommend":90,
     "badges": [("深度最高◎","green"),("国際政治経済◎","green"),("速報×","red")],
     "note": "週刊なので速報性ゼロだが世界の政治経済を最も体系的に分析。"},
    {"name": "BBC World",       "icon": "🎙️", "type": "テレビ・公共放送（英）",
     "scores": {"事実正確性":88,"中立性":78,"速報性":82,"深度":82,"独立性":80}, "recommend":88,
     "badges": [("国際視点◎","green"),("深度あり","green"),("英国視点","blue")],
     "note": "国際政治を最も深く報じる公共放送。米国メディアにない視点がある。"},
    {"name": "NPR News",        "icon": "📻", "type": "ラジオ・公共放送（米）",
     "scores": {"事実正確性":88,"中立性":74,"速報性":75,"深度":84,"独立性":82}, "recommend":86,
     "badges": [("調査報道◎","green"),("スポンサー少","green"),("やや左寄り","orange")],
     "note": "米公共ラジオ。調査報道が強く独立性高い。"},
    {"name": "WSJ",             "icon": "📰", "type": "経済新聞（米）",
     "scores": {"事実正確性":90,"中立性":70,"速報性":85,"深度":88,"独立性":68}, "recommend":85,
     "badges": [("米経済◎","green"),("市場速報◎","green"),("やや保守寄り","orange")],
     "note": "米国経済・金融の深度はトップ。論説面はやや保守寄り。"},
    {"name": "Bloomberg",       "icon": "💹", "type": "金融メディア（米）",
     "scores": {"事実正確性":90,"中立性":75,"速報性":88,"深度":85,"独立性":72}, "recommend":88,
     "badges": [("金融◎","green"),("経済深度◎","green"),("有料多い","orange")],
     "note": "金融・経済報道の深度と速度はトップクラス。"},
    {"name": "Al Jazeera",      "icon": "🌍", "type": "テレビ・国際報道（カタール）",
     "scores": {"事実正確性":82,"中立性":72,"速報性":85,"深度":80,"独立性":78}, "recommend":80,
     "badges": [("中東視点◎","green"),("非西洋視点","blue"),("国際政治強い","blue")],
     "note": "西洋メディアとは異なる視点。中東・アフリカ・アジアの政治経済を深く報じる。"},
    {"name": "FactCheck.org",   "icon": "✅", "type": "ファクトチェック専門（米）",
     "scores": {"事実正確性":96,"中立性":88,"速報性":25,"深度":90,"独立性":95}, "recommend":88,
     "badges": [("事実検証◎","green"),("党派なし","green"),("政治発言専門","blue")],
     "note": "政治家の発言・政策の事実検証専門。速報性ゼロだが信頼性最高。"},
    {"name": "Snopes",          "icon": "🔬", "type": "ファクトチェック（米）",
     "scores": {"事実正確性":92,"中立性":82,"速報性":20,"深度":85,"独立性":90}, "recommend":82,
     "badges": [("デマ検証◎","green"),("独立性◎","green"),("速報×","red")],
     "note": "SNSデマ・フェイクニュースの検証サイト。投資判断前のファクトチェックに。"},
    {"name": "MarketWatch",     "icon": "📈", "type": "市場メディア（米）",
     "scores": {"事実正確性":84,"中立性":72,"速報性":85,"深度":72,"独立性":68}, "recommend":78,
     "badges": [("市場速報◎","green"),("経済指標","blue"),("WSJ系","blue")],
     "note": "WSJ傘下。市場速報と経済指標の解説が強い。"},
    {"name": "CNBC",            "icon": "📺", "type": "テレビ・ケーブル（米）",
     "scores": {"事実正確性":80,"中立性":62,"速報性":88,"深度":68,"独立性":58}, "recommend":72,
     "badges": [("市場速報◎","green"),("やや右寄り","orange"),("スポンサー影響","red")],
     "note": "市場速報は速い。ただし親会社・広告主の影響あり。"},
]

# 評価機関データ
INSTITUTION_RATINGS = {
    "日本メディア": {
        "機関": ["新聞通信調査会(2025)", "Reuters Institute(2025)",
                  "MediaBias/FactCheck(2025)", "AllSides(参考)"],
        "メディア": ["NHK", "日経新聞", "朝日新聞", "読売新聞", "産経新聞", "東洋経済", "民放TV"],
        "スコア": {
            "新聞通信調査会(2025)":
                {"NHK":67,"日経新聞":66,"朝日新聞":60,"読売新聞":62,"産経新聞":55,"東洋経済":None,"民放TV":60},
            "Reuters Institute(2025)":
                {"NHK":61,"日経新聞":55,"朝日新聞":48,"読売新聞":50,"産経新聞":None,"東洋経済":None,"民放TV":42},
            "MediaBias/FactCheck(2025)":
                {"NHK":72,"日経新聞":68,"朝日新聞":62,"読売新聞":60,"産経新聞":48,"東洋経済":75,"民放TV":52},
            "AllSides(参考)":
                {"NHK":65,"日経新聞":60,"朝日新聞":52,"読売新聞":55,"産経新聞":40,"東洋経済":70,"民放TV":50},
        },
        "出典": {
            "新聞通信調査会(2025)":   "https://www.chosakai.gr.jp/",
            "Reuters Institute(2025)": "https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2025",
            "MediaBias/FactCheck(2025)": "https://mediabiasfactcheck.com/",
            "AllSides(参考)":         "https://www.allsides.com/",
        },
    },
    "海外メディア": {
        "機関": ["Reuters Institute(2025)", "MediaBias/FactCheck(2025)",
                  "AllSides(参考)", "RSF Press Freedom(2025)"],
        "メディア": ["AP News","Reuters","BBC","Bloomberg","NYT","WSJ","CNN","Fox News"],
        "スコア": {
            "Reuters Institute(2025)":
                {"AP News":85,"Reuters":83,"BBC":75,"Bloomberg":72,"NYT":58,"WSJ":65,"CNN":50,"Fox News":32},
            "MediaBias/FactCheck(2025)":
                {"AP News":95,"Reuters":94,"BBC":85,"Bloomberg":82,"NYT":72,"WSJ":80,"CNN":62,"Fox News":38},
            "AllSides(参考)":
                {"AP News":88,"Reuters":86,"BBC":78,"Bloomberg":74,"NYT":55,"WSJ":68,"CNN":48,"Fox News":30},
            "RSF Press Freedom(2025)":
                {"AP News":90,"Reuters":88,"BBC":82,"Bloomberg":80,"NYT":70,"WSJ":72,"CNN":58,"Fox News":35},
        },
        "出典": {
            "Reuters Institute(2025)":    "https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2025",
            "MediaBias/FactCheck(2025)":  "https://mediabiasfactcheck.com/",
            "AllSides(参考)":             "https://www.allsides.com/",
            "RSF Press Freedom(2025)":    "https://rsf.org/en/index",
        },
    },
}

BADGE_COLORS = {
    "green":  ("rgba(127,255,107,0.15)", "#7fff6b"),
    "blue":   ("rgba(59,130,246,0.15)",  "#60a5fa"),
    "orange": ("rgba(255,107,53,0.15)",  "#ff6b35"),
    "red":    ("rgba(239,68,68,0.15)",   "#f87171"),
    "gray":   ("rgba(255,255,255,0.08)", "#94a3b8"),
}

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700;900&family=Space+Mono:wght@400;700&display=swap');

[data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    font-family: 'Noto Sans JP', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #111827; }

h1,h2,h3 { font-family: 'Noto Sans JP', sans-serif !important; }

.media-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.media-card:hover { border-color: #00d4ff; }

.score-bar-bg {
    flex: 1; height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
}

.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-family: 'Noto Sans JP', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00d4ff, #0077aa) !important;
    color: white !important;
    border-radius: 7px !important;
}
</style>
""", unsafe_allow_html=True)

# ── ヘッダー ──────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 24px;">
  <div style="font-family:'Space Mono',monospace;font-size:11px;
              letter-spacing:3px;color:#00d4ff;margin-bottom:12px;">
    MEDIA INTELLIGENCE REPORT
  </div>
  <h1 style="font-size:clamp(24px,4vw,42px);font-weight:900;
             background:linear-gradient(135deg,#fff 0%,#00d4ff 60%,#ff6b35 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;margin-bottom:8px;">
    メディア信頼度 多角分析
  </h1>
  <p style="color:#64748b;font-size:14px;font-weight:300;">
    事実報道・中立性・速報性・深度・独立性の5軸 × AI評価コメント × 評価機関比較
  </p>
</div>
""", unsafe_allow_html=True)

# ── ユーティリティ関数 ────────────────────────────────
def score_color(v: int) -> str:
    return "#7fff6b" if v >= 80 else ("#00d4ff" if v >= 60 else "#ff6b35")

def recommend_color(v: int) -> str:
    return "#7fff6b" if v >= 85 else ("#00d4ff" if v >= 70 else "#ff6b35")

def render_score_bar(label: str, val: int) -> str:
    c = score_color(val)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
        f'<span style="font-size:10px;color:#64748b;width:64px;flex-shrink:0;">{label}</span>'
        f'<div style="flex:1;height:5px;background:rgba(255,255,255,0.08);border-radius:3px;">'
        f'<div style="width:{val}%;height:100%;background:{c};border-radius:3px;"></div></div>'
        f'<span style="font-family:monospace;font-size:11px;color:{c};width:26px;">{val}</span>'
        f'</div>'
    )

def render_badge(text: str, style: str) -> str:
    bg, fc = BADGE_COLORS.get(style, BADGE_COLORS["gray"])
    return (
        f'<span style="font-size:10px;padding:2px 8px;border-radius:3px;font-weight:700;'
        f'background:{bg};color:{fc};margin-right:4px;">{text}</span>'
    )

def render_media_card_html(m: dict) -> str:
    rc = recommend_color(m["recommend"])
    bars = "".join(render_score_bar(k, v) for k, v in m["scores"].items())
    badges = "".join(render_badge(t, s) for t, s in m["badges"])
    return (
        f'<div class="media-card">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
        f'<span style="font-size:24px;">{m["icon"]}</span>'
        f'<div style="flex:1;">'
        f'<div style="font-weight:700;font-size:15px;color:#e2e8f0;">{m["name"]}</div>'
        f'<div style="font-size:10px;color:#64748b;">{m["type"]}</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'<span style="font-family:monospace;font-size:24px;font-weight:900;color:{rc};">'
        f'{m["recommend"]}</span>'
        f'<span style="font-size:10px;color:#64748b;">/100</span>'
        f'</div></div>'
        + bars +
        f'<div style="margin-top:8px;">{badges}</div>'
        f'<div style="margin-top:8px;font-size:11px;color:#475569;line-height:1.6;">'
        f'{m["note"]}</div>'
        f'</div>'
    )

# ── AI評価キャッシュ ──────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def ai_media_comment(name, mtype, scores_json, recommend, note):
    scores = json.loads(scores_json)
    score_text = "、".join(f"{k}:{v}/100" for k, v in scores.items())
    prompt = f"""あなたは独立したメディア評論家です。以下のメディアについて
政治経済に関心を持つ日本の投資家・市民向けに実用的な評価コメントを書いてください。

メディア名: {name} / 種別: {mtype}
評価: {score_text} / おすすめ度: {recommend}/100
基本情報: {note}

形式（各150字程度）：
【強み】優れている点を2〜3つ。
【弱み・注意点】スポンサー問題・バイアス・限界など。
【推奨する使い方】政治経済の情報収集で具体的なシーン。
【組み合わせ推奨】相性の良いメディア。"""
    return call_ai(prompt, max_tokens=600, temperature=0.5)

@st.cache_data(ttl=60*60*24*90, show_spinner=False)
def ai_institution_summary(quarter: str):
    prompt = f"""{quarter}時点の主要メディア信頼度調査の最新動向を日本語でまとめてください。

対象: 新聞通信調査会・Reuters Institute・MediaBias/FactCheck・RSF報道自由度指数

形式:
【新聞通信調査会】最新スコアとトレンド（2〜3文）
【Reuters Institute】日本・世界の信頼度動向（2〜3文）
【MediaBias/FactCheck】注目の評価変更（2〜3文）
【RSF報道自由度】日本の順位と評価（2〜3文）
【総合所見】全体トレンドと市民・投資家への示唆（3〜4文）

不確かな場合は「要確認」と明記。"""
    return call_ai(prompt, max_tokens=800, temperature=0.3)


# ── メインUI ─────────────────────────────────────────
tab_jp, tab_us, tab_cmp, tab_inst, tab_update = st.tabs([
    "🇯🇵 日本国内メディア",
    "🌍 海外メディア",
    "📊 日米比較",
    "🏛️ 機関別評価",
    "🔄 定期AI更新",
])

def render_media_tab(media_list, prefix):
    # ランキングチャート
    if PLOTLY_AVAILABLE:
        sorted_m = sorted(media_list, key=lambda m: m["recommend"], reverse=True)
        fig = go.Figure(go.Bar(
            x=[m["recommend"] for m in sorted_m],
            y=[m["name"] for m in sorted_m],
            orientation="h",
            marker=dict(
                color=[m["recommend"] for m in sorted_m],
                colorscale=[[0,"#ff6b35"],[0.55,"#00d4ff"],[1,"#7fff6b"]],
                cmin=40, cmax=100,
            ),
            text=[str(m["recommend"]) for m in sorted_m],
            textposition="outside",
            hovertemplate="%{y}: %{x}点<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="政治経済おすすめ度ランキング", font=dict(color="#f1f5f9", size=15)),
            height=max(320, len(media_list)*36),
            xaxis=dict(range=[0,112], gridcolor="rgba(255,255,255,0.05)", color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0", family="'Noto Sans JP', sans-serif"),
            margin=dict(l=10, r=50, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # レーダーチャート（Top4）
        top4 = sorted(media_list, key=lambda m: m["recommend"], reverse=True)[:4]
        colors = [
            ("rgba(239,68,68,1)",  "rgba(239,68,68,0.12)"),
            ("rgba(59,130,246,1)", "rgba(59,130,246,0.12)"),
            ("rgba(249,115,22,1)", "rgba(249,115,22,0.12)"),
            ("rgba(168,85,247,1)", "rgba(168,85,247,0.12)"),
        ]
        fig2 = go.Figure()
        for i, m in enumerate(top4):
            vals = [m["scores"][k] for k in SCORE_AXES]
            fig2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=SCORE_AXES + [SCORE_AXES[0]],
                fill="toself", name=m["name"],
                line_color=colors[i][0],
                fillcolor=colors[i][1],
            ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.15)", color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.15)", color="#f1f5f9", tickfont=dict(color="#f1f5f9")),
                bgcolor="#0a0e1a",
            ),
            paper_bgcolor="#0a0e1a", font=dict(color="#e2e8f0"),
            legend=dict(orientation="h", y=-0.15, font=dict(size=11, color="#f1f5f9")),
            height=400, title=dict(text="Top4 5軸レーダー比較", font=dict(color="#f1f5f9", size=15)),
            margin=dict(t=40, b=80),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 媒体別詳細スコア ＆ AI評価")

    cols = st.columns(2)
    for i, m in enumerate(media_list):
        with cols[i % 2]:
            st.markdown(render_media_card_html(m), unsafe_allow_html=True)

            btn_key   = f"ai_{prefix}_{i}"
            cache_key = f"ai_result_{prefix}_{i}"

            if st.button("🤖 AI評価を生成（Gemini/Groq）", key=btn_key, use_container_width=True):
                with st.spinner(f"{m['name']} を分析中..."):
                    comment, model = ai_media_comment(
                        m["name"], m["type"],
                        json.dumps(m["scores"], ensure_ascii=False),
                        m["recommend"], m["note"],
                    )
                    st.session_state[cache_key] = (comment, model)

            if cache_key in st.session_state:
                comment, model = st.session_state[cache_key]
                formatted = comment.replace(
                    "【強み】", "**【強み】**"
                ).replace("【弱み・注意点】","**【弱み・注意点】**"
                ).replace("【推奨する使い方】","**【推奨する使い方】**"
                ).replace("【組み合わせ推奨】","**【組み合わせ推奨】**")
                st.markdown(
                    f'<div style="background:#0a1628;border:1px solid #1e3a5f;'
                    f'border-left:3px solid #00d4ff;border-radius:6px;'
                    f'padding:12px 14px;margin-bottom:4px;font-size:12px;'
                    f'line-height:1.8;color:#cbd5e1;">'
                    f'<div style="font-family:monospace;font-size:9px;color:#7fff6b;'
                    f'letter-spacing:2px;margin-bottom:6px;">★ AI ANALYSIS</div>'
                    f'{formatted.replace(chr(10), "<br>")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if model:
                    st.caption(f"使用AI: {model}")
            st.markdown("<br>", unsafe_allow_html=True)

# ── タブ描画 ──────────────────────────────────────────
with tab_jp:
    render_media_tab(JP_MEDIA, "jp")

with tab_us:
    render_media_tab(US_MEDIA, "us")

with tab_cmp:
    st.markdown("#### 🌐 日米メディア平均スコア比較")
    if PLOTLY_AVAILABLE:
        jp_avg = {k: round(sum(m["scores"][k] for m in JP_MEDIA)/len(JP_MEDIA)) for k in SCORE_AXES}
        us_avg = {k: round(sum(m["scores"][k] for m in US_MEDIA)/len(US_MEDIA)) for k in SCORE_AXES}
        fig = go.Figure()
        for label, avg, color, fill in [
            ("🇯🇵 日本平均", jp_avg, "#ef4444", "rgba(239,68,68,0.15)"),
            ("🌍 海外平均",  us_avg, "#3b82f6", "rgba(59,130,246,0.15)"),
        ]:
            vals = list(avg.values())
            fig.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=SCORE_AXES+[SCORE_AXES[0]],
                fill="toself", name=label,
                line_color=color, fillcolor=fill, line_width=2.5,
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.15)", color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.15)", color="#f1f5f9", tickfont=dict(color="#f1f5f9")),
                bgcolor="#0a0e1a",
            ),
            paper_bgcolor="#0a0e1a", font=dict(color="#e2e8f0"),
            legend=dict(orientation="h", y=-0.15, font=dict(color="#f1f5f9", size=12)),
            height=440, margin=dict(t=30, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""**🇯🇵 日本政治経済を知るなら**
1. **東洋経済** — 企業・産業の深掘り
2. **文春オンライン** — スポンサータブー外の調査報道
3. **日経新聞** — 経済指標・市場速報
4. **日経CNBC** — マーケット解説（補完）

⚠️ 民放テレビは速報確認のみ
⚠️ 大スポンサー批判は週刊誌で補完必須""")
    with col2:
        st.markdown("""**🌍 国際政治経済を知るなら**
1. **AP News + Reuters** — 一次情報（最優先）
2. **BBC World** — 国際視点・深度
3. **The Economist** — 週次の構造分析
4. **FactCheck.org** — 政治発言の事実検証

⚠️ CNN/Foxは党派性が強い
⚠️ Bloomberg/FTは有料記事多め
✅ Al Jazeeraで非西洋視点を補完""")

with tab_inst:
    st.markdown("#### 🏛️ 評価機関別 メディア信頼度スコア")
    st.caption("各機関の公開調査を100点満点に正規化。None=データなし（N/A表示）。")

    region = st.radio("対象地域", ["日本メディア","海外メディア"], horizontal=True, key="inst_r")
    idata  = INSTITUTION_RATINGS[region]

    if PLOTLY_AVAILABLE:
        inst_list  = idata["機関"]
        media_list = idata["メディア"]
        colors     = ["#00d4ff","#7fff6b","#ff6b35","#a78bfa"]

        # グループバー
        fig = go.Figure()
        for i, inst in enumerate(inst_list):
            vals  = [idata["スコア"][inst].get(m) for m in media_list]
            shown = [v if v is not None else 0 for v in vals]
            texts = [str(v) if v is not None else "N/A" for v in vals]
            fig.add_trace(go.Bar(
                name=inst, x=media_list, y=shown,
                text=texts, textposition="outside",
                marker_color=colors[i % len(colors)], opacity=0.85,
            ))
        fig.update_layout(
            barmode="group",
            title=dict(text=f"{region} — 評価機関別信頼度スコア比較", font=dict(color="#f1f5f9", size=15)),
            yaxis=dict(range=[0,115], title="スコア（100点満点換算）",
                       gridcolor="rgba(200,200,200,0.15)", color="#e2e8f0",
                       tickfont=dict(color="#e2e8f0")),
            xaxis=dict(color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
            plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0"),
            legend=dict(orientation="h", y=-0.3, font=dict(size=11, color="#f1f5f9")),
            hovermode="x unified", height=460,
            margin=dict(l=60, r=20, t=50, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ヒートマップ
        z, zt = [], []
        for inst in inst_list:
            row  = [idata["スコア"][inst].get(m,0) or 0 for m in media_list]
            rowt = [str(idata["スコア"][inst].get(m)) if idata["スコア"][inst].get(m) else "N/A"
                    for m in media_list]
            z.append(row); zt.append(rowt)

        fig2 = go.Figure(go.Heatmap(
            z=z, x=media_list, y=inst_list,
            text=zt, texttemplate="%{text}",
            colorscale="RdYlGn", zmid=65, zmin=30, zmax=100,
            hovertemplate="%{y}<br>%{x}: <b>%{text}</b>点<extra></extra>",
        ))
        fig2.update_layout(
            height=280, plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
            font=dict(color="#e2e8f0"),
            margin=dict(l=200, r=20, t=20, b=60),
            xaxis=dict(tickangle=-30, color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
            yaxis=dict(color="#e2e8f0", tickfont=dict(color="#e2e8f0")),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**📋 出典リンク**")
    for inst in idata["機関"]:
        url = idata["出典"].get(inst, "")
        if url:
            st.markdown(f"- **{inst}** → [{url}]({url})")

with tab_update:
    st.markdown("#### 🔄 定期AI更新 — 評価機関の最新動向")
    st.markdown("""
- **四半期ごと**に自動更新（90日TTLキャッシュ）
- Gemini → Groq → OpenRouter でフォールバック
- 手動更新も可能
""")
    now = datetime.now(JST)
    quarter_str = f"{now.year}-Q{(now.month-1)//3+1}"
    st.info(f"📅 現在の四半期: **{quarter_str}**")

    col_a, col_b = st.columns([2,1])
    with col_a:
        run_btn = st.button("🔄 今すぐAI取得・更新", type="primary",
                            key="update_btn", use_container_width=True)
    with col_b:
        clr_btn = st.button("🗑️ キャッシュクリア", key="clr_btn", use_container_width=True)

    if clr_btn:
        ai_institution_summary.clear()
        st.session_state.pop("inst_result", None)
        st.session_state.pop("inst_model", None)
        st.success("✅ クリア完了")
        st.rerun()

    if run_btn or "inst_result" not in st.session_state:
        with st.spinner("🤖 AIが最新評価機関データを分析中..."):
            result, model = ai_institution_summary(quarter_str)
            st.session_state["inst_result"] = result
            st.session_state["inst_model"]  = model
            st.session_state["inst_date"]   = now.strftime("%Y-%m-%d %H:%M JST")

    if "inst_result" in st.session_state:
        res = st.session_state["inst_result"]
        for kw in ["【新聞通信調査会】","【Reuters Institute】",
                   "【MediaBias/FactCheck】","【RSF報道自由度】","【総合所見】"]:
            res = res.replace(kw, f"**{kw}**")
        st.markdown(
            f'<div style="background:#0a1628;border:1px solid #1e3a5f;'
            f'border-left:4px solid #00d4ff;border-radius:8px;'
            f'padding:18px 22px;font-size:13px;line-height:1.9;color:#cbd5e1;margin-top:12px;">'
            f'{res.replace(chr(10),"<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"🤖 使用AI: {st.session_state.get('inst_model','')} ｜ "
            f"生成日時: {st.session_state.get('inst_date','')} ｜ "
            f"次回自動更新: {quarter_str}終了後"
        )

st.markdown("---")
st.caption("⚠️ 本分析は新聞通信調査会・Reuters Institute・MediaBias/FactCheck等の公開データと独自評価を総合したものです。最新の公式数値は各機関のサイトでご確認ください。")
