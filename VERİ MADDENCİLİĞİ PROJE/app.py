# -*- coding: utf-8 -*-
"""
app.py – Öğrenci Başarısı Tahmin Sistemi | Ana Streamlit Uygulaması
CRISP-DM Metodolojisi · 5 ML Modeli · UCI Student Performance
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import veri_indir_ve_yukle, veri_isle
from utils.models import modelleri_egit, MODEL_KONFIGURASYONLARI

# ─────────────────────────────────────────────────────────────
# SAYFA AYARI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Öğrenci Başarısı Tahmin Sistemi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}

/* Başlık Gradient */
.hero-title {
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* Metrik Kartları */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.25);
}
.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f1f5f9;
}
.metric-label {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
}

/* Bölüm Başlıkları */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, #6366f1, transparent);
    margin-bottom: 1.5rem;
    border-radius: 2px;
}

/* Bilgi Kartı */
.info-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 0.6rem 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Model Kartı */
.model-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    position: relative;
    overflow: hidden;
}
.model-badge {
    display: inline-block;
    background: #6366f1;
    color: white;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Risk Seviye Badges */
.risk-high   { background:#ef444420; border:1px solid #ef4444; color:#ef4444; border-radius:8px; padding:0.2rem 0.6rem; font-weight:700; font-size:0.85rem; }
.risk-medium { background:#f59e0b20; border:1px solid #f59e0b; color:#f59e0b; border-radius:8px; padding:0.2rem 0.6rem; font-weight:700; font-size:0.85rem; }
.risk-low    { background:#22c55e20; border:1px solid #22c55e; color:#22c55e; border-radius:8px; padding:0.2rem 0.6rem; font-weight:700; font-size:0.85rem; }

/* Tahmin Sonuç Kartı */
.predict-success {
    background: linear-gradient(135deg, #14532d, #166534);
    border: 2px solid #22c55e;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.predict-fail {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 2px solid #ef4444;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.predict-score {
    font-size: 3.5rem;
    font-weight: 800;
}

/* CRISP-DM Adım Kartları */
.crisp-step {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    font-size: 0.88rem;
    color: #cbd5e1;
}
.crisp-step-num {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Sidebar nav butonu */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    color: #94a3b8;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-size: 0.92rem;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #6366f120;
    color: #f1f5f9;
}

/* Tablo */
.styled-table { border-collapse: collapse; width: 100%; }
.styled-table th { background: #6366f1; color: white; padding: 0.6rem 1rem; font-size: 0.85rem; }
.styled-table td { padding: 0.55rem 1rem; border-bottom: 1px solid #334155; color: #e2e8f0; font-size: 0.87rem; }
.styled-table tr:hover td { background: #6366f115; }

/* Plotly grafik arka planı */
.js-plotly-plot .plotly { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PLOTLY TEMA
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(15,23,42,0)",
    plot_bgcolor="rgba(30,41,59,0.5)",
    font=dict(family="Inter", color="#f1f5f9"),
    margin=dict(l=40, r=40, t=50, b=40)
)
RENKLER = ["#6366f1","#a855f7","#ec4899","#f59e0b","#22c55e","#06b6d4","#ef4444"]

# ─────────────────────────────────────────────────────────────
# VERİ YÜKLEMESİ
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def veri_yukle_ve_isle():
    df_ham = veri_indir_ve_yukle()
    return df_ham, veri_isle(df_ham)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🎓</div>
        <div style='font-weight:800; font-size:1.1rem; color:#f1f5f9; margin-top:0.3rem;'>
            Öğrenci Başarısı
        </div>
        <div style='font-size:0.78rem; color:#64748b;'>Tahmin &amp; Analiz Sistemi</div>
    </div>
    <hr style='border-color:#334155; margin: 0.5rem 0 1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; padding: 0 1rem; margin-bottom:0.5rem;'>NAVİGASYON</div>", unsafe_allow_html=True)

    sayfalar = {
        "🏠   Ana Sayfa": "ana",
        "📊  Veri Seti & EDA": "eda",
        "⚙️  Veri Ön İşleme": "onisleme",
        "🤖  Model Eğitimi": "modeller",
        "🎯  Tahmin Aracı": "tahmin",
    }

    if "sayfa" not in st.session_state:
        st.session_state["sayfa"] = "ana"

    for label, key in sayfalar.items():
        if st.button(label, key=f"nav_{key}"):
            st.session_state["sayfa"] = key

    sayfa = st.session_state["sayfa"]

    st.markdown("""
    <hr style='border-color:#334155; margin: 1.5rem 0 1rem 0;'>
    <div style='color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; padding: 0 1rem; margin-bottom:0.5rem;'>BİLGİ</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='padding: 0 0.5rem;'>
    <div class='info-card'>
    <b>📚 Kaynak:</b> UCI ML Repository<br>
    <b>📋 Yöntem:</b> CRISP-DM<br>
    <b>🔬 Algoritmalar:</b> DT · RF · LR · KNN · SVM
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='position:absolute; bottom:1rem; padding: 0 1rem; color:#475569; font-size:0.72rem;'>
    Veri Madenciliği Ders Projesi © 2026
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# VERİ YÜKLEMESİ (Spinner ile)
# ─────────────────────────────────────────────────────────────
with st.spinner("📡 UCI veri seti yükleniyor..."):
    df_ham, isle_sonuc = veri_yukle_ve_isle()
    X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test, ig_df, ozellikler, scaler, df_islenmis = isle_sonuc

# ══════════════════════════════════════════════════════════════
# SAYFA 1: ANA SAYFA
# ══════════════════════════════════════════════════════════════
if sayfa == "ana":
    # Hero
    st.markdown("""
    <div class='hero-title'>Öğrenci Başarısı<br>Tahmin Sistemi</div>
    <div class='hero-sub'>CRISP-DM Metodolojisi · UCI Student Performance · 5 Makine Öğrenmesi Modeli</div>
    """, unsafe_allow_html=True)

    # Metrik Kartları
    basarili = int(df_islenmis["hedef"].sum())
    basarisiz = int((df_islenmis["hedef"] == 0).sum())
    toplam = len(df_islenmis)
    basari_oran = basarili / toplam * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    kartlar = [
        (c1, "👨‍🎓", str(toplam), "Toplam Öğrenci"),
        (c2, "✅", str(basarili), "Başarılı"),
        (c3, "❌", str(basarisiz), "Başarısız"),
        (c4, "📈", f"%{basari_oran:.1f}", "Başarı Oranı"),
        (c5, "🧬", str(len(ozellikler)), "Özellik Sayısı"),
    ]
    for col, icon, val, label in kartlar:
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>{icon}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # CRISP-DM v2 Adımlar
    st.markdown("<div class='section-title'>🔍 CRISP-DM Metodolojisi</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    adimlar = [
        ("1", "İş\nAnlayışı", "Hedef tanımlama ve\nbaşarı kriteri belirleme"),
        ("2", "Veri\nAnlayışı", "EDA, korelasyon\nve dağılım analizi"),
        ("3", "Veri Ön\nİşleme", "Encoding, scaling\nve özellik seçimi"),
        ("4", "Modelleme", "5 algoritma +\nGridSearchCV"),
        ("5", "Değerlendirme", "Accuracy, F1,\nROC-AUC karşılaştırması"),
        ("6", "Dağıtım", "Eğitimci karar\ndestek aracı"),
    ]
    cols = st.columns(6)
    for col, (num, baslik, acik) in zip(cols, adimlar):
        col.markdown(f"""
        <div class='crisp-step'>
            <div class='crisp-step-num'>{num}</div>
            <b style='color:#f1f5f9;'>{baslik}</b><br>
            <span style='font-size:0.78rem; color:#64748b;'>{acik}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # G3 Dağılım Grafiği
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("<div class='section-title'>📊 Final Notu (G3) Dağılımı</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        g3_sayim = df_islenmis["G3"].value_counts().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=g3_sayim.index, y=g3_sayim.values,
            marker=dict(
                color=g3_sayim.index,
                colorscale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#22c55e"]],
                line=dict(color="#0f172a", width=1)
            ),
            text=g3_sayim.values, textposition="outside", textfont=dict(color="#94a3b8")
        ))
        fig.add_vline(x=9.5, line_dash="dash", line_color="#f59e0b", line_width=2,
                      annotation_text="Başarı Eşiği (10)", annotation_font_color="#f59e0b")
        fig.update_layout(**PLOTLY_LAYOUT, title="G3 Not Dağılımı",
                          xaxis_title="G3 Notu (0-20)", yaxis_title="Öğrenci Sayısı")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>🎯 Sınıf Dengesi</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=["Başarılı", "Başarısız"],
            values=[basarili, basarisiz],
            hole=0.6,
            marker=dict(colors=["#22c55e", "#ef4444"],
                        line=dict(color="#0f172a", width=3)),
            textinfo="label+percent",
            textfont=dict(size=13, color="white")
        ))
        fig2.update_layout(**PLOTLY_LAYOUT,
                           annotations=[dict(text=f"<b>{basari_oran:.0f}%</b><br>Başarılı",
                                            x=0.5, y=0.5, font=dict(size=16), showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    # Hızlı İçgörüler
    st.markdown("<div class='section-title'>💡 Hızlı İçgörüler</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    g1_col = "G1" if "G1" in df_islenmis.columns else ozellikler[0]
    g2_col = "G2" if "G2" in df_islenmis.columns else ozellikler[1]

    icegorular_data = []
    if "G1" in df_islenmis.columns:
        g1_corr = df_islenmis[["G1", "G3"]].corr().iloc[0, 1]
        icegorular_data.append(("📈", f"G1-G3 Korelasyonu: **{g1_corr:.3f}**", "Dönemler arası çok güçlü bağ"))
    if "G2" in df_islenmis.columns:
        g2_corr = df_islenmis[["G2", "G3"]].corr().iloc[0, 1]
        icegorular_data.append(("📈", f"G2-G3 Korelasyonu: **{g2_corr:.3f}**", "2. dönem de kritik gösterge"))
    if "calisma_suresi" in df_islenmis.columns:
        cs1 = df_islenmis[df_islenmis["calisma_suresi"] <= 1]["hedef"].mean() * 100
        cs4 = df_islenmis[df_islenmis["calisma_suresi"] >= 3]["hedef"].mean() * 100
        icegorular_data.append(("⏱️", f"Çalışma süresi <2 saat: **%{cs1:.0f}** başarı", f"5+ saat: %{cs4:.0f} başarı"))
    if "devamsizlik" in df_islenmis.columns:
        dev_az = df_islenmis[df_islenmis["devamsizlik"] <= 3]["hedef"].mean() * 100
        dev_cok = df_islenmis[df_islenmis["devamsizlik"] > 14]["hedef"].mean() * 100
        icegorular_data.append(("📅", f"15+ gün devamsız: **%{dev_cok:.0f}** başarı", f"0-3 gün: %{dev_az:.0f} başarı"))

    cols_ic = st.columns(len(icegorular_data))
    for col, (icon, baslik, acik) in zip(cols_ic, icegorular_data):
        col.markdown(f"""
        <div class='info-card'>
        <span style='font-size:1.5rem;'>{icon}</span><br>
        {baslik}<br>
        <span style='color:#64748b; font-size:0.82rem;'>{acik}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SAYFA 2: VERİ & EDA
# ══════════════════════════════════════════════════════════════
elif sayfa == "eda":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>📊 Veri Seti &amp; Keşifsel Veri Analizi</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>UCI Student Performance Dataset – Keşifsel İçgörüler</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Veri Önizleme", "📈 Dağılımlar", "🔥 Korelasyon", "🔍 Detaylı Analiz"])

    with tab1:
        st.markdown("### Veri Seti Yapısı")
        c1, c2, c3 = st.columns(3)
        c1.metric("Satır", df_ham.shape[0])
        c2.metric("Sütun", df_ham.shape[1])
        c3.metric("Eksik Değer", int(df_ham.isnull().sum().sum()))
        st.dataframe(df_ham.head(20), use_container_width=True, height=400)

        st.markdown("### Temel İstatistikler")
        sayisal = df_ham.select_dtypes(include="number")
        st.dataframe(sayisal.describe().round(2), use_container_width=True)

    with tab2:
        st.markdown("### Sayısal Değişken Dağılımları")
        sayisal_sutunlar = [c for c in ["G1","G2","G3","devamsizlik","calisma_suresi","yas"]
                            if c in df_islenmis.columns]

        secili = st.selectbox("Değişken Seç:", sayisal_sutunlar)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Dağılım (Başarı Durumuna Göre)", "Kutu Grafiği"])
        for hedef_val, renk, isim in [(1, "#22c55e", "Başarılı"), (0, "#ef4444", "Başarısız")]:
            veri = df_islenmis[df_islenmis["hedef"] == hedef_val][secili]
            fig.add_trace(go.Histogram(x=veri, name=isim, opacity=0.7,
                                       marker_color=renk, nbinsx=20), row=1, col=1)
            fig.add_trace(go.Box(y=veri, name=isim, marker_color=renk,
                                 boxmean=True, showlegend=False), row=1, col=2)
        fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", title=f"{secili} Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Korelasyon Isı Haritası")
        sayisal_df = df_islenmis.select_dtypes(include="number")
        corr = sayisal_df.corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
            text=corr.values, texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="r")
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="Değişkenler Arası Korelasyon",
                          height=600, xaxis=dict(tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)

        if "G1" in corr.columns and "G3" in corr.columns:
            st.info(f"**G1 ↔ G3 korelasyonu: {corr.loc['G1','G3']:.3f}** – Çok güçlü pozitif ilişki ✅")
        if "G2" in corr.columns and "G3" in corr.columns:
            st.info(f"**G2 ↔ G3 korelasyonu: {corr.loc['G2','G3']:.3f}** – Çok güçlü pozitif ilişki ✅")

    with tab4:
        st.markdown("### Çalışma Süresi & Devamsızlık Analizi")
        col1, col2 = st.columns(2)

        with col1:
            if "calisma_suresi" in df_islenmis.columns:
                etiketler = {1: "<2 Saat", 2: "2-5 Saat", 3: "5-10 Saat", 4: ">10 Saat"}
                cs_b = df_islenmis.groupby("calisma_suresi")["hedef"].mean() * 100
                fig = go.Figure(go.Bar(
                    x=[etiketler.get(i, str(i)) for i in cs_b.index],
                    y=cs_b.values,
                    marker=dict(color=RENKLER[:len(cs_b)], line=dict(color="#0f172a", width=1)),
                    text=[f"%{v:.0f}" for v in cs_b.values], textposition="outside"
                ))
                fig.add_hline(y=75, line_dash="dash", line_color="#f59e0b",
                              annotation_text="%75 Eşiği", annotation_font_color="#f59e0b")
                fig.update_layout(**PLOTLY_LAYOUT, title="Çalışma Süresi → Başarı Oranı",
                                  yaxis=dict(range=[0, 110]))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "devamsizlik" in df_islenmis.columns:
                fig = go.Figure()
                for hv, renk, isim in [(1,"#22c55e","Başarılı"), (0,"#ef4444","Başarısız")]:
                    fig.add_trace(go.Box(
                        y=df_islenmis[df_islenmis["hedef"]==hv]["devamsizlik"],
                        name=isim, marker_color=renk, boxmean=True
                    ))
                fig.update_layout(**PLOTLY_LAYOUT, title="Devamsızlık → Başarı Grubu")
                st.plotly_chart(fig, use_container_width=True)

        # Scatter G1 vs G3
        if "G1" in df_islenmis.columns and "G3" in df_islenmis.columns:
            st.markdown("### G1 → G3 İlişkisi")
            fig = px.scatter(
                df_islenmis, x="G1", y="G3",
                color=df_islenmis["hedef"].map({1:"Başarılı", 0:"Başarısız"}),
                color_discrete_map={"Başarılı":"#22c55e","Başarısız":"#ef4444"},
                trendline="ols",
                labels={"G1":"1. Dönem Notu (G1)","G3":"Final Notu (G3)","color":"Durum"},
                title="G1 & G3 Dağılımı + Regresyon Çizgisi"
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# SAYFA 3: VERİ ÖN İŞLEME
# ══════════════════════════════════════════════════════════════
elif sayfa == "onisleme":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>⚙️ Veri Ön İşleme</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Eksik Veri · Encoding · Standardizasyon · Özellik Seçimi</div>", unsafe_allow_html=True)

    # Adımlar
    adimlar_on = ["Eksik Veri\nKontrolü", "Label\nEncoding", "Train/Test\nAyrımı", "Standard\nScaler", "Information\nGain"]
    ar_cols = st.columns(len(adimlar_on))
    for col, (i, ad) in zip(ar_cols, enumerate(adimlar_on, 1)):
        col.markdown(f"""
        <div class='crisp-step'>
            <div class='crisp-step-num'>{i}</div>
            <b style='color:#f1f5f9; font-size:0.85rem;'>{ad}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>🔍 Eksik Veri Analizi</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        eksik = df_ham.isnull().sum()
        if eksik.sum() == 0:
            st.success(f"✅ Veri setinde **hiç eksik değer yok!** ({len(df_ham)} satır × {df_ham.shape[1]} sütun)")
        else:
            st.warning(f"⚠️ Toplam {eksik.sum()} eksik değer bulundu. Strateji: sayısal→ortalama, kategorik→mod.")
            st.dataframe(eksik[eksik > 0].rename("Eksik Sayısı"), use_container_width=True)

        st.markdown("### 📊 Eğitim/Test Dağılımı")
        fig = go.Figure(go.Pie(
            labels=["Eğitim", "Test"],
            values=[len(X_train), len(X_test)],
            hole=0.55,
            marker=dict(colors=["#6366f1","#a855f7"], line=dict(color="#0f172a", width=3)),
            textinfo="label+value+percent"
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=280,
                          annotations=[dict(text=f"<b>{len(df_islenmis)}</b><br>Toplam",
                                            x=0.5, y=0.5, showarrow=False, font=dict(size=14))])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>🔢 Encoding &amp; Standardizasyon</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        kat_sutunlar = df_ham.select_dtypes(include="object").columns.tolist()
        enc_rows = []
        for s in kat_sutunlar[:8]:
            uniq = sorted(df_ham[s].dropna().unique())[:4]
            enc_rows.append({"Sütun": s, "Örnek Değerler": str(uniq)[1:-1], "Yöntem": "Label Encoding"})
        st.dataframe(pd.DataFrame(enc_rows), use_container_width=True, hide_index=True)

        st.markdown("### 🔐 Standardizasyon (StandardScaler)")
        st.markdown("""
        <div class='info-card'>
        <b>Formül:</b> z = (x − μ) / σ<br><br>
        Uygulandığı modeller: <b>Lojistik Regresyon</b>, <b>K-NN</b>, <b>SVM</b><br>
        Uygulanmadığı modeller: <b>Karar Ağacı</b>, <b>Rastgele Orman</b>
        (ağaç tabanlı modeller ölçekleme gerektirmez)
        </div>
        """, unsafe_allow_html=True)

    # Information Gain
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🧬 Özellik Önemi – Information Gain</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    ig_plot = ig_df.copy()
    ig_plot["Renk"] = ["#22c55e" if i < 5 else "#6366f1" if i < 10 else "#475569"
                       for i in range(len(ig_plot))]
    ig_plot["Etiket"] = ["⭐ Kritik" if i < 3 else "🔵 Yüksek" if i < 7 else "⚪ Düşük"
                         for i in range(len(ig_plot))]

    fig = go.Figure(go.Bar(
        x=ig_plot["IG"], y=ig_plot["Ozellik"],
        orientation="h",
        marker=dict(color=ig_plot["Renk"], line=dict(color="#0f172a", width=1)),
        text=[f"{v:.3f}" for v in ig_plot["IG"]],
        textposition="outside", textfont=dict(color="#94a3b8")
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Information Gain (Mutual Information) Sıralaması",
                      height=max(400, len(ig_plot) * 28),
                      yaxis=dict(autorange="reversed"),
                      xaxis_title="Information Gain Skoru")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.markdown(f"""
    <div class='info-card'>
    🏆 <b>En Önemli Özellik:</b> {ig_df.iloc[0]['Ozellik']} (IG = {ig_df.iloc[0]['IG']:.4f})<br>
    2. <b>İkinci:</b> {ig_df.iloc[1]['Ozellik']} (IG = {ig_df.iloc[1]['IG']:.4f})<br>
    3. <b>Üçüncü:</b> {ig_df.iloc[2]['Ozellik']} (IG = {ig_df.iloc[2]['IG']:.4f})
    </div>
    """, unsafe_allow_html=True)
    col2.markdown(f"""
    <div class='info-card'>
    📊 <b>Toplam Özellik:</b> {len(ozellikler)}<br>
    ✅ <b>Eğitim Seti:</b> {len(X_train)} örnek<br>
    🧪 <b>Test Seti:</b> {len(X_test)} örnek
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SAYFA 4: MODEL EĞİTİMİ
# ══════════════════════════════════════════════════════════════
elif sayfa == "modeller":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>🤖 Model Eğitimi &amp; Değerlendirme</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>5 Algoritma · GridSearchCV · Çapraz Doğrulama · ROC Eğrileri</div>", unsafe_allow_html=True)

    with st.spinner("🔄 Modeller eğitiliyor (GridSearchCV + 5-Fold CV)... Bu ~60 saniye sürebilir."):
        sonuclar = modelleri_egit(X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test)

    st.success(f"✅ **{len(sonuclar)} model** başarıyla eğitildi!")

    # En İyi Model
    en_iyi = max(sonuclar, key=lambda k: sonuclar[k]["f1"])
    s_iyi = sonuclar[en_iyi]

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e1b4b, #312e81); border: 2px solid #6366f1;
                border-radius: 16px; padding: 1.5rem 2rem; margin: 1rem 0;'>
        <div style='font-size:0.8rem; color:#818cf8; text-transform:uppercase; letter-spacing:0.1em;'>🏆 EN İYİ MODEL</div>
        <div style='font-size:1.8rem; font-weight:800; color:#f1f5f9; margin: 0.3rem 0;'>
            {s_iyi['icon']} {en_iyi}
        </div>
        <div style='display:flex; gap:2rem; margin-top:0.5rem;'>
            <div><span style='color:#64748b; font-size:0.8rem;'>F1-SKORU</span><br>
                 <b style='color:#22c55e; font-size:1.3rem;'>{s_iyi['f1']:.4f}</b></div>
            <div><span style='color:#64748b; font-size:0.8rem;'>ROC-AUC</span><br>
                 <b style='color:#6366f1; font-size:1.3rem;'>{s_iyi['auc']:.4f}</b></div>
            <div><span style='color:#64748b; font-size:0.8rem;'>DOĞRULUK</span><br>
                 <b style='color:#a855f7; font-size:1.3rem;'>{s_iyi['accuracy']:.4f}</b></div>
            <div><span style='color:#64748b; font-size:0.8rem;'>CV ORTALAMA</span><br>
                 <b style='color:#f59e0b; font-size:1.3rem;'>{s_iyi['cv_mean']:.4f}</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Karşılaştırma", "📈 ROC Eğrileri", "🧩 Confusion Matrix", "⚙️ Hiperparametreler"])

    with tab1:
        # Tablo
        tablo_rows = []
        for madi, s in sonuclar.items():
            tablo_rows.append({
                "Model": f"{s['icon']} {madi}",
                "Doğruluk": f"{s['accuracy']:.4f}",
                "Kesinlik": f"{s['precision']:.4f}",
                "Duyarlılık": f"{s['recall']:.4f}",
                "F1-Skoru": f"{s['f1']:.4f}",
                "ROC-AUC": f"{s['auc']:.4f}",
                "CV (±std)": f"{s['cv_mean']:.4f} ±{s['cv_std']:.4f}"
            })
        tablo_df = pd.DataFrame(tablo_rows)
        st.dataframe(tablo_df, use_container_width=True, hide_index=True)

        # Grouped bar
        modeller_list = list(sonuclar.keys())
        metrikler_list = ["accuracy", "precision", "recall", "f1", "auc"]
        metrik_etiket = ["Doğruluk", "Kesinlik", "Duyarlılık", "F1", "AUC"]

        fig = go.Figure()
        for i, (madi, s) in enumerate(sonuclar.items()):
            vals = [s[m] for m in metrikler_list]
            fig.add_trace(go.Bar(
                name=f"{s['icon']} {madi}",
                x=metrik_etiket, y=vals,
                marker=dict(color=RENKLER[i], opacity=0.85, line=dict(color="#0f172a", width=1)),
                text=[f"{v:.3f}" for v in vals], textposition="outside"
            ))
        fig.add_hline(y=0.8, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                      annotation_text="%80 hedef", annotation_font_color="#f59e0b")
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                          title="Model Karşılaştırması – Tüm Metrikler",
                          yaxis=dict(range=[0, 1.12]))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        for i, (madi, s) in enumerate(sonuclar.items()):
            fig.add_trace(go.Scatter(
                x=s["fpr"], y=s["tpr"], mode="lines",
                name=f"{s['icon']} {madi} (AUC={s['auc']:.3f})",
                line=dict(color=RENKLER[i], width=2.5)
            ))
        fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Rastgele (AUC=0.5)",
            line=dict(color="#475569", width=1.5, dash="dash")
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title="ROC Eğrileri – Tüm Modeller",
                          xaxis_title="Yanlış Pozitif Oranı (FPR)",
                          yaxis_title="Doğru Pozitif Oranı (TPR)",
                          height=520)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cm_cols = st.columns(5)
        for col, (madi, s) in zip(cm_cols, sonuclar.items()):
            cm = s["cm"]
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Başarısız","Başarılı"], y=["Başarısız","Başarılı"],
                colorscale="Blues",
                text=cm, texttemplate="%{text}",
                textfont=dict(size=18, color="white"),
                showscale=False
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280,
                              title=dict(text=f"{s['icon']} {madi}<br><sup>Acc: {s['accuracy']:.2f}</sup>",
                                         font=dict(size=11)),
                              xaxis_title="Tahmin", yaxis_title="Gerçek")
            col.plotly_chart(fig, use_container_width=True)

    with tab4:
        for madi, s in sonuclar.items():
            cfg = MODEL_KONFIGURASYONLARI[madi]
            with st.expander(f"{s['icon']} {madi} – Hiperparametreler", expanded=False):
                col1, col2 = st.columns(2)
                col1.markdown("**Arama Uzayı:**")
                col1.json(cfg["params"])
                col2.markdown("**En İyi Parametreler:**")
                col2.json(s["en_iyi_params"])
                col2.markdown(f"""
                <div class='info-card'>
                F1: <b>{s['f1']:.4f}</b> | AUC: <b>{s['auc']:.4f}</b><br>
                CV: <b>{s['cv_mean']:.4f}</b> ±{s['cv_std']:.4f}
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SAYFA 5: TAHMİN ARACI
# ══════════════════════════════════════════════════════════════
elif sayfa == "tahmin":
    st.markdown("<div class='hero-title' style='font-size:2rem;'>🎯 Öğrenci Başarı Tahmin Aracı</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Öğrenci profilini girerek tüm modellerin tahminini anlık görün</div>", unsafe_allow_html=True)

    with st.spinner("🔄 Modeller yükleniyor..."):
        sonuclar = modelleri_egit(X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test)

    st.markdown("---")

    # INPUT FORMU
    with st.form("tahmin_formu"):
        st.markdown("### 📝 Öğrenci Profili")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Demografik Bilgiler**")
            yas = st.slider("Yaş", 15, 22, 17, key="yas", help="Öğrencinin yaşı (15-22 arası)")
            cinsiyet_sec = st.selectbox("Cinsiyet", ["M (Erkek)", "F (Kız)"], key="cins", help="Öğrencinin cinsiyeti")
            adres_sec = st.selectbox("Yerleşim Yeri", ["U (Kent)", "R (Kırsal)"], key="adres", help="U: Şehir merkezi, R: Kırsal bölge")

        with col2:
            st.markdown("**📚 Akademik Bilgiler**")
            st.info("💡 **Not:** Notlar Portekiz sistemine göre 20 üzerindendir. (x5 = 100'lük sistem)")
            g1 = st.slider("G1 (1. Dönem Notu)", 0, 20, 10, key="g1", 
                           help="0-20 arası. Örneğin: 14 notu 70 puana denk gelir.")
            g2 = st.slider("G2 (2. Dönem Notu)", 0, 20, 10, key="g2", 
                           help="0-20 arası. Örneğin: 10 notu 50 puana (geçer not) denk gelir.")
            calisma = st.slider("Çalışma Süresi", 1, 4, 2, key="calisma",
                                help="1: <2 saat, 2: 2-5 saat, 3: 5-10 saat, 4: >10 saat")
            devamsizlik = st.slider("Devamsızlık (gün)", 0, 93, 5, key="dev", help="Toplam devamsızlık gün sayısı")
            sinif_tekrar = st.slider("Sınıf Tekrarı Sayısı", 0, 3, 0, key="tekrar", help="Daha önce başarısız olunan sınıf sayısı")

        with col3:
            st.markdown("**🏠 Sosyal & Aile Bilgileri**")
            anne_egitim_v = st.slider("Anne Eğitim Seviyesi", 0, 4, 2, key="anne", 
                                     help="0: Yok, 1: İlkokul, 2: Ortaokul, 3: Lise, 4: Yükseköğretim")
            baba_egitim_v = st.slider("Baba Eğitim Seviyesi", 0, 4, 2, key="baba",
                                     help="0: Yok, 1: İlkokul, 2: Ortaokul, 3: Lise, 4: Yükseköğretim")
            hafta_ici_alkolv = st.slider("Hafta İçi Alkol Tüketimi", 0, 5, 0, key="hia",
                                        help="0: Hiç, 1: Çok Az, ..., 5: Çok Fazla")
            hafta_sonu_alkolv = st.slider("Hafta Sonu Alkol Tüketimi", 0, 5, 0, key="hsa",
                                         help="0: Hiç, 1: Çok Az, ..., 5: Çok Fazla")
            saglik_v = st.slider("Sağlık Durumu", 1, 5, 3, key="saglik", help="1: Çok Kötü, ..., 5: Çok İyi")

        tahmin_butonu = st.form_submit_button("🔮 Tahmin Et", use_container_width=True,
                                               type="primary")

    if tahmin_butonu:
        # Feature vektörü oluştur – eğitim sırasındaki ozellikler listesini kullan
        ornek = {f: 0 for f in ozellikler}

        hia_val = max(1, hafta_ici_alkolv)
        hsa_val = max(1, hafta_sonu_alkolv)

        # Bilinen değerleri doldur
        esleme = {
            "yas": yas, "G1": g1, "G2": g2,
            "calisma_suresi": calisma, "devamsizlik": devamsizlik,
            "sinif_tekrar": sinif_tekrar,
            "anne_egitim": anne_egitim_v, "baba_egitim": baba_egitim_v,
            "hafta_ici_alkol": hia_val, "hafta_sonu_alkol": hsa_val,
            "saglik": saglik_v,
            # Binary alanlar (encoded)
            "cinsiyet": 0 if cinsiyet_sec.startswith("M") else 1,
            "adres": 0 if adres_sec.startswith("U") else 1,
            # Feature Engineering özellikleri
            "not_ortalama": (g1 + g2) / 2.0,
            "akademik_trend": g2 - g1,
            "g1_g2_carpim": g1 * g2,
            "not_varyans": ((g1 - (g1+g2)/2)**2 + (g2 - (g1+g2)/2)**2) / 2,
            "min_not": min(g1, g2),
            "dusuk_not_var": 1 if (g1 < 10 or g2 < 10) else 0,
            "ebeveyn_egitim_toplam": anne_egitim_v + baba_egitim_v,
            "ebeveyn_egitim_max": max(anne_egitim_v, baba_egitim_v),
            "toplam_alkol": hia_val + hsa_val,
            "sosyal_skor": 3 + 3,  # varsayılan serbest_zaman + sosyallestirme
            "calisma_devamsizlik_oran": calisma / (devamsizlik + 1),
            "risk_skoru": (sinif_tekrar * 2
                          + (1.5 if devamsizlik > 10 else 0)
                          + (1 if hsa_val >= 4 else 0)
                          + (1 if calisma <= 1 else 0)),
            "devamsizlik_kategori": 0 if devamsizlik <= 3 else (1 if devamsizlik <= 7 else (2 if devamsizlik <= 14 else 3)),
        }
        for k, v in esleme.items():
            if k in ornek:
                ornek[k] = v

        X_ornek = pd.DataFrame([ornek])
        X_ornek_scaled = pd.DataFrame(scaler.transform(X_ornek), columns=ozellikler)

        st.markdown("---")
        st.markdown("### 🎯 Model Tahmin Sonuçları")

        # Tahminleri topla
        tahmin_sonuclari = {}
        for madi, s in sonuclar.items():
            cfg = MODEL_KONFIGURASYONLARI[madi]
            X_in = X_ornek_scaled if cfg["scaled"] else X_ornek
            prob = s["model"].predict_proba(X_in)[0][1]
            pred = int(prob >= 0.5)
            tahmin_sonuclari[madi] = {"prob": prob, "pred": pred}

        # Oylamayla genel karar
        oylar = [v["pred"] for v in tahmin_sonuclari.values()]
        genel_pred = 1 if sum(oylar) >= 3 else 0
        ortalama_prob = np.mean([v["prob"] for v in tahmin_sonuclari.values()])

        # Büyük sonuç kartı
        if genel_pred == 1:
            st.markdown(f"""
            <div class='predict-success'>
                <div class='predict-score'>✅</div>
                <div style='font-size:2rem; font-weight:800; color:#86efac;'>BAŞARILI</div>
                <div style='color:#22c55e; font-size:1.1rem; margin:0.5rem 0;'>
                    Ortalama Başarı Olasılığı: <b>%{ortalama_prob*100:.1f}</b>
                </div>
                <div style='color:#bbf7d0; font-size:0.9rem;'>Model oylaması: {sum(oylar)}/5 başarılı tahmini</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='predict-fail'>
                <div class='predict-score'>⚠️</div>
                <div style='font-size:2rem; font-weight:800; color:#fca5a5;'>BAŞARISIZLIK RİSKİ</div>
                <div style='color:#ef4444; font-size:1.1rem; margin:0.5rem 0;'>
                    Ortalama Başarı Olasılığı: <b>%{ortalama_prob*100:.1f}</b>
                </div>
                <div style='color:#fecaca; font-size:0.9rem;'>Model oylaması: {sum(oylar)}/5 başarılı tahmini</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Her model için oran çubuğu
        cols_tahmin = st.columns(5)
        for col, (madi, sonuc) in zip(cols_tahmin, tahmin_sonuclari.items()):
            s = sonuclar[madi]
            prob_pct = sonuc["prob"] * 100
            renk = "#22c55e" if sonuc["pred"] == 1 else "#ef4444"
            dur = "Başarılı" if sonuc["pred"] == 1 else "Riskli"
            col.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:1.5rem;'>{s['icon']}</div>
                <div style='font-size:0.78rem; color:#64748b; margin-bottom:0.3rem;'>{madi}</div>
                <div style='font-size:1.8rem; font-weight:800; color:{renk};'>%{prob_pct:.0f}</div>
                <div style='font-size:0.8rem; color:{renk};'>{dur}</div>
                <div style='background:#0f172a; border-radius:4px; margin-top:0.5rem; height:6px;'>
                    <div style='background:{renk}; width:{prob_pct:.0f}%; height:100%; border-radius:4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk Faktörü Analizi
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 💡 Risk Faktörü Analizi")

        riskler = []
        if g1 < 10:
            riskler.append(("🔴 Yüksek Risk", f"G1 notu ({g1}) kritik eşiğin altında. G1 < 10 olanların yalnızca %13'ü başarılı."))
        if calisma <= 1:
            riskler.append(("🔴 Yüksek Risk", "Haftalık çalışma süresi < 2 saat. Bu grupta başarı oranı %52."))
        if devamsizlik > 14:
            riskler.append(("🟡 Orta Risk", f"Devamsızlık ({devamsizlik} gün) yüksek. Bu grupta başarı oranı ~%64."))
        if hafta_sonu_alkolv >= 4:
            riskler.append(("🟡 Orta Risk", "Hafta sonu alkol tüketimi yüksek. Akademik performansla negatif korelasyon."))
        if g1 >= 14 and calisma >= 3:
            riskler.append(("🟢 Düşük Risk", "G1 notu ve çalışma süresi yeterli. Başarı bekleniyor."))

        if not riskler:
            st.info("ℹ️ Girilen profil için belirgin bir risk faktörü tespit edilmedi.")
        else:
            for seviye, mesaj in riskler:
                renk_cls = "risk-high" if "Yüksek" in seviye else "risk-medium" if "Orta" in seviye else "risk-low"
                st.markdown(f"""
                <div class='info-card' style='margin:0.4rem 0;'>
                <span class='{renk_cls}'>{seviye}</span>&nbsp;&nbsp;{mesaj}
                </div>
                """, unsafe_allow_html=True)
