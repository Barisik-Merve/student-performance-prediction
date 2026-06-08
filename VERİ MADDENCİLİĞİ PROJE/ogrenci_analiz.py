# -*- coding: utf-8 -*-
# =============================================================================
# OGRENCİ BAŞARISININ TAHMİNİ VE ANALİZİ ICIN VERİ MADDENCİLİGİ
# CRISP-DM Metodolojisi ile Tam Uygulama
# =============================================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

warnings.filterwarnings('ignore')

# Grafik kayıt klasörü
GRAFIK_KLASORU = "grafikler"
os.makedirs(GRAFIK_KLASORU, exist_ok=True)

# Renk paleti
RENKLER = {
    'birincil': '#2E86AB',
    'ikincil': '#A23B72',
    'basarili': '#22c55e',
    'basarisiz': '#ef4444',
    'arkaplan': '#f8fafc',
    'metin': '#1e293b',
    'vurgu': '#f59e0b'
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.facecolor': RENKLER['arkaplan'],
    'axes.facecolor': RENKLER['arkaplan'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

np.random.seed(42)

# =============================================================================
# 1. SENTETİK VERİ SETİ OLUŞTURMA
# =============================================================================

def veri_seti_olustur(n=395):
    """
    UCI Student Performance veri setine dayalı gerçekçi sentetik veri oluşturur.
    Tüm değişkenler gerçek akademik araştırmadaki dağılımları yansıtır.
    """
    print("\n" + "="*70)
    print("  VERİ SETİ OLUŞTURULUYOR")
    print("="*70)

    # Demografik değişkenler
    yas = np.random.randint(15, 23, n)
    cinsiyet = np.random.choice(['K', 'E'], n, p=[0.52, 0.48])
    adres = np.random.choice(['Kentsel', 'Kırsal'], n, p=[0.77, 0.23])

    # Aile değişkenleri
    anne_egitim = np.random.choice([0, 1, 2, 3, 4], n, p=[0.17, 0.20, 0.31, 0.23, 0.09])
    baba_egitim = np.random.choice([0, 1, 2, 3, 4], n, p=[0.22, 0.23, 0.28, 0.20, 0.07])
    aile_buyuklugu = np.random.choice(['<=3', '>3'], n, p=[0.33, 0.67])
    ebeveyn_durum = np.random.choice(['beraberler', 'ayri'], n, p=[0.89, 0.11])

    # Akademik değişkenler
    calisma_suresi = np.random.choice([1, 2, 3, 4], n, p=[0.24, 0.40, 0.23, 0.13])
    devamsizlik = np.random.randint(0, 28, n)
    sinif_tekrar = np.random.choice([0, 1, 2, 3], n, p=[0.82, 0.10, 0.05, 0.03])
    dershane_destegi = np.random.choice([0, 1], n, p=[0.71, 0.29])
    aile_egitim_destegi = np.random.choice([0, 1], n, p=[0.40, 0.60])
    ekstra_aktiviteler = np.random.choice([0, 1], n, p=[0.47, 0.53])
    internet = np.random.choice([0, 1], n, p=[0.34, 0.66])

    # Sosyal değişkenler
    hafta_ici_alkol = np.random.choice([1, 2, 3, 4, 5], n, p=[0.59, 0.20, 0.10, 0.07, 0.04])
    hafta_sonu_alkol = np.random.choice([1, 2, 3, 4, 5], n, p=[0.30, 0.24, 0.20, 0.14, 0.12])
    serbest_zaman = np.random.choice([1, 2, 3, 4, 5], n, p=[0.07, 0.18, 0.33, 0.29, 0.13])
    sosyallestirme = np.random.choice([1, 2, 3, 4, 5], n, p=[0.11, 0.22, 0.32, 0.24, 0.11])
    saglik = np.random.choice([1, 2, 3, 4, 5], n, p=[0.09, 0.12, 0.22, 0.26, 0.31])

    # G1 ve G2 notlarını gerçekçi faktörlerle hesapla
    temel_not = (
        calisma_suresi * 1.8 +
        (anne_egitim + baba_egitim) * 0.4 +
        (aile_buyuklugu == '<=3').astype(int) * 0.5 +
        dershane_destegi * 1.2 +
        aile_egitim_destegi * 0.8 +
        internet * 0.7 +
        saglik * 0.5 -
        devamsizlik * 0.18 -
        sinif_tekrar * 1.5 -
        hafta_sonu_alkol * 0.6 -
        (yas - 15) * 0.1 +
        np.random.normal(8, 1.5, n)
    )

    # Cinsiyet etkisi (kızlar hafif daha yüksek)
    cinsiyet_etkisi = np.where(cinsiyet == 'K', 0.4, 0)
    temel_not += cinsiyet_etkisi

    # G1: 0-20 arasına kırp
    G1 = np.clip(np.round(temel_not + np.random.normal(0, 1.2, n)), 0, 20).astype(int)
    # G2: G1'e bağımlı
    G2 = np.clip(np.round(G1 * 0.85 + np.random.normal(1.5, 1.0, n)), 0, 20).astype(int)
    # G3: G1 ve G2'ye bağımlı
    G3 = np.clip(np.round(G1 * 0.55 + G2 * 0.35 + np.random.normal(0.8, 0.9, n)), 0, 20).astype(int)

    df = pd.DataFrame({
        'yas': yas,
        'cinsiyet': cinsiyet,
        'adres': adres,
        'aile_buyuklugu': aile_buyuklugu,
        'ebeveyn_durum': ebeveyn_durum,
        'anne_egitim': anne_egitim,
        'baba_egitim': baba_egitim,
        'calisma_suresi': calisma_suresi,
        'sinif_tekrar': sinif_tekrar,
        'dershane_destegi': dershane_destegi,
        'aile_egitim_destegi': aile_egitim_destegi,
        'ekstra_aktiviteler': ekstra_aktiviteler,
        'internet': internet,
        'hafta_ici_alkol': hafta_ici_alkol,
        'hafta_sonu_alkol': hafta_sonu_alkol,
        'serbest_zaman': serbest_zaman,
        'sosyallestirme': sosyallestirme,
        'saglik': saglik,
        'devamsizlik': devamsizlik,
        'G1': G1,
        'G2': G2,
        'G3': G3
    })

    # Hedef değişken: G3 >= 10 ise Başarılı
    df['hedef'] = (df['G3'] >= 10).astype(int)

    print(f"  ✓ Toplam kayıt: {len(df)}")
    print(f"  ✓ Özellik sayısı: {df.shape[1] - 1}")
    print(f"  ✓ Başarılı öğrenci: {df['hedef'].sum()} ({df['hedef'].mean()*100:.1f}%)")
    print(f"  ✓ Başarısız öğrenci: {(~df['hedef'].astype(bool)).sum()} ({(1-df['hedef'].mean())*100:.1f}%)")

    return df


# =============================================================================
# 2. KEŞİFSEL VERİ ANALİZİ (EDA)
# =============================================================================

def veri_kesifsel_analiz(df):
    print("\n" + "="*70)
    print("  KEŞİFSEL VERİ ANALİZİ (EDA)")
    print("="*70)

    # ── 2.1 Temel İstatistikler ──────────────────────────────────────────────
    print("\n[2.1] Temel İstatistikler:")
    sayisal_sutunlar = ['yas', 'calisma_suresi', 'devamsizlik', 'G1', 'G2', 'G3']
    istatistikler = df[sayisal_sutunlar].describe().round(2)
    print(istatistikler.to_string())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Sayısal Değişkenlerin Dağılımları", fontsize=16, fontweight='bold', y=1.02)

    for idx, sutun in enumerate(sayisal_sutunlar):
        ax = axes[idx // 3][idx % 3]
        veriler_basarili = df[df['hedef'] == 1][sutun]
        veriler_basarisiz = df[df['hedef'] == 0][sutun]

        ax.hist(veriler_basarili, alpha=0.7, color=RENKLER['basarili'], label='Başarılı', bins=15, edgecolor='white')
        ax.hist(veriler_basarisiz, alpha=0.7, color=RENKLER['basarisiz'], label='Başarısız', bins=15, edgecolor='white')
        ax.set_title(f'{sutun.upper()} Dağılımı')
        ax.set_xlabel(sutun)
        ax.set_ylabel('Frekans')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/01_dagilimlar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/01_dagilimlar.png")

    # ── 2.2 Korelasyon Isı Haritası ─────────────────────────────────────────
    print("\n[2.2] Korelasyon Analizi:")
    sayisal_df = df[sayisal_sutunlar + ['hedef']]
    korelasyon = sayisal_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(korelasyon, dtype=bool))
    sns.heatmap(korelasyon, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8}, annot_kws={'size': 11, 'weight': 'bold'})
    ax.set_title("Değişkenler Arası Korelasyon Matrisi", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/02_korelasyon.png", dpi=150, bbox_inches='tight')
    plt.close()

    g1_g3 = korelasyon.loc['G1', 'G3']
    g2_g3 = korelasyon.loc['G2', 'G3']
    print(f"  ✓ G1-G3 korelasyonu: {g1_g3:.3f}")
    print(f"  ✓ G2-G3 korelasyonu: {g2_g3:.3f}")
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/02_korelasyon.png")

    # ── 2.3 G1-G2-G3 Scatter Matrisi ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dönem Notları Arasındaki İlişki", fontsize=14, fontweight='bold')

    renkler = [RENKLER['basarisiz'] if h == 0 else RENKLER['basarili'] for h in df['hedef']]
    for ax, (x_col, y_col) in zip(axes, [('G1', 'G3'), ('G2', 'G3')]):
        ax.scatter(df[x_col], df[y_col], c=renkler, alpha=0.6, edgecolors='white', linewidth=0.5, s=50)
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        ax.plot(x_line, p(x_line), color=RENKLER['vurgu'], lw=2.5, linestyle='--', label=f'Eğilim')
        corr_val = df[[x_col, 'G3']].corr().iloc[0, 1]
        ax.set_xlabel(f'{x_col} Notu', fontsize=12)
        ax.set_ylabel('G3 (Final Notu)', fontsize=12)
        ax.set_title(f'{x_col} → G3  (r = {corr_val:.2f})')

        basarili_yama = mpatches.Patch(color=RENKLER['basarili'], label='Başarılı')
        basarisiz_yama = mpatches.Patch(color=RENKLER['basarisiz'], label='Başarısız')
        ax.legend(handles=[basarili_yama, basarisiz_yama])

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/03_not_iliskisi.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/03_not_iliskisi.png")

    # ── 2.4 Çalışma Süresi Analizi ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Çalışma Süresi ve Başarı İlişkisi", fontsize=14, fontweight='bold')

    calisma_etiketleri = {1: '<2 Saat', 2: '2-5 Saat', 3: '5-10 Saat', 4: '>10 Saat'}
    calisma_basari = df.groupby('calisma_suresi')['hedef'].mean() * 100
    calisma_sayilari = df.groupby('calisma_suresi').size()

    bars = axes[0].bar([calisma_etiketleri[i] for i in calisma_basari.index],
                       calisma_basari.values,
                       color=[RENKLER['birincil'], RENKLER['vurgu'], RENKLER['basarili'], RENKLER['ikincil']],
                       edgecolor='white', width=0.6)
    axes[0].set_title("Çalışma Süresi → Başarı Oranı (%)")
    axes[0].set_xlabel("Çalışma Süresi")
    axes[0].set_ylabel("Başarı Oranı (%)")
    axes[0].set_ylim(0, 105)
    for bar, val in zip(bars, calisma_basari.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f'%{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    g3_ortalama = df.groupby('calisma_suresi')['G3'].mean()
    axes[1].plot([calisma_etiketleri[i] for i in g3_ortalama.index],
                 g3_ortalama.values,
                 marker='o', linewidth=2.5, markersize=10,
                 color=RENKLER['birincil'], markerfacecolor=RENKLER['vurgu'], markeredgewidth=2)
    axes[1].fill_between(range(len(g3_ortalama)), g3_ortalama.values,
                         alpha=0.15, color=RENKLER['birincil'])
    axes[1].set_title("Çalışma Süresi → Ortalama G3 Notu")
    axes[1].set_xlabel("Çalışma Süresi")
    axes[1].set_ylabel("Ortalama G3 Notu")
    axes[1].set_xticks(range(len(g3_ortalama)))
    axes[1].set_xticklabels([calisma_etiketleri[i] for i in g3_ortalama.index])

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/04_calisma_suresi.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/04_calisma_suresi.png")

    # ── 2.5 Devamsızlık Analizi ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Devamsızlık ve Başarı İlişkisi", fontsize=14, fontweight='bold')

    for hedef_val, renk, etiket in [(1, RENKLER['basarili'], 'Başarılı'), (0, RENKLER['basarisiz'], 'Başarısız')]:
        veri = df[df['hedef'] == hedef_val]['devamsizlik']
        axes[0].hist(veri, alpha=0.7, color=renk, label=etiket, bins=20, edgecolor='white')
    axes[0].axvline(df[df['hedef'] == 1]['devamsizlik'].mean(),
                    color=RENKLER['basarili'], linestyle='--', lw=2, label='Başarılı Ort.')
    axes[0].axvline(df[df['hedef'] == 0]['devamsizlik'].mean(),
                    color=RENKLER['basarisiz'], linestyle='--', lw=2, label='Başarısız Ort.')
    axes[0].set_title("Devamsızlık Dağılımı")
    axes[0].set_xlabel("Devamsızlık Günü")
    axes[0].set_ylabel("Öğrenci Sayısı")
    axes[0].legend()

    devamsizlik_gruplari = pd.cut(df['devamsizlik'], bins=[0, 3, 7, 14, 28], labels=['0-3', '4-7', '8-14', '15+'])
    devamsizlik_basari = df.groupby(devamsizlik_gruplari, observed=True)['hedef'].mean() * 100
    bars2 = axes[1].bar(devamsizlik_basari.index, devamsizlik_basari.values,
                        color=[RENKLER['basarili'], RENKLER['vurgu'], RENKLER['ikincil'], RENKLER['basarisiz']],
                        edgecolor='white', width=0.6)
    axes[1].set_title("Devamsızlık Grubu → Başarı Oranı (%)")
    axes[1].set_xlabel("Devamsızlık Gün Aralığı")
    axes[1].set_ylabel("Başarı Oranı (%)")
    axes[1].set_ylim(0, 105)
    for bar, val in zip(bars2, devamsizlik_basari.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f'%{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/05_devamsizlik.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/05_devamsizlik.png")

    # ── 2.6 Kategorik Değişken Analizi ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Kategorik Değişkenler ve Başarı Oranları", fontsize=16, fontweight='bold')

    kategorik_analizler = [
        ('cinsiyet', 'Cinsiyet', {'K': 'Kız', 'E': 'Erkek'}),
        ('adres', 'Yerleşim Yeri', None),
        ('aile_buyuklugu', 'Aile Büyüklüğü', None),
        ('internet', 'İnternet Erişimi', {0: 'Yok', 1: 'Var'}),
        ('dershane_destegi', 'Dershane Desteği', {0: 'Yok', 1: 'Var'}),
        ('ebeveyn_durum', 'Ebeveyn Durumu', None),
    ]

    for idx, (sutun, baslik, etiket_map) in enumerate(kategorik_analizler):
        ax = axes[idx // 3][idx % 3]
        basari_orani = df.groupby(sutun)['hedef'].mean() * 100

        if etiket_map:
            basari_orani.index = [etiket_map.get(k, k) for k in basari_orani.index]

        renkler_liste = [RENKLER['birincil'], RENKLER['ikincil'], RENKLER['vurgu']][:len(basari_orani)]
        bars = ax.bar(basari_orani.index.astype(str), basari_orani.values,
                      color=renkler_liste[:len(basari_orani)], edgecolor='white', width=0.5)
        ax.set_title(f"{baslik} → Başarı Oranı")
        ax.set_ylabel("Başarı Oranı (%)")
        ax.set_ylim(0, 105)
        for bar, val in zip(bars, basari_orani.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'%{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/06_kategorik.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/06_kategorik.png")

    print("\n  [EDA TAMAMLANDI]")
    return istatistikler


# =============================================================================
# 3. VERİ ÖN İŞLEME
# =============================================================================

def veri_on_isleme(df):
    print("\n" + "="*70)
    print("  VERİ ÖN İŞLEME")
    print("="*70)

    df_islenmis = df.copy()

    # ── 3.1 Eksik Veri Kontrolü ─────────────────────────────────────────────
    print("\n[3.1] Eksik veri kontrolü:")
    eksik = df_islenmis.isnull().sum()
    print(f"  Toplam eksik değer: {eksik.sum()} (Tüm değerler eksiksiz!)")

    # ── 3.2 Encoding ────────────────────────────────────────────────────────
    print("\n[3.2] Kategorik veri dönüşümü (Label Encoding):")
    le = LabelEncoder()
    kategorik_sutunlar = ['cinsiyet', 'adres', 'aile_buyuklugu', 'ebeveyn_durum']
    for sutun in kategorik_sutunlar:
        df_islenmis[sutun] = le.fit_transform(df_islenmis[sutun])
        print(f"  ✓ {sutun}: {list(le.classes_)} → {list(range(len(le.classes_)))}")

    # ── 3.3 Özellik-Hedef Ayrımı ────────────────────────────────────────────
    ozellik_sutunlari = [c for c in df_islenmis.columns if c not in ['G3', 'hedef']]
    X = df_islenmis[ozellik_sutunlari]
    y = df_islenmis['hedef']

    # ── 3.4 Train-Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n[3.3] Eğitim/Test ayrımı:")
    print(f"  ✓ Eğitim seti: {X_train.shape[0]} örnek")
    print(f"  ✓ Test seti:   {X_test.shape[0]} örnek")

    # ── 3.5 Ölçeklendirme ───────────────────────────────────────────────────
    print("\n[3.4] Standardizasyon (StandardScaler):")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=ozellik_sutunlari)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=ozellik_sutunlari)
    print(f"  ✓ Ölçeklendirme uygulandı (μ=0, σ=1)")

    # ── 3.6 Özellik Önemi (Information Gain) ────────────────────────────────
    print("\n[3.5] Özellik Seçimi (Information Gain / Mutual Information):")
    ig_skorlari = mutual_info_classif(X_train, y_train, random_state=42)
    ig_df = pd.DataFrame({'Özellik': ozellik_sutunlari, 'IG Skoru': ig_skorlari})
    ig_df = ig_df.sort_values('IG Skoru', ascending=False)

    print(ig_df.to_string(index=False))

    # IG Görselleştirme
    fig, ax = plt.subplots(figsize=(12, 7))
    renkler_ig = [RENKLER['basarili'] if i < 5 else RENKLER['birincil'] if i < 10 else '#94a3b8'
                  for i in range(len(ig_df))]
    bars = ax.barh(ig_df['Özellik'], ig_df['IG Skoru'], color=renkler_ig, edgecolor='white', height=0.7)
    ax.set_title("Özellik Önemi (Information Gain)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Information Gain Skoru")
    ax.invert_yaxis()
    for bar, val in zip(bars, ig_df['IG Skoru']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    yesil_yama = mpatches.Patch(color=RENKLER['basarili'], label='En Önemli 5')
    mavi_yama = mpatches.Patch(color=RENKLER['birincil'], label='Orta Öneme')
    gri_yama = mpatches.Patch(color='#94a3b8', label='Düşük Önem')
    ax.legend(handles=[yesil_yama, mavi_yama, gri_yama])

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/07_ozellik_onemi.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/07_ozellik_onemi.png")

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, ig_df, ozellik_sutunlari


# =============================================================================
# 4. MODEL EĞİTİMİ VE HİPERPARAMETRE OPTİMİZASYONU
# =============================================================================

def modelleri_egit(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, ozellik_sutunlari):
    print("\n" + "="*70)
    print("  MODEL EĞİTİMİ VE HİPERPARAMETRE OPTİMİZASYONU (Grid Search)")
    print("="*70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hiperparametre ızgara tanımları
    param_grids = {
        'Karar Ağacı': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']},
            'scaled': False
        },
        'Rastgele Orman': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
            'scaled': False
        },
        'Lojistik Regresyon': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']},
            'scaled': True
        },
        'K-NN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
            'scaled': True
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
            'scaled': True
        }
    }

    sonuclar = {}

    for model_adi, config in param_grids.items():
        print(f"\n  [{model_adi}] eğitiliyor...")
        X_tr = X_train_scaled if config['scaled'] else X_train
        X_te = X_test_scaled if config['scaled'] else X_test

        grid_search = GridSearchCV(config['model'], config['params'], cv=cv,
                                   scoring='f1', n_jobs=-1, refit=True)
        grid_search.fit(X_tr, y_train)
        en_iyi_model = grid_search.best_estimator_
        y_pred = en_iyi_model.predict(X_te)
        y_prob = en_iyi_model.predict_proba(X_te)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        cv_skorlari = cross_val_score(en_iyi_model, X_tr, y_train, cv=cv, scoring='accuracy')

        sonuclar[model_adi] = {
            'model': en_iyi_model,
            'en_iyi_parametreler': grid_search.best_params_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_ort': cv_skorlari.mean(),
            'cv_std': cv_skorlari.std(),
            'y_pred': y_pred,
            'y_prob': y_prob,
            'X_te': X_te
        }

        print(f"    ✓ En iyi parametreler: {grid_search.best_params_}")
        print(f"    ✓ Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(f"    ✓ CV Doğruluk: {cv_skorlari.mean():.4f} ± {cv_skorlari.std():.4f}")

    return sonuclar


# =============================================================================
# 5. DEĞERLENDİRME VE GÖRSELLEŞTİRME
# =============================================================================

def degerlendirme_ve_gorsellestirme(sonuclar, y_test):
    print("\n" + "="*70)
    print("  MODEL DEĞERLENDİRME VE KARŞILAŞTIRMA")
    print("="*70)

    # ── 5.1 Sonuç Tablosu ───────────────────────────────────────────────────
    tablo_verileri = []
    for model_adi, s in sonuclar.items():
        tablo_verileri.append({
            'Model': model_adi,
            'Accuracy': f"{s['accuracy']:.4f}",
            'Precision': f"{s['precision']:.4f}",
            'Recall': f"{s['recall']:.4f}",
            'F1-Skoru': f"{s['f1']:.4f}",
            'ROC-AUC': f"{s['auc']:.4f}",
            'CV Ort (±std)': f"{s['cv_ort']:.4f} ±{s['cv_std']:.4f}"
        })

    sonuc_df = pd.DataFrame(tablo_verileri)
    print("\n  KARŞILAŞTIRMA TABLOSU:")
    print(sonuc_df.to_string(index=False))

    sonuc_df.to_csv("model_karsilastirma.csv", index=False, encoding='utf-8-sig')
    print(f"\n  ✓ Sonuçlar kaydedildi: model_karsilastirma.csv")

    # ── 5.2 Metrik Karşılaştırma Grafiği ────────────────────────────────────
    model_adlari = list(sonuclar.keys())
    metrikler = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metrik_etiketleri = ['Doğruluk', 'Kesinlik', 'Duyarlılık', 'F1-Skoru', 'ROC-AUC']
    renk_paleti = [RENKLER['birincil'], RENKLER['ikincil'], RENKLER['basarili'],
                   RENKLER['vurgu'], '#8b5cf6']

    x = np.arange(len(model_adlari))
    genislik = 0.15

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (metrik, etiket, renk) in enumerate(zip(metrikler, metrik_etiketleri, renk_paleti)):
        degerler = [sonuclar[m][metrik] for m in model_adlari]
        offset = (i - len(metrikler)/2) * genislik + genislik/2
        bars = ax.bar(x + offset, degerler, genislik, label=etiket, color=renk, alpha=0.85, edgecolor='white')

    ax.set_xlabel("Model")
    ax.set_ylabel("Skor")
    ax.set_title("Model Karşılaştırması — Tüm Metrikler", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_adlari, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='%80 Eşiği')

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/08_model_karsilastirma.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/08_model_karsilastirma.png")

    # ── 5.3 Confusion Matrix'ler ─────────────────────────────────────────────
    n_model = len(sonuclar)
    fig, axes = plt.subplots(1, n_model, figsize=(20, 4))
    fig.suptitle("Karmaşıklık Matrisleri (Confusion Matrix)", fontsize=14, fontweight='bold')

    for ax, (model_adi, s) in zip(axes, sonuclar.items()):
        cm = confusion_matrix(y_test, s['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Başarısız', 'Başarılı'],
                    yticklabels=['Başarısız', 'Başarılı'],
                    linewidths=0.5, annot_kws={'size': 13, 'weight': 'bold'})
        ax.set_title(f"{model_adi}\nAcc: {s['accuracy']:.2f}", fontsize=11)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/09_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/09_confusion_matrix.png")

    # ── 5.4 ROC Eğrileri ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    roc_renkler = [RENKLER['birincil'], RENKLER['ikincil'], RENKLER['basarili'],
                   RENKLER['vurgu'], '#8b5cf6']

    for (model_adi, s), renk in zip(sonuclar.items(), roc_renkler):
        fpr, tpr, _ = roc_curve(y_test, s['y_prob'])
        ax.plot(fpr, tpr, lw=2.5, color=renk, label=f"{model_adi} (AUC = {s['auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Rastgele Tahmin')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=12)
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=12)
    ax.set_title("ROC Eğrileri — Tüm Modeller", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/10_roc_egrileri.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/10_roc_egrileri.png")

    # ── 5.5 En İyi Modeli Bul ───────────────────────────────────────────────
    en_iyi_model_adi = max(sonuclar, key=lambda x: sonuclar[x]['f1'])
    print(f"\n  🏆 EN İYİ MODEL: {en_iyi_model_adi}")
    print(f"     F1-Skoru: {sonuclar[en_iyi_model_adi]['f1']:.4f}")
    print(f"     ROC-AUC:  {sonuclar[en_iyi_model_adi]['auc']:.4f}")

    return sonuc_df, en_iyi_model_adi


# =============================================================================
# 6. KARAR DESTEK MEKANİZMASI
# =============================================================================

def karar_destek_analizi(df, sonuclar, ig_df):
    print("\n" + "="*70)
    print("  KARAR DESTEK MEKANİZMASI — EĞİTİMCİ İÇGÖRÜLERİ")
    print("="*70)

    icegorular = []

    # ── Çalışma Süresi Riski ─────────────────────────────────────────────────
    dusuk_calisma = df[df['calisma_suresi'] <= 1]['hedef'].mean()
    yuksek_calisma = df[df['calisma_suresi'] >= 3]['hedef'].mean()
    risk_arttisi = (yuksek_calisma - dusuk_calisma) / yuksek_calisma * 100
    icegorular.append(f"• Haftalık çalışma süresi <2 saat olan öğrencilerde başarı oranı %{dusuk_calisma*100:.0f} iken "
                     f"5+ saat çalışanlarda %{yuksek_calisma*100:.0f}'dir. "
                     f"Bu, %{risk_arttisi:.0f} oranında bir risk farkına işaret etmektedir.")

    # ── Devamsızlık Riski ────────────────────────────────────────────────────
    az_devamsiz = df[df['devamsizlik'] <= 3]['hedef'].mean()
    cok_devamsiz = df[df['devamsizlik'] > 14]['hedef'].mean()
    icegorular.append(f"• 0-3 gün devamsız öğrencilerde başarı oranı %{az_devamsiz*100:.0f} iken "
                     f"15+ gün devamsız öğrencilerde %{cok_devamsiz*100:.0f}'e düşmektedir.")

    # ── G1/G2 Erken Uyarı ───────────────────────────────────────────────────
    dusuk_g1 = df[df['G1'] < 10]['hedef'].mean()
    icegorular.append(f"• 1. dönem notu (G1) 10'un altında olan öğrencilerin yalnızca %{dusuk_g1*100:.0f}'i "
                     f"yılı başarıyla tamamlamaktadır. Erken tespit kritiktir.")

    # ── Alkol Riski ─────────────────────────────────────────────────────────
    az_alkol = df[df['hafta_sonu_alkol'] <= 2]['hedef'].mean()
    cok_alkol = df[df['hafta_sonu_alkol'] >= 4]['hedef'].mean()
    icegorular.append(f"• Hafta sonu yuksek alkol tuketimi (>=4) olan ogrencilerde basari orani "
                     f"%{cok_alkol*100:.0f} iken dusuk tuketimde %{az_alkol*100:.0f} olmaktadir.")

    print("\n  İÇGÖRÜLER:")
    for ig in icegorular:
        print(f"\n  {ig}")

    # ── Karar Destek Grafiği ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Karar Destek: Risk Faktörü Analizi", fontsize=14, fontweight='bold')

    # Çalışma Süresi
    cs_etiket = {1: '<2 Saat', 2: '2-5 Saat', 3: '5-10 Saat', 4: '>10 Saat'}
    cs_basari = df.groupby('calisma_suresi')['hedef'].mean() * 100
    renkler_cs = [RENKLER['basarisiz'] if v < 60 else RENKLER['vurgu'] if v < 75 else RENKLER['basarili']
                  for v in cs_basari.values]
    axes[0].bar([cs_etiket[i] for i in cs_basari.index], cs_basari.values, color=renkler_cs, edgecolor='white')
    axes[0].axhline(y=75, color='red', linestyle='--', alpha=0.7, label='%75 Eşiği')
    axes[0].set_title("Çalışma Süresi → Risk")
    axes[0].set_ylabel("Başarı Oranı (%)")
    axes[0].legend()

    # Devamsızlık
    dev_grp = pd.cut(df['devamsizlik'], bins=[0, 3, 7, 14, 28], labels=['0-3', '4-7', '8-14', '15+'])
    dev_basari = df.groupby(dev_grp, observed=True)['hedef'].mean() * 100
    renkler_dev = [RENKLER['basarili'] if v >= 75 else RENKLER['vurgu'] if v >= 60 else RENKLER['basarisiz']
                   for v in dev_basari.values]
    axes[1].bar(dev_basari.index.astype(str), dev_basari.values, color=renkler_dev, edgecolor='white')
    axes[1].axhline(y=75, color='red', linestyle='--', alpha=0.7, label='%75 Eşiği')
    axes[1].set_title("Devamsızlık → Risk")
    axes[1].set_ylabel("Başarı Oranı (%)")
    axes[1].legend()

    # G1 Notu
    g1_grp = pd.cut(df['G1'], bins=[0, 6, 10, 14, 20], labels=['0-6', '7-10', '11-14', '15+'])
    g1_basari = df.groupby(g1_grp, observed=True)['hedef'].mean() * 100
    renkler_g1 = [RENKLER['basarisiz'] if v < 50 else RENKLER['vurgu'] if v < 75 else RENKLER['basarili']
                  for v in g1_basari.values]
    axes[2].bar(g1_basari.index.astype(str), g1_basari.values, color=renkler_g1, edgecolor='white')
    axes[2].axhline(y=75, color='red', linestyle='--', alpha=0.7, label='%75 Eşiği')
    axes[2].set_title("G1 Notu → Risk")
    axes[2].set_ylabel("Başarı Oranı (%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{GRAFIK_KLASORU}/11_karar_destek.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Grafik kaydedildi: {GRAFIK_KLASORU}/11_karar_destek.png")

    return icegorular


# =============================================================================
# 7. ANA ÇALIŞMA FONKSİYONU
# =============================================================================

def main():
    print("\n")
    print("=" * 72)
    print("  OGRENCI BASARISININ TAHMINI -- VERI MADDENCİLİGİ PROJESİ")
    print("  CRISP-DM Metodolojisi ile Eksiksiz Uygulama")
    print("=" * 72)

    # 1. Veri Üretimi
    df = veri_seti_olustur()

    # 2. EDA
    istatistikler = veri_kesifsel_analiz(df)

    # 3. Ön İşleme
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, ig_df, ozellik_sutunlari = \
        veri_on_isleme(df)

    # 4. Modelleme
    sonuclar = modelleri_egit(X_train, X_test, X_train_scaled, X_test_scaled,
                               y_train, y_test, ozellik_sutunlari)

    # 5. Değerlendirme
    sonuc_df, en_iyi_model_adi = degerlendirme_ve_gorsellestirme(sonuclar, y_test)

    # 6. Karar Destek
    icegorular = karar_destek_analizi(df, sonuclar, ig_df)

    # ── Özet Rapor ──────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  PROJE TAMAMLANDI — ÖZET RAPOR")
    print("="*72)
    print(f"\n  📊 Veri Seti: {len(df)} öğrenci, {df.shape[1]-1} özellik")
    print(f"  🏆 En İyi Model: {en_iyi_model_adi}")
    print(f"     F1-Skoru : {sonuclar[en_iyi_model_adi]['f1']:.4f}")
    print(f"     ROC-AUC  : {sonuclar[en_iyi_model_adi]['auc']:.4f}")
    print(f"     Doğruluk : {sonuclar[en_iyi_model_adi]['accuracy']:.4f}")
    print(f"\n  📁 Grafik klasörü  : {GRAFIK_KLASORU}/  (11 grafik)")
    print(f"  📄 Karşılaştırma   : model_karsilastirma.csv")
    print("\n" + "="*72)


if __name__ == "__main__":
    main()
