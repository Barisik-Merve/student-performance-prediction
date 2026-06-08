# -*- coding: utf-8 -*-
"""
data_loader.py — UCI Student Performance veri seti indirme ve ön işleme.
Feature engineering ve gelişmiş ön işleme ile model başarısını artırır.
"""

import os
import io
import requests
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "student-mat.csv")

# UCI doğrudan indirme URL (zip içinde student-mat.csv)
UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


@st.cache_data(show_spinner=False)
def veri_indir_ve_yukle() -> pd.DataFrame:
    """
    UCI'dan veri setini indirir (ilk çalışmada), önbelleğe alır ve yükler.
    Eğer indirme başarısız olursa sentetik yedek veri üretir.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        try:
            r = requests.get(UCI_URL, timeout=15)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                # zip içinde Student.zip daha var
                if "Student.zip" in z.namelist():
                    inner_bytes = z.read("Student.zip")
                    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z2:
                        with z2.open("student-mat.csv") as f:
                            content = f.read().decode("utf-8", errors="replace")
                else:
                    # Direkt csv
                    with z.open("student-mat.csv") as f:
                        content = f.read().decode("utf-8", errors="replace")
            with open(CSV_PATH, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            # İndirme başarısız → sentetik veri üret ve kaydet (;  sep  ile UCI formatında)
            df_synth = _sentetik_veri_uret(395)
            df_synth.to_csv(CSV_PATH, index=False, sep=";")

    # Separator otomatik tespiti
    with open(CSV_PATH, "r", encoding="utf-8", errors="replace") as _f:
        _first = _f.readline()
    _sep = ";" if ";" in _first else ","
    try:
        df = pd.read_csv(CSV_PATH, sep=_sep, encoding="utf-8")
    except Exception:
        df = pd.read_csv(CSV_PATH, sep=_sep, encoding="latin-1")

    # Sütun adlarını Türkçeye eşle
    df = _sutun_adlarini_turkcelestir(df)
    return df


def _sutun_adlarini_turkcelestir(df: pd.DataFrame) -> pd.DataFrame:
    kolumlar = {
        "school": "okul", "sex": "cinsiyet", "age": "yas", "address": "adres",
        "famsize": "aile_buyuklugu", "Pstatus": "ebeveyn_durum",
        "Medu": "anne_egitim", "Fedu": "baba_egitim",
        "Mjob": "anne_meslek", "Fjob": "baba_meslek",
        "reason": "tercih_nedeni", "guardian": "vasi",
        "traveltime": "ulasim_suresi", "studytime": "calisma_suresi",
        "failures": "sinif_tekrar", "schoolsup": "okul_destegi",
        "famsup": "aile_egitim_destegi", "paid": "ekstra_ders",
        "activities": "ekstra_aktiviteler", "nursery": "anaokulu",
        "higher": "universite_hedefi", "internet": "internet",
        "romantic": "romantik_iliski", "famrel": "aile_iliskisi",
        "freetime": "serbest_zaman", "goout": "sosyallestirme",
        "Dalc": "hafta_ici_alkol", "Walc": "hafta_sonu_alkol",
        "health": "saglik", "absences": "devamsizlik",
        "G1": "G1", "G2": "G2", "G3": "G3"
    }
    df = df.rename(columns={k: v for k, v in kolumlar.items() if k in df.columns})
    return df


def _sentetik_veri_uret(n=395) -> pd.DataFrame:
    """UCI veri setine dayalı gerçekçi sentetik yedek."""
    np.random.seed(42)
    yas = np.random.randint(15, 23, n)
    cinsiyet = np.random.choice(["F", "M"], n, p=[0.52, 0.48])
    adres = np.random.choice(["U", "R"], n, p=[0.77, 0.23])
    aile_buyuklugu = np.random.choice(["LE3", "GT3"], n, p=[0.33, 0.67])
    ebeveyn_durum = np.random.choice(["T", "A"], n, p=[0.89, 0.11])
    anne_egitim = np.random.choice([0,1,2,3,4], n, p=[0.17,0.20,0.31,0.23,0.09])
    baba_egitim = np.random.choice([0,1,2,3,4], n, p=[0.22,0.23,0.28,0.20,0.07])
    calisma_suresi = np.random.choice([1,2,3,4], n, p=[0.24,0.40,0.23,0.13])
    devamsizlik = np.random.randint(0, 28, n)
    sinif_tekrar = np.random.choice([0,1,2,3], n, p=[0.82,0.10,0.05,0.03])
    aile_egitim_destegi = np.random.choice(["yes","no"], n, p=[0.60,0.40])
    ekstra_aktiviteler = np.random.choice(["yes","no"], n, p=[0.53,0.47])
    internet = np.random.choice(["yes","no"], n, p=[0.66,0.34])
    hafta_ici_alkol = np.random.choice([1,2,3,4,5], n, p=[0.59,0.20,0.10,0.07,0.04])
    hafta_sonu_alkol = np.random.choice([1,2,3,4,5], n, p=[0.30,0.24,0.20,0.14,0.12])
    serbest_zaman = np.random.choice([1,2,3,4,5], n, p=[0.07,0.18,0.33,0.29,0.13])
    sosyallestirme = np.random.choice([1,2,3,4,5], n, p=[0.11,0.22,0.32,0.24,0.11])
    saglik = np.random.choice([1,2,3,4,5], n, p=[0.09,0.12,0.22,0.26,0.31])
    internet_bin = (internet == "yes").astype(int)
    saglik_effect = saglik * 0.5
    temel = (calisma_suresi*1.8 + (anne_egitim+baba_egitim)*0.4 +
             saglik_effect + internet_bin*0.7 -
             devamsizlik*0.18 - sinif_tekrar*1.5 -
             hafta_sonu_alkol*0.6 + np.random.normal(8,1.5,n))
    G1 = np.clip(np.round(temel + np.random.normal(0,1.2,n)), 0, 20).astype(int)
    G2 = np.clip(np.round(G1*0.85 + np.random.normal(1.5,1.0,n)), 0, 20).astype(int)
    G3 = np.clip(np.round(G1*0.55 + G2*0.35 + np.random.normal(0.8,0.9,n)), 0, 20).astype(int)

    return pd.DataFrame({
        "okul": np.random.choice(["GP","MS"], n, p=[0.79, 0.21]),
        "cinsiyet": cinsiyet, "yas": yas, "adres": adres,
        "aile_buyuklugu": aile_buyuklugu, "ebeveyn_durum": ebeveyn_durum,
        "anne_egitim": anne_egitim, "baba_egitim": baba_egitim,
        "calisma_suresi": calisma_suresi, "sinif_tekrar": sinif_tekrar,
        "aile_egitim_destegi": aile_egitim_destegi,
        "ekstra_aktiviteler": ekstra_aktiviteler,
        "internet": internet,
        "hafta_ici_alkol": hafta_ici_alkol, "hafta_sonu_alkol": hafta_sonu_alkol,
        "serbest_zaman": serbest_zaman, "sosyallestirme": sosyallestirme,
        "saglik": saglik, "devamsizlik": devamsizlik,
        "G1": G1, "G2": G2, "G3": G3
    })


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yeni özellikler türeterek model başarısını artır.
    """
    df2 = df.copy()

    # 1. Not ortalaması (G1 ve G2)
    if "G1" in df2.columns and "G2" in df2.columns:
        df2["not_ortalama"] = (df2["G1"] + df2["G2"]) / 2.0
        # 2. Akademik trend (G2 - G1: yükselen/düşen performans)
        df2["akademik_trend"] = df2["G2"] - df2["G1"]
        # 3. G1 * G2 etkileşim terimi
        df2["g1_g2_carpim"] = df2["G1"] * df2["G2"]
        # 4. Not varyansı (tutarlılık göstergesi)
        df2["not_varyans"] = df2[["G1", "G2"]].var(axis=1)
        # 5. Minimum not (en zayıf dönem)
        df2["min_not"] = df2[["G1", "G2"]].min(axis=1)
        # 6. Başarı eşiği altı mı (G1 veya G2 < 10)
        df2["dusuk_not_var"] = ((df2["G1"] < 10) | (df2["G2"] < 10)).astype(int)

    # 7. Ebeveyn eğitim toplamı ve farkı
    if "anne_egitim" in df2.columns and "baba_egitim" in df2.columns:
        df2["ebeveyn_egitim_toplam"] = df2["anne_egitim"] + df2["baba_egitim"]
        df2["ebeveyn_egitim_max"] = df2[["anne_egitim", "baba_egitim"]].max(axis=1)

    # 8. Toplam alkol tüketimi
    if "hafta_ici_alkol" in df2.columns and "hafta_sonu_alkol" in df2.columns:
        df2["toplam_alkol"] = df2["hafta_ici_alkol"] + df2["hafta_sonu_alkol"]

    # 9. Sosyal aktivite skoru
    if "serbest_zaman" in df2.columns and "sosyallestirme" in df2.columns:
        df2["sosyal_skor"] = df2["serbest_zaman"] + df2["sosyallestirme"]

    # 10. Çalışma-devamsızlık oranı
    if "calisma_suresi" in df2.columns and "devamsizlik" in df2.columns:
        df2["calisma_devamsizlik_oran"] = df2["calisma_suresi"] / (df2["devamsizlik"] + 1)

    # 11. Risk skoru (birleşik)
    risk = np.zeros(len(df2))
    if "sinif_tekrar" in df2.columns:
        risk += df2["sinif_tekrar"] * 2
    if "devamsizlik" in df2.columns:
        risk += (df2["devamsizlik"] > 10).astype(int) * 1.5
    if "hafta_sonu_alkol" in df2.columns:
        risk += (df2["hafta_sonu_alkol"] >= 4).astype(int) * 1
    if "calisma_suresi" in df2.columns:
        risk += (df2["calisma_suresi"] <= 1).astype(int) * 1
    df2["risk_skoru"] = risk

    # 12. Devamsızlık kategorisi
    if "devamsizlik" in df2.columns:
        df2["devamsizlik_kategori"] = pd.cut(
            df2["devamsizlik"],
            bins=[-1, 3, 7, 14, 100],
            labels=[0, 1, 2, 3]
        ).astype(int)

    return df2


@st.cache_data(show_spinner=False)
def veri_isle(df: pd.DataFrame):
    """
    Ham veriyi makine öğrenmesi için hazırlar.
    Feature engineering ile zenginleştirilmiş versiyon.
    Döndürür: X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test, ig_df, ozellikler, scaler, df_islenmis
    """
    df2 = df.copy()

    # Hedef
    df2["hedef"] = (df2["G3"] >= 10).astype(int)

    # Feature Engineering uygula
    df2 = _feature_engineering(df2)

    # Kategorik sütunları encode et
    le = LabelEncoder()
    kat_sutunlar = df2.select_dtypes(include="object").columns.tolist()
    for s in kat_sutunlar:
        df2[s] = le.fit_transform(df2[s].astype(str))

    # Özellikler — G3 ve hedef hariç tüm sütunları kullan (G1, G2 dahil)
    exclude = {"G3", "hedef"}
    ozellikler = [c for c in df2.columns if c not in exclude]

    X = df2[ozellikler]
    y = df2["hedef"]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=ozellikler)
    X_te_sc = pd.DataFrame(scaler.transform(X_test), columns=ozellikler)

    # Information Gain
    ig_scores = mutual_info_classif(X_train, y_train, random_state=42)
    ig_df = pd.DataFrame({"Ozellik": ozellikler, "IG": ig_scores})
    ig_df = ig_df.sort_values("IG", ascending=False).reset_index(drop=True)

    return X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test, ig_df, ozellikler, scaler, df2
