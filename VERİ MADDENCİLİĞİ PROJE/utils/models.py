# -*- coding: utf-8 -*-
"""
models.py — 5 makine öğrenmesi modeli, GridSearchCV ile hiperparametre optimizasyonu.
Geliştirilmiş parametre arama alanı ve SMOTE dengeli eğitim.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

MODEL_KONFIGURASYONLARI = {
    "Karar Agaci": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 3, 5, 10, 15],
            "min_samples_leaf": [1, 2, 3, 5],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2", None]
        },
        "scaled": False,
        "icon": "🌳",
        "renk": "#22c55e",
        "aciklama": "Yorumlanabilir karar kuralları oluşturur"
    },
    "Rastgele Orman": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 3, 5],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", None]
        },
        "scaled": False,
        "icon": "🌲",
        "renk": "#6366f1",
        "aciklama": "Ensemble öğrenme ile overfitting azaltılır"
    },
    "Lojistik Regresyon": {
        "model": LogisticRegression(random_state=42, max_iter=2000),
        "params": {
            "C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
            "solver": ["lbfgs", "liblinear", "saga"],
            "penalty": ["l2"],
            "class_weight": ["balanced", None]
        },
        "scaled": True,
        "icon": "📈",
        "renk": "#f59e0b",
        "aciklama": "İstatistiksel olasılık tahmini"
    },
    "K-NN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "p": [1, 2, 3]
        },
        "scaled": True,
        "icon": "🔵",
        "renk": "#ec4899",
        "aciklama": "Benzer öğrenci profili gruplaması"
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 0.5, 1, 5, 10, 50],
            "kernel": ["rbf", "linear", "poly"],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "class_weight": ["balanced", None]
        },
        "scaled": True,
        "icon": "⚡",
        "renk": "#a855f7",
        "aciklama": "Yüksek boyutlu karar sınırları"
    }
}


@st.cache_data(show_spinner=False)
def modelleri_egit(_X_train, _X_test, _X_tr_sc, _X_te_sc, _y_train, _y_test):
    """
    5 modeli GridSearchCV ile eğitir ve metrikleri döndürür.
    SMOTE ile sınıf dengeleme uygulanır (varsa).
    st.cache_data ile bir kez eğitilir, sonraki çalışmalarda önbellekten gelir.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # SMOTE ile eğitim verilerini dengele
    if SMOTE_AVAILABLE:
        try:
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(_X_train, _y_train)
            X_tr_sc_res, _ = smote.fit_resample(_X_tr_sc, _y_train)
            y_tr_sc_res = y_train_res  # aynı resampling
        except Exception:
            X_train_res, y_train_res = _X_train, _y_train
            X_tr_sc_res, y_tr_sc_res = _X_tr_sc, _y_train
    else:
        X_train_res, y_train_res = _X_train, _y_train
        X_tr_sc_res, y_tr_sc_res = _X_tr_sc, _y_train

    sonuclar = {}

    for model_adi, cfg in MODEL_KONFIGURASYONLARI.items():
        if cfg["scaled"]:
            X_tr = X_tr_sc_res if SMOTE_AVAILABLE else _X_tr_sc
            y_tr = y_tr_sc_res if SMOTE_AVAILABLE else _y_train
        else:
            X_tr = X_train_res if SMOTE_AVAILABLE else _X_train
            y_tr = y_train_res if SMOTE_AVAILABLE else _y_train
        X_te = _X_te_sc if cfg["scaled"] else _X_test

        gs = GridSearchCV(cfg["model"], cfg["params"], cv=cv,
                          scoring="f1", n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        model = gs.best_estimator_

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
        fpr, tpr, thresholds = roc_curve(_y_test, y_prob)
        cm = confusion_matrix(_y_test, y_pred)

        sonuclar[model_adi] = {
            "model": model,
            "en_iyi_params": gs.best_params_,
            "accuracy": accuracy_score(_y_test, y_pred),
            "precision": precision_score(_y_test, y_pred, zero_division=0),
            "recall": recall_score(_y_test, y_pred, zero_division=0),
            "f1": f1_score(_y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(_y_test, y_prob),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "fpr": fpr,
            "tpr": tpr,
            "cm": cm,
            "icon": cfg["icon"],
            "renk": cfg["renk"],
            "aciklama": cfg["aciklama"]
        }

    return sonuclar
