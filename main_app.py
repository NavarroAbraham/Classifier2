import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digits Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        padding: 1rem; border-radius: 10px; color: white;
        text-align: center; margin: 0.25rem 0;
    }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #e94560; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    .info-box {
        background: #16213e; border-left: 4px solid #e94560;
        padding: 0.75rem 1rem; border-radius: 5px; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    digits = load_digits()
    df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
    df["digit"] = digits.target
    return df, digits

df, digits = load_data()

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
    "Árbol de Decisión":   DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
}

METRICS_FN = {
    "Accuracy":  lambda y, yp: accuracy_score(y, yp),
    "Precision": lambda y, yp: precision_score(y, yp, average="weighted", zero_division=0),
    "Recall":    lambda y, yp: recall_score(y, yp, average="weighted", zero_division=0),
    "F1-Score":  lambda y, yp: f1_score(y, yp, average="weighted", zero_division=0),
}

PALETTE = px.colors.qualitative.Bold

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuración")

st.sidebar.header("📊 Datos")
test_size    = st.sidebar.slider("Tamaño del conjunto de prueba", 0.10, 0.40, 0.20, 0.05)
random_state = st.sidebar.number_input("Semilla aleatoria", 0, 999, 42)
scale_data   = st.sidebar.checkbox("Normalizar datos (StandardScaler)", value=True)
use_pca      = st.sidebar.checkbox("Reducir con PCA antes de entrenar", value=False)
if use_pca:
    n_components = st.sidebar.slider("Componentes PCA", 5, 64, 30)

st.sidebar.header("🤖 Modelos")
selected_models = st.sidebar.multiselect(
    "Seleccionar modelos",
    list(MODELS.keys()),
    default=["Regresión Logística", "Random Forest", "SVM (RBF)"],
)

st.sidebar.header("📐 Métricas")
selected_metrics = st.sidebar.multiselect(
    "Seleccionar métricas",
    list(METRICS_FN.keys()),
    default=list(METRICS_FN.keys()),
)

st.sidebar.header("🧮 PCA / Visualización")
pca_components_viz = st.sidebar.slider("Componentes PCA para visualización 2D/3D", 2, 3, 2)
show_cv            = st.sidebar.checkbox("Validación cruzada (5-fold)", value=True)
show_lc            = st.sidebar.checkbox("Curvas de aprendizaje", value=False)
show_digit_gallery = st.sidebar.checkbox("Galería de dígitos", value=True)
n_digits_gallery   = st.sidebar.slider("Dígitos en galería", 10, 50, 20)
n_error_show       = st.sidebar.slider("Errores a mostrar", 5, 30, 10)

# ── Prepare data ──────────────────────────────────────────────────────────────
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)

# ── Train & Evaluate ──────────────────────────────────────────────────────────
@st.cache_data
def train_and_evaluate(model_name, _Xtr, _Xte, ytr, yte, scale, pca, n_comp, cv, seed, n_feat):
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if pca:
        steps.append(("pca", PCA(n_components=n_comp, random_state=seed)))
    steps.append(("clf", MODELS[model_name]))
    pipe = Pipeline(steps)
    pipe.fit(_Xtr, ytr)
    yp   = pipe.predict(_Xte)
    yprob = pipe.predict_proba(_Xte)
    scores = {m: fn(yte, yp) for m, fn in METRICS_FN.items()}
    cv_sc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy") if cv else None
    return pipe, yp, yprob, scores, cv_sc

if not selected_models:
    st.warning("⚠️ Selecciona al menos un modelo en el panel lateral.")
    st.stop()

with st.spinner("Entrenando modelos..."):
    results = {}
    for name in selected_models:
        pipe, yp, yprob, scores, cv_sc = train_and_evaluate(
            name, X_train, X_test, y_train, y_test,
            scale_data, use_pca, int(n_components) if use_pca else 30,
            show_cv, int(random_state), X_train.shape[1],
        )
        results[name] = {"pipe": pipe, "y_pred": yp, "y_prob": yprob,
                         "scores": scores, "cv_scores": cv_sc}

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔢 MNIST Digits — Panel de Clasificación")
st.caption(
    f"Dataset: {len(df)} muestras · 64 píxeles (8×8) · 10 clases (0–9) · "
    f"Train: {len(X_train)} | Test: {len(X_test)}"
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Métricas",
    "🔢 Matrices de Confusión",
    "📉 Curvas ROC",
    "🖼️ Galería & Errores",
    "🌀 Visualización PCA",
    "🌳 Importancia",
    "📈 Curvas de Aprendizaje",
    "🔍 Dataset",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · Métricas
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Comparativa de Métricas de Desempeño")

    for mname, res in results.items():
        st.markdown(f"#### 🤖 {mname}")
        cols = st.columns(len(selected_metrics))
        for col, metric in zip(cols, selected_metrics):
            val = res["scores"][metric]
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val:.4f}</div>
                <div class="metric-label">{metric}</div>
            </div>""", unsafe_allow_html=True)

        if show_cv and res["cv_scores"] is not None:
            cv = res["cv_scores"]
            st.caption(f"CV Accuracy (5-fold): {cv.mean():.4f} ± {cv.std():.4f}  |  "
                       f"Min: {cv.min():.4f}  Max: {cv.max():.4f}")
        st.markdown("---")

    # Grouped bar comparison
    if len(results) > 1 and selected_metrics:
        st.subheader("📊 Comparación Visual entre Modelos")
        comp_rows = []
        for mname, res in results.items():
            for metric in selected_metrics:
                comp_rows.append({"Modelo": mname, "Métrica": metric,
                                  "Valor": res["scores"][metric]})
        fig = px.bar(
            pd.DataFrame(comp_rows), x="Métrica", y="Valor", color="Modelo",
            barmode="group", range_y=[0, 1.05],
            color_discrete_sequence=PALETTE,
            title="Métricas comparadas",
        )
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          font_color="white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # CV boxplot
    if show_cv:
        cv_rows = []
        for mname, res in results.items():
            if res["cv_scores"] is not None:
                for s in res["cv_scores"]:
                    cv_rows.append({"Modelo": mname, "Accuracy (CV)": s})
        if cv_rows:
            st.subheader("🎯 Distribución Validación Cruzada 5-fold")
            fig2 = px.box(
                pd.DataFrame(cv_rows), x="Modelo", y="Accuracy (CV)",
                color="Modelo", points="all",
                color_discrete_sequence=PALETTE,
            )
            fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                               font_color="white", height=380)
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🔢 Matrices de Confusión")

    conf_mode = st.radio("Mostrar valores como:", ["Conteos", "Porcentajes"], horizontal=True)

    for mname, res in results.items():
        cm = confusion_matrix(y_test, res["y_pred"])
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True) if conf_mode == "Porcentajes" else cm
        fmt = ".2f" if conf_mode == "Porcentajes" else "d"

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            cm_display, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=range(10), yticklabels=range(10),
            ax=ax, linewidths=0.3, linecolor="#333",
        )
        ax.set_title(f"Matriz de Confusión — {mname}", fontsize=13, fontweight="bold", color="white")
        ax.set_xlabel("Predicho", color="white")
        ax.set_ylabel("Real", color="white")
        ax.tick_params(colors="white")
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1a1a2e")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown("---")

    # Per-class metrics
    st.subheader("📋 Reporte por Clase")
    report_sel = st.selectbox("Modelo:", list(results.keys()), key="rpt")
    rpt = classification_report(
        y_test, results[report_sel]["y_pred"],
        output_dict=True, zero_division=0,
    )
    rpt_df = pd.DataFrame(rpt).T
    numeric_cols = rpt_df.select_dtypes(include="number").columns
    st.dataframe(
        rpt_df.style.background_gradient(cmap="Blues", subset=["f1-score"]).format(
            {c: "{:.4f}" for c in numeric_cols}
        ),
        use_container_width=True,
    )

    # Per-class F1 bar
    class_rows = []
    for mname, res in results.items():
        f1s = f1_score(y_test, res["y_pred"], average=None, zero_division=0)
        for digit, f1 in enumerate(f1s):
            class_rows.append({"Modelo": mname, "Dígito": str(digit), "F1": f1})
    fig3 = px.bar(
        pd.DataFrame(class_rows), x="Dígito", y="F1", color="Modelo",
        barmode="group", range_y=[0, 1.05],
        color_discrete_sequence=PALETTE,
        title="F1-Score por dígito",
    )
    fig3.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                       font_color="white", height=380)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("📉 Curvas ROC (One-vs-Rest)")

    roc_model = st.selectbox("Modelo:", list(results.keys()), key="roc_sel")
    y_bin = label_binarize(y_test, classes=list(range(10)))
    y_prob = results[roc_model]["y_prob"]

    fig = make_subplots(rows=2, cols=5,
                        subplot_titles=[f"Dígito {i}" for i in range(10)])
    colors_roc = px.colors.qualitative.Alphabet

    auc_vals = []
    for digit in range(10):
        fpr, tpr, _ = roc_curve(y_bin[:, digit], y_prob[:, digit])
        roc_auc = auc(fpr, tpr)
        auc_vals.append(roc_auc)
        r, c = divmod(digit, 5)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{digit} (AUC={roc_auc:.3f})",
            line=dict(color=colors_roc[digit], width=2),
        ), row=r+1, col=c+1)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            showlegend=False,
        ), row=r+1, col=c+1)

    fig.update_layout(height=600, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font_color="white",
                      title_text=f"ROC por dígito — {roc_model}")
    st.plotly_chart(fig, use_container_width=True)

    # AUC summary
    auc_df = pd.DataFrame({
        "Dígito": list(range(10)),
        "AUC": [round(v, 4) for v in auc_vals],
    })
    macro_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    st.info(f"**Macro AUC — {roc_model}: {macro_auc:.4f}**")

    # AUC comparison across all models
    if len(results) > 1:
        st.subheader("Comparación de Macro AUC entre modelos")
        auc_comp = []
        for mname, res in results.items():
            ma = roc_auc_score(y_bin, res["y_prob"], multi_class="ovr", average="macro")
            wa = roc_auc_score(y_bin, res["y_prob"], multi_class="ovr", average="weighted")
            auc_comp.append({"Modelo": mname, "Macro AUC": round(ma, 4), "Weighted AUC": round(wa, 4)})
        st.dataframe(pd.DataFrame(auc_comp).set_index("Modelo"), use_container_width=True)

        fig_auc = px.bar(
            pd.DataFrame(auc_comp).melt(id_vars="Modelo", var_name="Tipo", value_name="AUC"),
            x="Modelo", y="AUC", color="Tipo", barmode="group",
            range_y=[0.9, 1.01], color_discrete_sequence=PALETTE,
            title="AUC comparado",
        )
        fig_auc.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="white", height=380)
        st.plotly_chart(fig_auc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · Digit Gallery & Errors
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    col_gal, col_err = st.columns(2)

    with col_gal:
        if show_digit_gallery:
            st.subheader("🖼️ Galería de Dígitos")
            digit_filter = st.multiselect(
                "Mostrar dígitos:", list(range(10)), default=list(range(10)), key="gal_filter"
            )
            indices = []
            for d in digit_filter:
                idx = np.where(y == d)[0]
                indices.extend(idx[:max(1, n_digits_gallery // len(digit_filter))])
            indices = indices[:n_digits_gallery]

            n_cols = 5
            n_rows = int(np.ceil(len(indices) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
            axes = np.array(axes).flatten()
            for ax in axes:
                ax.axis("off")
            for i, idx in enumerate(indices):
                axes[i].imshow(digits.images[idx], cmap="gray_r", interpolation="nearest")
                axes[i].set_title(f"{y[idx]}", fontsize=9, color="white")
                axes[i].axis("off")
            fig.patch.set_facecolor("#0e1117")
            fig.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col_err:
        st.subheader("❌ Errores de Clasificación")
        err_model = st.selectbox("Modelo:", list(results.keys()), key="err_model")
        y_pred_err = results[err_model]["y_pred"]
        error_idx = np.where(y_pred_err != y_test)[0]

        st.caption(f"Total de errores: **{len(error_idx)}** / {len(y_test)}  "
                   f"({len(error_idx)/len(y_test)*100:.1f}%)")

        show_n = min(n_error_show, len(error_idx))
        if show_n == 0:
            st.success("¡Sin errores en el conjunto de prueba!")
        else:
            sample_err = error_idx[:show_n]
            n_cols_e = 5
            n_rows_e = int(np.ceil(show_n / n_cols_e))
            fig2, axes2 = plt.subplots(n_rows_e, n_cols_e,
                                       figsize=(n_cols_e * 1.6, n_rows_e * 1.8))
            axes2 = np.array(axes2).flatten()
            for ax in axes2:
                ax.axis("off")
            for i, idx in enumerate(sample_err):
                test_img_idx = np.where(np.arange(len(y)) == np.where(
                    np.isin(np.arange(len(y)), np.where(y == y_test[idx])[0])
                )[0][0])[0][0] if False else idx
                # Reconstruct image from test set
                img = X_test[idx].reshape(8, 8)
                axes2[i].imshow(img, cmap="Reds", interpolation="nearest")
                axes2[i].set_title(
                    f"Real:{y_test[idx]}\nPred:{y_pred_err[idx]}",
                    fontsize=8, color="white",
                )
                axes2[i].axis("off")
            fig2.patch.set_facecolor("#0e1117")
            fig2.suptitle(f"Errores — {err_model}", color="white", fontsize=11)
            fig2.tight_layout(pad=0.4)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

    # Confusion by digit: most confused pairs
    st.subheader("🔀 Pares de Dígitos Más Confundidos")
    confused_model = st.selectbox("Modelo:", list(results.keys()), key="conf_pairs")
    cm_c = confusion_matrix(y_test, results[confused_model]["y_pred"])
    np.fill_diagonal(cm_c, 0)
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm_c[i, j] > 0:
                pairs.append({"Real": i, "Predicho": j, "Confusiones": cm_c[i, j]})
    pairs_df = pd.DataFrame(pairs).sort_values("Confusiones", ascending=False).head(15)
    pairs_df["Par"] = pairs_df["Real"].astype(str) + " → " + pairs_df["Predicho"].astype(str)
    fig_pairs = px.bar(
        pairs_df, x="Par", y="Confusiones", color="Confusiones",
        color_continuous_scale="Reds",
        title=f"Top 15 confusiones — {confused_model}",
    )
    fig_pairs.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font_color="white", height=360)
    st.plotly_chart(fig_pairs, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · PCA Visualization
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🌀 Visualización PCA del Espacio de Características")

    @st.cache_data
    def compute_pca(n_comp):
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        pca = PCA(n_components=n_comp, random_state=42)
        return pca.fit_transform(X_sc), pca.explained_variance_ratio_

    pca_dim = pca_components_viz
    X_pca, evr = compute_pca(pca_dim)

    pca_df = pd.DataFrame(
        X_pca[:, :pca_dim],
        columns=[f"PC{i+1}" for i in range(pca_dim)],
    )
    pca_df["Dígito"] = y.astype(str)

    if pca_dim >= 3:
        fig_pca = px.scatter_3d(
            pca_df, x="PC1", y="PC2", z="PC3", color="Dígito",
            color_discrete_sequence=PALETTE,
            title=f"PCA 3D — Varianza explicada: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}, PC3={evr[2]:.2%}",
            opacity=0.7,
        )
        fig_pca.update_traces(marker=dict(size=3))
    else:
        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2", color="Dígito",
            color_discrete_sequence=PALETTE,
            title=f"PCA 2D — Varianza explicada: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}",
            opacity=0.7,
        )
        fig_pca.update_traces(marker=dict(size=4))

    fig_pca.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          font_color="white", height=550)
    st.plotly_chart(fig_pca, use_container_width=True)

    # Scree plot
    st.subheader("📉 Scree Plot — Varianza Explicada")
    _, evr_full = compute_pca(64)
    scree_df = pd.DataFrame({
        "Componente": range(1, 65),
        "Varianza Explicada": evr_full,
        "Varianza Acumulada": np.cumsum(evr_full),
    })
    fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
    fig_scree.add_trace(
        go.Bar(x=scree_df["Componente"], y=scree_df["Varianza Explicada"],
               name="Varianza individual", marker_color="#e94560"),
        secondary_y=False,
    )
    fig_scree.add_trace(
        go.Scatter(x=scree_df["Componente"], y=scree_df["Varianza Acumulada"],
                   name="Varianza acumulada", line=dict(color="#00b4d8", width=2.5)),
        secondary_y=True,
    )
    fig_scree.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font_color="white", height=380,
                            title_text="Varianza explicada por componente PCA")
    fig_scree.update_yaxes(title_text="Varianza individual", secondary_y=False)
    fig_scree.update_yaxes(title_text="Varianza acumulada", secondary_y=True)
    st.plotly_chart(fig_scree, use_container_width=True)

    # Mean digit images per class
    st.subheader("🖼️ Imagen Promedio por Dígito")
    fig_avg, axes_avg = plt.subplots(2, 5, figsize=(10, 4))
    for digit in range(10):
        ax = axes_avg[digit // 5][digit % 5]
        mean_img = X[y == digit].mean(axis=0).reshape(8, 8)
        ax.imshow(mean_img, cmap="hot", interpolation="nearest")
        ax.set_title(f"Dígito {digit}", color="white", fontsize=10)
        ax.axis("off")
    fig_avg.patch.set_facecolor("#0e1117")
    fig_avg.suptitle("Imagen promedio por clase", color="white", fontsize=12)
    fig_avg.tight_layout()
    st.pyplot(fig_avg, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 · Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🌳 Importancia de Píxeles")

    feat_models = [m for m in selected_models
                   if m in ("Random Forest", "Árbol de Decisión")]

    if feat_models:
        for mname in feat_models:
            clf = results[mname]["pipe"].named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                n_feat = clf.feature_importances_.shape[0]
                imp = clf.feature_importances_
                # If PCA was applied the importance is over components, not pixels
                if n_feat == 64:
                    imp_img = imp.reshape(8, 8)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(imp_img, cmap="hot", interpolation="nearest")
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"Mapa de importancia — {mname}", color="white", fontsize=12)
                    ax.axis("off")
                    fig.patch.set_facecolor("#0e1117")
                    st.pyplot(fig, use_container_width=False)
                    plt.close()
                else:
                    st.info(f"{mname}: importancia sobre {n_feat} componentes PCA (no píxeles originales).")
                    fig_imp = px.bar(
                        x=[f"PC{i+1}" for i in range(n_feat)], y=imp,
                        labels={"x": "Componente PCA", "y": "Importancia"},
                        title=f"Importancia por componente PCA — {mname}",
                        color=imp, color_continuous_scale="Oranges",
                    )
                    fig_imp.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                          font_color="white", height=350)
                    st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Selecciona Random Forest o Árbol de Decisión para ver mapas de importancia de píxeles.")

    # Logistic Regression coefficients as pixel maps
    if "Regresión Logística" in selected_models:
        clf_lr = results["Regresión Logística"]["pipe"].named_steps["clf"]
        n_feat_lr = clf_lr.coef_.shape[1]
        if n_feat_lr == 64:
            st.subheader("Mapas de Coeficientes — Regresión Logística")
            fig_lr, axes_lr = plt.subplots(2, 5, figsize=(11, 4.5))
            for digit in range(10):
                ax = axes_lr[digit // 5][digit % 5]
                coef_img = clf_lr.coef_[digit].reshape(8, 8)
                im = ax.imshow(coef_img, cmap="RdBu_r", interpolation="nearest")
                ax.set_title(f"Dígito {digit}", color="white", fontsize=9)
                ax.axis("off")
            fig_lr.patch.set_facecolor("#0e1117")
            fig_lr.suptitle("Coeficientes por clase — Regresión Logística", color="white", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_lr, use_container_width=True)
            plt.close()

    # Pixel variance heatmap
    st.subheader("📊 Varianza de Píxeles por Clase")
    var_digit = st.selectbox("Dígito:", list(range(10)), key="var_d")
    var_img = X[y == var_digit].var(axis=0).reshape(8, 8)
    fig_var, ax_var = plt.subplots(figsize=(4, 4))
    im_v = ax_var.imshow(var_img, cmap="viridis", interpolation="nearest")
    plt.colorbar(im_v, ax=ax_var)
    ax_var.set_title(f"Varianza por píxel — Dígito {var_digit}", color="white")
    ax_var.axis("off")
    fig_var.patch.set_facecolor("#0e1117")
    st.pyplot(fig_var, use_container_width=False)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 · Learning Curves
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("📈 Curvas de Aprendizaje")

    if not show_lc:
        st.info("Activa **'Curvas de aprendizaje'** en el panel lateral para generar las curvas. "
                "Nota: puede tardar unos segundos por modelo.")
    else:
        @st.cache_data
        def compute_learning_curve(model_name, scale, pca, n_comp, seed):
            steps = []
            if scale:
                steps.append(("scaler", StandardScaler()))
            if pca:
                steps.append(("pca", PCA(n_components=n_comp, random_state=seed)))
            steps.append(("clf", MODELS[model_name]))
            pipe = Pipeline(steps)
            train_sizes, train_sc, val_sc = learning_curve(
                pipe, X, y, cv=5, scoring="accuracy",
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=-1, random_state=seed,
            )
            return train_sizes, train_sc, val_sc

        for mname in selected_models:
            with st.spinner(f"Calculando curva para {mname}..."):
                tr_sz, tr_sc, val_sc = compute_learning_curve(
                    mname, scale_data, use_pca,
                    int(n_components) if use_pca else 30,
                    int(random_state),
                )
            tr_mean, tr_std   = tr_sc.mean(axis=1), tr_sc.std(axis=1)
            val_mean, val_std = val_sc.mean(axis=1), val_sc.std(axis=1)

            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=np.concatenate([tr_sz, tr_sz[::-1]]),
                y=np.concatenate([tr_mean + tr_std, (tr_mean - tr_std)[::-1]]),
                fill="toself", fillcolor="rgba(0,180,216,0.15)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))
            fig_lc.add_trace(go.Scatter(
                x=np.concatenate([tr_sz, tr_sz[::-1]]),
                y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                fill="toself", fillcolor="rgba(233,69,96,0.15)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))
            fig_lc.add_trace(go.Scatter(
                x=tr_sz, y=tr_mean, mode="lines+markers",
                name="Train", line=dict(color="#00b4d8", width=2.5),
            ))
            fig_lc.add_trace(go.Scatter(
                x=tr_sz, y=val_mean, mode="lines+markers",
                name="Validación", line=dict(color="#e94560", width=2.5),
            ))
            fig_lc.update_layout(
                title=f"Curva de Aprendizaje — {mname}",
                xaxis_title="Muestras de entrenamiento",
                yaxis_title="Accuracy",
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", height=380, yaxis_range=[0, 1.05],
            )
            st.plotly_chart(fig_lc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 · Dataset Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("🔍 Explorador del Dataset MNIST (scikit-learn)")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("**Distribución de clases**")
        dist_df = pd.DataFrame({"Dígito": y}).value_counts().reset_index()
        dist_df.columns = ["Dígito", "Cantidad"]
        dist_df["Dígito"] = dist_df["Dígito"].astype(str)
        fig_dist = px.bar(
            dist_df.sort_values("Dígito"), x="Dígito", y="Cantidad",
            color="Dígito", color_discrete_sequence=PALETTE,
            title="Muestras por dígito",
        )
        fig_dist.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                               font_color="white", showlegend=False, height=350)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_r:
        st.markdown("**Estadísticas de píxeles**")
        st.dataframe(
            pd.DataFrame(X, columns=[f"px{i}" for i in range(64)])
            .describe().round(3),
            use_container_width=True, height=350,
        )

    st.subheader("Intensidad Promedio Global de Píxeles por Dígito")
    mean_intensities = pd.concat(
        [pd.DataFrame({"Dígito": str(d), "Intensidad promedio": X[y == d].mean(axis=1)})
         for d in range(10)],
        ignore_index=True,
    )
    fig_box = px.box(
        mean_intensities, x="Dígito", y="Intensidad promedio", color="Dígito",
        color_discrete_sequence=PALETTE,
        title="Distribución de intensidad promedio por dígito",
    )
    fig_box.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          font_color="white", height=380, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # Raw data table
    with st.expander("📄 Ver datos crudos (primeras 100 filas)"):
        st.dataframe(df.head(100), use_container_width=True)
