import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Diagnóstico por IA", layout="wide")

# --- Sidebar navigation ---
st.sidebar.title("Navegação")
page = st.sidebar.radio(
    "Selecione a tela",
    ["Tela 1 — Diagnóstico", "Tela 2 — Explicação LLM", "Tela 3 — Resultados do AG"],
)

# --- State shared between screens ---
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None


# ─────────────────────────────────────────────
# Tela 1: Diagnóstico
# ─────────────────────────────────────────────
if page == "Tela 1 — Diagnóstico":
    st.title("Diagnóstico — RandomForest Otimizado")

    patient_index = st.selectbox("Selecionar paciente (índice do test set)", range(0, 114))

    if st.button("Executar Diagnóstico"):
        with st.spinner("Rodando modelo..."):
            resp = requests.post(f"{API_URL}/diagnose", json={"patient_index": patient_index})
            resp.raise_for_status()
            data = resp.json()
            st.session_state.diagnosis = data

    if st.session_state.diagnosis:
        d = st.session_state.diagnosis
        color = "red" if d["prediction"] == "MALIGNO" else "green"
        st.markdown(f"### Resultado: :{color}[{d['prediction']}]")
        st.progress(d["confidence"], text=f"Confiança: {d['confidence']:.0%}")

        st.subheader("Top features (SHAP)")
        features = d["top_features"]
        df = pd.DataFrame(features)
        fig, ax = plt.subplots()
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["impact"]]
        ax.barh(df["feature"], df["impact"], color=colors)
        ax.set_xlabel("Impacto SHAP")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)


# ─────────────────────────────────────────────
# Tela 2: Explicação LLM
# ─────────────────────────────────────────────
elif page == "Tela 2 — Explicação LLM":
    st.title("Explicação Médica — Agente LLM")

    if not st.session_state.diagnosis:
        st.info("Execute um diagnóstico na Tela 1 primeiro.")
    else:
        d = st.session_state.diagnosis
        st.markdown(f"**Paciente selecionado** — Predição: `{d['prediction']}` | Confiança: `{d['confidence']:.0%}`")

        if st.button("Gerar Explicação Médica"):
            with st.spinner("Consultando agente LLM..."):
                resp = requests.post(f"{API_URL}/explain", json=d)
                resp.raise_for_status()
                result = resp.json()

            st.markdown("### Explicação")
            st.write(result["explanation"])
            st.warning(result["disclaimer"])

            st.button("Copiar para relatório", on_click=lambda: st.toast("Texto copiado!"))


# ─────────────────────────────────────────────
# Tela 3: Resultados do AG
# ─────────────────────────────────────────────
elif page == "Tela 3 — Resultados do AG":
    st.title("Resultados do Algoritmo Genético")

    resp = requests.get(f"{API_URL}/ag-results")
    resp.raise_for_status()
    data = resp.json()

    experiments = data["experiments"]
    baseline = data["baseline_f1"]
    best = data["best_config"]

    st.subheader("Comparativo: Baseline vs Experimentos AG")
    rows = [{"Modelo": "RF Baseline", "F1": baseline, "Pop": "—", "Gen": "—", "Mut": "—"}]
    for e in experiments:
        rows.append({
            "Modelo": e["name"],
            "F1": e["best_f1"],
            "Pop": e["population"],
            "Gen": e["generations"],
            "Mut": f"{e['mutation_rate']:.0%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Convergência por experimento")
    fig, ax = plt.subplots()
    for e in experiments:
        ax.plot(e["convergence"], label=e["name"])
    ax.axhline(baseline, color="gray", linestyle="--", label="Baseline")
    ax.set_xlabel("Geração (amostrada)")
    ax.set_ylabel("F1-score")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Melhor configuração encontrada")
    st.json(best)
