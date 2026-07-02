from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from tech_challenge.paths import BREAST_CANCER_DATASET
from tech_challenge.presentation.formatting import (
    ag_result_rows,
    chat_answer_text,
    filter_patient_table,
    paginate_patient_table,
    patient_table_rows,
)

API_URL = os.getenv("TECH_CHALLENGE_API_URL", "http://localhost:8000")


def main() -> None:
    st.set_page_config(page_title="Diagnóstico por IA", layout="wide")

    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Selecione a seção",
        ["Análise de Paciente", "Resultados do AG"],
    )

    if "diagnosis" not in st.session_state:
        st.session_state.diagnosis = None
    if "llm_explanation" not in st.session_state:
        st.session_state.llm_explanation = None
    if "llm_error" not in st.session_state:
        st.session_state.llm_error = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_patient_index" not in st.session_state:
        st.session_state.selected_patient_index = None

    if page == "Análise de Paciente":
        _patient_analysis_screen()
    elif page == "Resultados do AG":
        _ag_results_screen()


def _patient_analysis_screen() -> None:
    if st.session_state.selected_patient_index is None or not st.session_state.diagnosis:
        _patient_selection_screen()
        return

    _case_review_screen()


def _patient_selection_screen() -> None:
    st.title("Análise de Paciente")

    try:
        metadata = _patient_metadata()
    except requests.RequestException as exc:
        st.error(f"Não foi possível carregar os pacientes da API: {exc}")
        return

    st.caption(f"{metadata['count']} pacientes disponíveis para análise")
    _patient_selection_table(metadata)


@st.cache_data
def _load_patient_dataset() -> pd.DataFrame:
    return pd.read_csv(BREAST_CANCER_DATASET).drop(columns=["Unnamed: 32"], errors="ignore")


def _patient_selection_table(metadata: dict[str, int]) -> None:
    rows = patient_table_rows(_load_patient_dataset(), metadata["min_index"], metadata["max_index"])

    if "patient_table_page_size" not in st.session_state:
        st.session_state.patient_table_page_size = 25
    if "patient_table_page" not in st.session_state:
        st.session_state.patient_table_page = 1
    if "patient_table_diagnosis_filter" not in st.session_state:
        st.session_state.patient_table_diagnosis_filter = "Todos"

    search_col, diagnosis_col = st.columns([2.5, 1])
    query = search_col.text_input("Buscar por ID", placeholder="Digite o ID do paciente")
    diagnosis_filter = diagnosis_col.segmented_control(
        "Diagnóstico real",
        ["Todos", "Benigno", "Maligno"],
        key="patient_table_diagnosis_filter",
    )

    if st.session_state.get("patient_table_query") != query:
        st.session_state.patient_table_query = query
        st.session_state.patient_table_page = 1
    if st.session_state.get("patient_table_previous_diagnosis_filter") != diagnosis_filter:
        st.session_state.patient_table_previous_diagnosis_filter = diagnosis_filter
        st.session_state.patient_table_page = 1

    filtered_rows = filter_patient_table(rows, query, str(diagnosis_filter))

    page_size = int(st.session_state.patient_table_page_size)
    total_rows = len(filtered_rows)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    st.session_state.patient_table_page = min(st.session_state.patient_table_page, total_pages)

    current_page = int(st.session_state.patient_table_page)
    page_rows = paginate_patient_table(filtered_rows, current_page, page_size)

    if page_rows.empty:
        st.info("Nenhum paciente encontrado para a busca informada.")
        _patient_table_footer(total_pages, total_rows)
        return

    _patient_table_rows(page_rows)
    _patient_table_footer(total_pages, total_rows)


def _patient_table_rows(page_rows: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] .patient-table-header {
            color: rgba(250, 250, 250, 0.68);
            font-size: 0.92rem;
            font-weight: 700;
            padding: 0.35rem 0 0.45rem;
        }
        .diagnosis-badge {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.86rem;
            font-weight: 700;
            line-height: 1;
            padding: 0.38rem 0.62rem;
        }
        .diagnosis-badge.benign {
            background: #dff3e8;
            color: #166534;
        }
        .diagnosis-badge.malignant {
            background: #fde2e2;
            color: #991b1b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    widths = [1.1, 1.25, 1, 1.15, 1.25, 1, 1.35, 0.9]
    headers = [
        "ID",
        "Diagnóstico real",
        "Raio médio",
        "Textura média",
        "Perímetro médio",
        "Área média",
        "Concavidade média",
        "Ação",
    ]
    header_cols = st.columns(widths, vertical_alignment="center")
    for column, header in zip(header_cols, headers):
        column.markdown(f"<div class='patient-table-header'>{header}</div>", unsafe_allow_html=True)

    for _, row in page_rows.iterrows():
        row_cols = st.columns(widths, vertical_alignment="center")
        row_cols[0].write(str(int(row["ID"])))
        row_cols[1].markdown(_diagnosis_badge(str(row["Diagnóstico real"])), unsafe_allow_html=True)
        row_cols[2].write(f"{row['Raio médio']:.3f}")
        row_cols[3].write(f"{row['Textura média']:.3f}")
        row_cols[4].write(f"{row['Perímetro médio']:.3f}")
        row_cols[5].write(f"{row['Área média']:.3f}")
        row_cols[6].write(f"{row['Concavidade média']:.3f}")

        patient_index = int(row["Índice"])
        if row_cols[7].button("Analisar", key=f"analyze_patient_{patient_index}", use_container_width=True):
            if _run_diagnosis(patient_index):
                st.rerun()


def _diagnosis_badge(label: str) -> str:
    class_name = "malignant" if label == "Maligno" else "benign"
    return f"<span class='diagnosis-badge {class_name}'>{label}</span>"


def _patient_table_footer(total_pages: int, total_rows: int) -> None:
    summary_col, page_size_col, prev_col, next_col = st.columns([2.5, 1.2, 0.9, 0.9], vertical_alignment="bottom")
    summary_col.caption(f"{total_rows} pacientes encontrados · Página {st.session_state.patient_table_page} de {total_pages}")
    selected_page_size = page_size_col.selectbox(
        "Linhas",
        [10, 25, 50, 100],
        index=[10, 25, 50, 100].index(int(st.session_state.patient_table_page_size)),
        key="patient_table_page_size_select",
    )
    if int(selected_page_size) != int(st.session_state.patient_table_page_size):
        st.session_state.patient_table_page_size = int(selected_page_size)
        st.session_state.patient_table_page = 1
        st.rerun()
    if prev_col.button("Anterior", disabled=st.session_state.patient_table_page <= 1, use_container_width=True):
        st.session_state.patient_table_page -= 1
        st.rerun()
    if next_col.button("Próxima", disabled=st.session_state.patient_table_page >= total_pages, use_container_width=True):
        st.session_state.patient_table_page += 1
        st.rerun()


def _run_diagnosis(patient_index: int) -> bool:
    with st.spinner("Rodando modelo..."):
        try:
            resp = requests.post(f"{API_URL}/diagnose", json={"patient_index": patient_index}, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            st.error(_api_error_message(exc.response))
            return False
        except requests.RequestException as exc:
            st.error(f"Não foi possível conectar ao serviço: {exc}")
            return False
        st.session_state.diagnosis = resp.json()
        st.session_state.selected_patient_index = patient_index
        st.session_state.llm_explanation = None
        st.session_state.llm_error = None
        st.session_state.chat_history = []
        return True


def _case_review_screen() -> None:
    patient_index = int(st.session_state.selected_patient_index)
    patient = _selected_patient_record(patient_index)
    diagnosis: dict[str, Any] = st.session_state.diagnosis

    header_col, action_col = st.columns([4, 1], vertical_alignment="center")
    header_col.title(f"Análise do Paciente {patient['id']}")
    if action_col.button("Trocar paciente", use_container_width=True):
        _clear_selected_case()
        st.rerun()

    _ensure_llm_explanation(diagnosis)

    result_col, llm_col = st.columns([1.05, 1], gap="large")
    with result_col:
        _diagnosis_panel(diagnosis, patient)
    with llm_col:
        _llm_panel(diagnosis)


def _diagnosis_panel(diagnosis: dict[str, Any], patient: dict[str, Any]) -> None:
    color = "red" if diagnosis["prediction"] == "MALIGNO" else "green"
    st.markdown(f"### Resultado do modelo: :{color}[{diagnosis['prediction']}]")
    st.progress(diagnosis["confidence"], text=f"Confiança: {diagnosis['confidence']:.0%}")

    st.subheader("Principais atributos")
    features = diagnosis["top_features"]
    df = pd.DataFrame(features)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if impact > 0 else "#1f77b4" for impact in df["impact"]]
    ax.barh(df["feature"], df["impact"], color=colors)
    ax.set_xlabel("Impacto no modelo")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Resumo do paciente")
    summary = pd.DataFrame(
        [
            {"Atributo": "ID", "Valor": str(patient["id"])},
            {"Atributo": "Raio médio", "Valor": f"{patient['radius_mean']:.3f}"},
            {"Atributo": "Textura média", "Valor": f"{patient['texture_mean']:.3f}"},
            {"Atributo": "Perímetro médio", "Valor": f"{patient['perimeter_mean']:.3f}"},
            {"Atributo": "Área média", "Valor": f"{patient['area_mean']:.3f}"},
            {"Atributo": "Concavidade média", "Valor": f"{patient['concavity_mean']:.3f}"},
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)
    st.caption(f"Referência do dataset: {patient['diagnosis_label']}")


def _llm_panel(diagnosis: dict[str, Any]) -> None:
    st.subheader("Explicação Médica")

    if st.session_state.llm_error:
        st.error(st.session_state.llm_error)
        if st.button("Regerar Explicação Médica", use_container_width=True):
            _generate_llm_explanation(diagnosis)
            st.rerun()
        return

    if not st.session_state.llm_explanation:
        st.info("A explicação médica será exibida assim que o agente LLM responder.")
        return

    result: dict[str, Any] = st.session_state.llm_explanation
    details = result.get("details", {})

    st.write(result["explanation"])

    if details.get("nivel_confianca"):
        st.markdown(f"**Nível de confiança:** {details['nivel_confianca']}")

    if details.get("recomendacoes"):
        st.markdown("### Recomendações")
        for item in details["recomendacoes"]:
            st.write(f"- {item}")

    if details.get("exames_complementares"):
        st.markdown("### Exames complementares")
        for item in details["exames_complementares"]:
            st.write(f"- {item}")

    st.warning(result["disclaimer"])
    st.button("Regerar Explicação Médica", use_container_width=True, on_click=_regenerate_llm_explanation, args=(diagnosis,))

    with st.expander("Perguntas de acompanhamento"):
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Pergunte sobre a interpretação deste caso")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            context = {"diagnosis": diagnosis, "explanation": result}
            with st.chat_message("assistant"):
                with st.spinner("Consultando agente LLM..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/chat",
                            json={"question": question, "context": context},
                            timeout=60,
                        )
                        resp.raise_for_status()
                    except requests.HTTPError as exc:
                        answer = _api_error_message(exc.response)
                        st.error(answer)
                    except requests.RequestException as exc:
                        answer = f"Não foi possível conectar ao serviço: {exc}"
                        st.error(answer)
                    else:
                        answer = chat_answer_text(resp.json()["answer"])
                        st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


def _ensure_llm_explanation(diagnosis: dict[str, Any]) -> None:
    if st.session_state.llm_explanation or st.session_state.llm_error:
        return

    _generate_llm_explanation(diagnosis)


def _generate_llm_explanation(diagnosis: dict[str, Any]) -> None:
    with st.spinner("Gerando explicação médica..."):
        try:
            resp = requests.post(f"{API_URL}/explain", json=diagnosis, timeout=60)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            st.session_state.llm_error = _api_error_message(exc.response)
            return
        except requests.RequestException as exc:
            st.session_state.llm_error = f"Não foi possível conectar ao serviço: {exc}"
            return
        st.session_state.llm_explanation = resp.json()
        st.session_state.llm_error = None


def _regenerate_llm_explanation(diagnosis: dict[str, Any]) -> None:
    st.session_state.llm_explanation = None
    st.session_state.llm_error = None
    st.session_state.chat_history = []
    _generate_llm_explanation(diagnosis)


def _clear_selected_case() -> None:
    st.session_state.selected_patient_index = None
    st.session_state.diagnosis = None
    st.session_state.llm_explanation = None
    st.session_state.llm_error = None
    st.session_state.chat_history = []


def _selected_patient_record(patient_index: int) -> dict[str, Any]:
    dataset = _load_patient_dataset()
    record = dataset.iloc[patient_index].to_dict()
    record["diagnosis_label"] = "Maligno" if record["diagnosis"] == "M" else "Benigno"
    return record


def _ag_results_screen() -> None:
    st.title("Resultados do Algoritmo Genético")

    try:
        resp = requests.get(f"{API_URL}/ag-results", timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        st.error(_api_error_message(exc.response))
        return
    except requests.RequestException as exc:
        st.error(f"Não foi possível conectar ao serviço: {exc}")
        return
    data = resp.json()

    experiments = data["experiments"]
    baseline = data["baseline"]
    best = data["best_config"]

    st.subheader("Comparativo: Baseline vs Experimentos AG")
    st.dataframe(pd.DataFrame(ag_result_rows(experiments, baseline)), width="stretch")

    st.subheader("Convergência por experimento")
    fig, ax = plt.subplots()
    for experiment in experiments:
        ax.plot(experiment["convergence"], label=experiment["name"])
    ax.axhline(baseline["metrics"]["f1"], color="gray", linestyle="--", label="Baseline F1")
    ax.set_xlabel("Geração (amostrada)")
    ax.set_ylabel("Fitness")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Melhor configuração encontrada")
    st.write(f"Experimento: {data['best_experiment']}")
    st.json(best)

    if data.get("best_model"):
        st.subheader("Resumo do melhor modelo")
        st.json(data["best_model"])


def _patient_metadata() -> dict[str, int]:
    resp = requests.get(f"{API_URL}/patients/metadata", timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        "count": int(data["count"]),
        "min_index": int(data["min_index"]),
        "max_index": int(data["max_index"]),
    }


def _api_error_message(response: requests.Response | None) -> str:
    if response is None:
        return "Não foi possível conectar ao serviço."

    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = response.text

    if response.status_code == 503:
        return f"LLM não configurado: {detail}"
    if response.status_code == 502:
        return f"Falha no provedor LLM: {detail}"
    return f"Erro da API ({response.status_code}): {detail}"


if __name__ == "__main__":
    main()
