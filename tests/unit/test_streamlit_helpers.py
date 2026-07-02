from __future__ import annotations

import json

import requests

from tech_challenge.presentation.streamlit_app import _api_error_message, _diagnosis_badge


def _response(status_code: int, payload: dict[str, str] | None = None, text: str = "") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode("utf-8") if payload is not None else text.encode("utf-8")
    return response


def test_diagnosis_badge_uses_label_specific_class() -> None:
    assert "malignant" in _diagnosis_badge("Maligno")
    assert "benign" in _diagnosis_badge("Benigno")


def test_api_error_message_formats_known_statuses() -> None:
    assert _api_error_message(None) == "Não foi possível conectar ao serviço."
    assert _api_error_message(_response(503, {"detail": "missing key"})) == "LLM não configurado: missing key"
    assert _api_error_message(_response(502, {"detail": "bad gateway"})) == "Falha no provedor LLM: bad gateway"
    assert _api_error_message(_response(400, text="bad request")) == "Erro da API (400): bad request"
