# 9IADT — Tech Challenge

Sistema de suporte ao diagnóstico do câncer de mama usando Machine Learning, Algoritmos Genéticos e LLMs.

## Requisitos

| Ferramenta | Versão mínima | Instalação |
| :--- | :--- | :--- |
| Python | 3.11 | [python.org](https://www.python.org/downloads/) |
| uv | 0.4+ | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

## Configuração do ambiente

```bash
# 1. Clone o repositório
git clone <repo-url>
cd 9IADT-tech-challenge

# 2. Copie o arquivo de variáveis de ambiente
cp .env.example .env
```

Edite o `.env` e preencha as chaves de API:

| Variável | Obrigatória | Descrição |
| :--- | :--- | :--- |
| `GOOGLE_API_KEY` | Sim | Chave da API Gemini (Google AI Studio) |
| `GROQ_API_KEY` | Opcional | Chave da API Groq (provider alternativo) |

```bash
# 3. Instale as dependências via uv
uv sync --frozen
```

## Como rodar

### API + Interface juntos (recomendado)

Requer Node.js instalado.

```bash
npm install
npm run dev
```

Isso sobe a API (`http://localhost:8000`) e o Streamlit (`http://localhost:8501`) em paralelo.

### Separadamente

#### API (FastAPI)

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Acesse a documentação em `http://localhost:8000/docs`.

#### Interface (Streamlit)

Requer a API rodando em paralelo.

```bash
uv run streamlit run app.py
```

Acesse `http://localhost:8501`.

### Notebooks

```bash
uv run jupyter notebook
```

| Notebook | Descrição |
| :--- | :--- |
| `tech_challenge_fase1.ipynb` | Análise exploratória, baselines KNN/RF e visão computacional |
| `tech_challenge_fase2.ipynb` | Algoritmos Genéticos, integração LLM e pipeline completo |

## Devcontainer (VS Code)

Abra o repositório no VS Code e aceite a sugestão de **Reopen in Container**. O ambiente é configurado automaticamente via `.devcontainer/`.

## Estrutura do projeto

```text
├── api/            # FastAPI — endpoints REST
├── llm/            # Agente LLM (Gemini) e pipeline de inferência
├── src/            # Utilitários compartilhados
├── data/           # Dataset Wisconsin Breast Cancer
├── model/          # Modelo treinado (.pkl)
├── results/        # Resultados do pipeline
└── app.py          # Interface Streamlit
```
