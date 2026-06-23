# 9IADT — Tech Challenge

Sistema de suporte ao diagnóstico do câncer de mama. Notebook único: `tech_challenge_fase1.ipynb`.

## Como rodar

### Opção 1 — Docker

```bash
docker build -t tech-challenge-fase1 .
docker run --rm -p 8888:8888 -v "$(pwd):/app" tech-challenge-fase1
```

Acesse `http://localhost:8888` e abra `tech_challenge_fase1.ipynb`.

### Opção 2 — Python local

```bash
python -m venv .venv
source .venv/Scripts/activate     # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Executar o notebook end-to-end (gerar outputs)

```bash
docker run --rm -v "$(pwd):/app" -w /app tech-challenge-fase1 \
    jupyter nbconvert --to notebook --execute tech_challenge_fase1.ipynb \
    --inplace --ExecutePreprocessor.timeout=1200
```

Tempo: ~5–10 min em CPU.
