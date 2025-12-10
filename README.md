# Iris Predictor (FastAPI + Streamlit)

## Requisitos
- Python 3.10+ (ou 3.8+)
- VS Code (opcional)
- pip

## Instalação (recomendo usar virtualenv)
1. Abra o terminal no diretório do projeto `penguim-predictor/`.

2. Criar e ativar ambiente virtual (Windows):
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`

   (Linux / macOS):
   - `python -m venv .venv`
   - `source .venv/bin/activate`

3. Instalar dependências:
   - `pip install -r requirements.txt`

## Preparar o modelo
Se você já tem `penguim_classifier_tree_model.pkl`, coloque-o em `app_backend/model/penguim_classifier_tree_model.pkl`.
Se NÃO tiver o modelo, rode:
   - `python scripts/train_and_save_model.py`
Isso vai treinar um RandomForest no dataset Iris e salvar em `app_backend/model/penguim_classifier_tree_model.pkl`.

## Rodar a API (FastAPI)
No terminal com ambiente ativado:
   - `uvicorn app_backend.api:app --reload --port 8000`
A API ficará disponível em `http://localhost:8000/`.

Testar endpoint:
   - `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"culmen_length_mm\":39.1, \"culmen_depth_mm\":18.7, \"flipper_length_mm\":181, \"body_mass_g\":3750}"`

## Rodar a interface (Streamlit)
Em outro terminal (também com o venv ativado):
   - `streamlit run app_frontend/streamlit_app.py`

A interface abrirá em `http://localhost:8501/` (padrão).

## Ordem recomendada
1. Iniciar API (uvicorn)
2. Abrir Streamlit (streamlit run ...)

## Observações
- A API responde com: `predicted_class`, `confidence` (0..1) e `probabilities` (lista).
- Em produção, configure CORS apropriado e cole o modelo com segurança.
