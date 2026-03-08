# Gelato Mágico – Previsão de Vendas de Sorvete com Machine Learning

Este projeto utiliza um modelo de regressão para prever a quantidade de sorvetes vendidos com base na temperatura do dia, ajudando a sorveteria **Gelato Mágico** a planejar melhor sua produção.

## Objetivos

- Treinar um modelo de Machine Learning para prever vendas a partir da temperatura.
- Registrar e gerenciar experimentos e modelos com **MLflow**.
- Estruturar um pipeline reprodutível de treino e teste.
- Preparar o modelo para uso em tempo real (API em ambiente de cloud).

## Estrutura do Projeto

- `data/ice_cream_sales.csv`: dados de temperatura x vendas.
- `inputs/frases.txt`: frases usadas para refletir sobre o negócio e possíveis features.
- `src/train.py`: script de treino com MLflow.
- `src/predict.py`: script de predição.
- `notebooks/`: exploração inicial dos dados.
- `mlruns/`: diretório criado pelo MLflow com os experimentos.

## Como Rodar

```bash
pip install -r requirements.txt

# Treino do modelo
python src/train.py

# Abrir interface do MLflow
mlflow ui
