import sys
import mlflow
import mlflow.sklearn
import numpy as np

def load_model(run_id: str, artifact_path: str = "model"):
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python src/predict.py <run_id> <temperatura>")
        sys.exit(1)

    run_id = sys.argv[1]
    temperature = float(sys.argv[2])

    model = load_model(run_id)
    prediction = model.predict(np.array([[temperature]]))[0]

    print(f"Temperatura: {temperature}°C")
    print(f"Vendas previstas de sorvete: {prediction:.2f} unidades"
