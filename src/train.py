import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_data(path: str):
    df = pd.read_csv(path)
    X = df[["temperature"]]
    y = df["sales"]
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2

if __name__ == "__main__":
    mlflow.set_experiment("gelato-magico-ice-cream-sales")

    with mlflow.start_run():
        X, y = load_data("data/ice_cream_sales.csv")
        model, rmse, r2 = train_model(X, y)

        # log de métricas
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # log de parâmetros (se tivesse mais)
        mlflow.log_param("model_type", "LinearRegression")

        # log do modelo
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")
