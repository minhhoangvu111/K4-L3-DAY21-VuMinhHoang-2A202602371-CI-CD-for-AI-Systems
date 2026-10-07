import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho GradientBoostingClassifier
        data_path: đường dẫn đến file dữ liệu huấn luyện
        eval_path: đường dẫn đến file dữ liệu đánh giá

    Trả về:
        f1 (float): điểm F1 của lớp dương trên tập holdout
    """

    # TODO 1.6.1: Đọc dữ liệu huấn luyện từ data_path vào DataFrame df_train
    #   và dữ liệu đánh giá từ eval_path vào DataFrame df_eval.
    # Gợi ý: sử dụng pd.read_csv(...)
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 1.6.2: Tách đặc trưng và nhãn.
    #   X_train, y_train từ df_train (bỏ cột "target")
    #   X_eval, y_eval từ df_eval (bỏ cột "target")
    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    X_eval = df_eval.drop("target", axis=1)
    y_eval = df_eval["target"]

    # TODO 1.6.3: Bắt đầu một MLflow run bằng `with mlflow.start_run():`
    #   Bên trong block này, thực hiện các bước sau:
    with mlflow.start_run():
        # TODO 1.6.4: Ghi nhận các siêu tham số vào MLflow.
        # Gợi ý: mlflow.log_params(params)
        mlflow.log_params(params)

        # TODO 1.6.5: Khởi tạo và huấn luyện mô hình GradientBoostingClassifier.
        # Gợi ý: model = GradientBoostingClassifier(**params, random_state=42)
        #          model.fit(X_train, y_train)
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 1.6.6: Tính f1_score và accuracy trên tập holdout.
        # Gợi ý: preds = model.predict(X_eval)
        #          f1  = f1_score(y_eval, preds)        <- lớp dương, KHÔNG dùng average
        #          acc = accuracy_score(y_eval, preds)
        preds = model.predict(X_eval)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)

        # TODO 1.6.7: Ghi nhận các chỉ số vào MLflow.
        # Gợi ý: mlflow.log_metric("f1_score", f1)
        #          mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # TODO 1.6.8: Log mô hình vào MLflow artifact.
        # Gợi ý: mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(model, "model")

        # TODO 1.6.9: In kết quả ra màn hình.
        # Gợi ý: print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # TODO 1.6.10: Lưu metrics ra file outputs/report.json.
        # File này sẽ được đọc bởi GitHub Actions ở Bước 2.
        # Gợi ý:
        #       os.makedirs("outputs", exist_ok=True)
        #       with open("outputs/report.json", "w") as f:
        #           json.dump({"f1_score": f1, "accuracy": acc}, f)
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.json", "w") as f:
            json.dump({"f1_score": f1, "accuracy": acc}, f)

        # TODO 1.6.11: Lưu mô hình ra file models/model.joblib.
        # File này sẽ được upload lên cloud storage ở Bước 2.
        # Gợi ý:
        #       os.makedirs("models", exist_ok=True)
        #       joblib.dump(model, "models/model.joblib")
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 1.6.12: Trả về f1 để các hàm gọi train() có thể đọc kết quả.
    return f1


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
