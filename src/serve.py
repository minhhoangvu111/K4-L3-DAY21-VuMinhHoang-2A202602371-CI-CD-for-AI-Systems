from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Cloud SDK: google-cloud-storage (GCP) | boto3 (AWS) | azure-storage-blob (Azure)
from google.cloud import storage   # thay bằng SDK của provider đã chọn
import joblib
import os

app = FastAPI()

# Đọc tên bucket từ biến môi trường (được đặt trong systemd service)
ARTIFACT_BUCKET = os.environ["ARTIFACT_BUCKET"]
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser("~/models/model.joblib")


def download_model():
    """Tải file model.joblib từ cloud storage về máy khi server khởi động."""
    # TODO 2.6.1: Tạo một storage.Client()
    client = storage.Client()
    # TODO 2.6.2: Lấy bucket bằng client.bucket(ARTIFACT_BUCKET)
    bucket = client.bucket(ARTIFACT_BUCKET)
    # TODO 2.6.3: Lấy blob bằng bucket.blob(MODEL_KEY)
    blob = bucket.blob(MODEL_KEY)
    # TODO 2.6.4: Tải file xuống bằng blob.download_to_filename(MODEL_PATH)
    blob.download_to_filename(MODEL_PATH)
    # TODO 2.6.5: In thông báo thành công
    print(f"Model downloaded from gs://{ARTIFACT_BUCKET}/{MODEL_KEY} to {MODEL_PATH}")


# Gọi hàm này khi module được import (chạy khi server khởi động)
download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz():
    """Endpoint kiểm tra sức khỏe server. GitHub Actions dùng endpoint này để xác nhận triển khai thành công."""
    # TODO 2.6.6: Trả về dict {"status": "ok"}
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest):
    """
    Endpoint suy luận.

    Đầu vào: JSON {"features": [f1, f2, ..., f10]}
    Đầu ra:  JSON {"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}
    """
    # TODO 2.6.7: Kiểm tra len(req.features) == 10.
    #   Nếu không, raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")

    # TODO 2.6.8: Gọi model.predict([req.features]) để lấy kết quả dự đoán.
    prediction = model.predict([req.features])[0]

    # TODO 2.6.9: Trả về dict chứa "prediction" (int) và "label" (string).
    #   Nhãn: 0 -> "thu_nhap_thap", 1 -> "thu_nhap_cao"
    label = "thu_nhap_thap" if prediction == 0 else "thu_nhap_cao"
    return {"prediction": int(prediction), "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
