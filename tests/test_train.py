import os
import json
import numpy as np
import pandas as pd
from src.train import train


FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]


def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Adult để sử dụng trong test.

    pytest cung cấp `tmp_path` là một thư mục tạm thời, tự động được xóa sau khi test kết thúc.
    """
    rng = np.random.default_rng(0)
    n = 200
    # TODO 2.10.1: Tạo mảng X có kích thước (n, len(FEATURE_NAMES)) với giá trị ngẫu nhiên [0, 1)
    X = rng.random((n, len(FEATURE_NAMES)))
    # TODO 2.10.2: Tạo mảng y có n phần tử, mỗi phần tử là số nguyên ngẫu nhiên trong [0, 2)
    #   Chú ý: bài toán này chỉ có HAI lớp, nên cận trên là 2 chứ không phải 3.
    y = rng.integers(0, 2, size=n)
    # TODO 2.10.3: Tạo DataFrame từ X với các cột là FEATURE_NAMES, thêm cột "target" = y
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    # TODO 2.10.4: Lưu 160 dòng đầu vào file train.csv và 40 dòng cuối vào file holdout.csv tại tmp_path
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "holdout.csv"
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    # TODO 2.10.5: Trả về (train_path, eval_path)
    return str(train_path), str(eval_path)


def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về một số thực trong khoảng [0, 1]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    # TODO 2.10.6: Gọi hàm train() với siêu tham số nhỏ
    #   (n_estimators=10, learning_rate=0.1, max_depth=2)
    f1 = train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    # TODO 2.10.7: assert kết quả trả về là float và nằm trong [0.0, 1.0]
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_report_file_created(tmp_path):
    """Kiểm tra file outputs/report.json được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    # TODO 2.10.8: assert file "outputs/report.json" tồn tại
    assert os.path.exists("outputs/report.json")
    # TODO 2.10.9: Đọc file report.json và assert nó chứa cả "f1_score" và "accuracy"
    with open("outputs/report.json", "r") as f:
        report = json.load(f)
    assert "f1_score" in report
    assert "accuracy" in report


def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.joblib được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )
    # TODO 2.10.10: assert file "models/model.joblib" tồn tại
    assert os.path.exists("models/model.joblib")
