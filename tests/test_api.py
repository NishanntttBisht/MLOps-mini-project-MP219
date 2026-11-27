from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction():
    payload = {
        "students_on_campus": 850,
        "menu_price_rs": 60,
        "temp_c": 28.5,
        "exam_days_left": 10,
        "special_dishes": "no",
        "queue_minutes_yesterday": 20,
        "day_type": "weekday",
        "block": "A-Block",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_plates_sold" in response.json()
    assert isinstance(response.json()["predicted_plates_sold"], int)


def test_invalid_menu_price():
    payload = {
        "students_on_campus": 850,
        "menu_price_rs": 250,
        "temp_c": 28.5,
        "exam_days_left": 10,
        "special_dishes": "no",
        "queue_minutes_yesterday": 20,
        "day_type": "weekday",
        "block": "A-Block",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422