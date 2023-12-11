from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Привет"}


def test_predict_positive():
    response = client.post("/predict/",
                           json={"text": "Мне нравится запах цветов"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_predict_negative():
    response = client.post("/predict/",
                           json={"text": "Я не люблю пить березовый сок"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'