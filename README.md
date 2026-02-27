# vision-fastapi-docker
Face Mask Detection model served with FastAPI + Docker (Colab-trained PyTorch model)
# predict endpoint’i test edilerek çıktı olarak label ve probability alanları JSON biçiminde alınmıştır.
{"label":"without_mask","probability":0.4951576888561249}
# vision-fastapi-docker

Colab’de eğitilen **PyTorch Face Mask Detection** modelini **FastAPI** ile servis eden ve **Docker** ile çalıştırılabilen API.

## Özellikler
- FastAPI + Swagger UI (`/docs`)
- Model yükleme: `models/detector.pth`
- LabelEncoder: `models/le.pickle`
- Endpoint’ler:
  - `GET /health` -> servis ayakta mı?
  - `POST /predict` -> görsel yükle, sınıf + olasılık dön

## Kurulum (Docker ile)
> Docker Desktop açık olmalı.

```bash
docker compose up --build