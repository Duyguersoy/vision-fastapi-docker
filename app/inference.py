import io
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

# Bu değerler eğitimde kullandıklarınla aynı olmalı.
# Eğer config.py'den biliyorsan aynen koy.
INPUT_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_transform = T.Compose([
    T.Resize(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

class Predictor:
    def __init__(self, model_path: str, le_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # PyTorch 2.6+ uyumu: weights_only=False
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

        self.le = pickle.loads(Path(le_path).read_bytes())

    @torch.no_grad()
    def predict_image_bytes(self, image_bytes: bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = _transform(img).unsqueeze(0).to(self.device)

        (bbPred, yLogits) = self.model(x)  # bbPred kullanılmasa da model böyle dönüyor
        probs = F.softmax(yLogits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())

        label = str(self.le.inverse_transform([idx])[0])
        prob = float(probs[idx])

        return label, prob