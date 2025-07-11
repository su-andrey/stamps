import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
import torchvision
from detection_models.yolo_stamp.utils import *


class YoloStampPipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2(p=1.0),
        ])

    @classmethod
    def from_pretrained(cls, model_path_hf: str = None, filename_hf: str = "weights.pt", local_model_path: str = None):
        yolo = cls()
        if model_path_hf is not None and filename_hf is not None:
            yolo.model = torch.load(hf_hub_download(model_path_hf, filename=filename_hf), map_location="cpu")
            yolo.model.to(yolo.device)
            yolo.model.eval()
        elif local_model_path is not None:
            yolo.model = torch.load(local_model_path, map_location="cpu")
            yolo.model.to(yolo.device)
            yolo.model.eval()
        return yolo

    def __call__(self, image) -> torch.Tensor:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        coef = torch.tensor([image.size[0] / 448, image.size[1] / 448], device=self.device).repeat(2)
        # Конвертируем и рассчитываем коэффиценты пропорциональности
        image = image.resize((448, 448))  # Меняем размер под стандарт модели
        image_tensor = self.transform(image=np.array(image))[
            "image"]  # albumentations трансформация, тут перегоняем изображение в numpy, иначе не сработает
        output = self.model(image_tensor.unsqueeze(0).to(self.device))[
            0].detach().cpu()  # отрываем от вычислений, дабы не занимать мощности и переносим обработку на CPU
        xywh = output[..., :4].reshape(-1, 4)  # Собираем вывод модели
        conf = torch.sigmoid(output[..., 4]).reshape(-1)
        mask = conf > 0.25  # Анализируем по уверенности, дабы отсеять ложные срабатывания (параметр 0.25 снижает кол-во зон с 147 до 6)
        xywh, conf = xywh[mask], conf[mask]
        boxes = xywh2xyxy(xywh)  # Преобразуем в формат (x,y),(x,y)
        return boxes * coef.to(boxes.device)  # Важно не забыть домножить на коэфицент пропорциональности
