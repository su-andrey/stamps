from detection_models.yolo_stamp.utils import *
import albumentations as A
from huggingface_hub import hf_hub_download
from albumentations.pytorch import ToTensorV2
from detection_models.yolo_stamp.utils import *
import torchvision

class YoloStampPipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2(p=1.0),
        ])
        self.anchors = torch.tensor([
            [10, 13],  # Малый anchor
            [16, 30],  # Средний anchor
            [33, 23]  # Большой anchor
        ], device=self.device) / 448  # Нормализуем к размеру сети

        # Количество anchors на ячейку
        self.num_anchors = len(self.anchors)

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
        # 1. Получение размеров изображения
        if isinstance(image, Image.Image):
            orig_size = torch.tensor(image.size, device=self.device)  # (width, height)
        else:  # numpy.ndarray
            orig_size = torch.tensor([image.shape[1], image.shape[0]], device=self.device)

        # 2. Преобразование изображения
        transformed = self.transform(image=image)
        img_tensor = transformed["image"].unsqueeze(0).to(self.device)  # [1, C, H, W]

        # 3. Получение выхода модели
        with torch.no_grad():
            output = self.model(img_tensor)[0]  # torch.Size([15, 12, 3, 5])

        # 4. Обработка выхода YOLO
        try:
            # Преобразуем формат вывода
            boxes, scores = self._decode_yolo_output(output)

            if boxes.shape[0] == 0:
                return torch.zeros((0, 4), device=self.device)

            # Масштабирование к исходному размеру
            boxes = self._rescale_boxes(boxes, img_tensor.shape[2:], orig_size)

            # NMS
            keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.45)
            return boxes[keep]

        except Exception as e:
            print(f"Detection error: {e}")
            return torch.zeros((0, 4), device=self.device)

    def _decode_yolo_output(self, output):
        """Преобразует выход YOLO в детекции"""
        # Создаем координатную сетку
        grid_h, grid_w = output.shape[:2]
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w))
        grid = torch.stack((grid_x, grid_y), dim=-1).float().to(self.device)  # [15,12,2]

        # Преобразование координат
        xy = (output[..., 0:2].sigmoid() + grid.unsqueeze(2)) / torch.tensor([grid_w, grid_h], device=self.device)
        wh = torch.exp(output[..., 2:4]) * self.anchors
        boxes = torch.cat([xy, wh], dim=-1)  # [15,12,3,4]

        # Преобразуем xywh -> xyxy
        boxes_xyxy = torch.empty_like(boxes)
        boxes_xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
        boxes_xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
        boxes_xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
        boxes_xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2

        # Получаем confidence
        conf = output[..., 4].sigmoid()  # [15,12,3]

        # Преобразуем в плоский формат
        boxes_flat = boxes_xyxy.reshape(-1, 4)  # [15*12*3, 4]
        conf_flat = conf.reshape(-1)  # [15*12*3]

        # Фильтрация по confidence
        mask = conf_flat > 0.25
        return boxes_flat[mask], conf_flat[mask]

    def _rescale_boxes(self, boxes, model_shape, orig_shape):
        """Масштабирует боксы к исходному размеру изображения"""
        # model_shape: [H,W] выходного тензора модели
        # orig_shape: [W,H] исходного изображения

        # Коэффициенты масштабирования (width, height)
        scale = torch.tensor([
            orig_shape[0] / model_shape[1],  # width scale
            orig_shape[1] / model_shape[0],  # height scale
            orig_shape[0] / model_shape[1],
            orig_shape[1] / model_shape[0]
        ], device=boxes.device)

        return boxes * scale