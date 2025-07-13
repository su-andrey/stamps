import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from pipelines.detection.yolo_stamp import YoloStampPipeline

app = FastAPI()


@app.post("/detect/")
async def get_boxes(file: UploadFile = File(...),
                    conf_value: float = Form(0.25, description="Confidence value from 0 to 1", gt=0, le=1)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pipe = YoloStampPipeline.from_pretrained("stamps-labs/yolo-stamp")
    results = pipe(image, conf_value)
    stamps = []
    for box in results:
        x1, y1, x2, y2 = box.tolist()  # Конвертируем tensor в list
        stamps.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    return JSONResponse(content={"detections": stamps})
