import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pipelines.detection.yolo_stamp import YoloStampPipeline
import numpy as np
app = FastAPI()


@app.post("/detect/")
async def get_boxes(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pipe = YoloStampPipeline.from_pretrained("stamps-labs/yolo-stamp")
    results = pipe(image)
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
