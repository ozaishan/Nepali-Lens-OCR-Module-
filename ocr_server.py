from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
import io
import numpy as np

app = FastAPI()

# Initialize EasyOCR reader with Nepali and English (add other langs if needed)
reader = easyocr.Reader(['ne'], gpu=True)  # set gpu=True if you have CUDA

@app.post("/ocr/")
async def perform_ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Convert PIL Image to numpy array
        image_np = np.array(image)

        results = reader.readtext(image_np, paragraph=True)
        print("Raw OCR Results:", results)

        extracted_texts = [text for (_, text) in results]

        return JSONResponse(content={"text": extracted_texts})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
