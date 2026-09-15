import os
import io
import base64
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import (
    CLASS_NAMES,
    API_HOST,
    API_PORT,
    NDVI_THRESHOLDS,
    NDVI_COLORS,
    CHANGE_THRESHOLD,
    PROJECT_ROOT
)

# Try importing the model; handle gracefully if not available
MODEL_AVAILABLE = False
_model = None
_device = None
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from src.model.classifier import get_model
    from src.config import BEST_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if BEST_MODEL_PATH.exists():
        _model = get_model(pretrained=False, freeze=False)
        _model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=_device))
        _model = _model.to(_device)
        _model.eval()
        MODEL_AVAILABLE = True
except Exception:
    pass

_classify_transform = None
if MODEL_AVAILABLE:
    _classify_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

app = FastAPI(title="Satellite Imagery Analyzer API")

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/classes")
async def get_classes():
    return CLASS_NAMES

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    if not MODEL_AVAILABLE or _model is None:
        raise HTTPException(status_code=503, detail="Model not available. Train the model first.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        input_tensor = _classify_transform(image).unsqueeze(0).to(_device)
        with torch.no_grad():
            output = _model(input_tensor)
            probs = F.softmax(output[0], dim=0)
            confidence, class_idx = torch.max(probs, dim=0)
        
        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}
        
        return {
            "class": CLASS_NAMES[class_idx.item()],
            "confidence": round(float(confidence), 4),
            "probabilities": prob_dict,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _calculate_ndvi(nir_arr, red_arr):
    # Avoid division by zero
    denominator = (nir_arr + red_arr)
    denominator[denominator == 0] = 1e-6
    ndvi = (nir_arr - red_arr) / denominator
    return np.clip(ndvi, -1.0, 1.0)

def _generate_heatmap_image(ndvi_arr):
    height, width = ndvi_arr.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    stats = {k: 0 for k in NDVI_THRESHOLDS.keys()}
    total_pixels = height * width
    
    # Simple color mapping based on thresholds
    for category, (low, high) in NDVI_THRESHOLDS.items():
        mask = (ndvi_arr >= low) & (ndvi_arr <= high)
        stats[category] = int(np.sum(mask))
        
        hex_color = NDVI_COLORS[category].lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        colored[mask] = rgb
        
    for k in stats:
        stats[k] = round((stats[k] / total_pixels) * 100, 2)
        
    img = Image.fromarray(colored)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), stats

@app.post("/api/ndvi")
async def calculate_ndvi(
    nir: UploadFile = File(None),
    red: UploadFile = File(None),
    demo: bool = Query(False)
):
    try:
        if demo:
            # Generate dummy demo data
            nir_arr = np.random.uniform(0.1, 0.9, (128, 128))
            red_arr = np.random.uniform(0.1, 0.4, (128, 128))
        else:
            if not nir or not red:
                raise HTTPException(status_code=400, detail="Must provide both 'nir' and 'red' image files or use ?demo=true")
            nir_img = Image.open(io.BytesIO(await nir.read())).convert("L")
            red_img = Image.open(io.BytesIO(await red.read())).convert("L")
            # Resize red to match nir if they differ
            if nir_img.size != red_img.size:
                red_img = red_img.resize(nir_img.size)
                
            nir_arr = np.array(nir_img) / 255.0
            red_arr = np.array(red_img) / 255.0
            
        ndvi_arr = _calculate_ndvi(nir_arr, red_arr)
        heatmap_b64, stats = _generate_heatmap_image(ndvi_arr)
        
        return {
            "heatmap": heatmap_b64,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/change-detect")
async def detect_changes(
    before: UploadFile = File(None),
    after: UploadFile = File(None),
    demo: bool = Query(False)
):
    try:
        if demo:
            before_arr = np.random.uniform(0.1, 0.9, (128, 128))
            after_arr = np.random.uniform(0.1, 0.9, (128, 128))
        else:
            if not before or not after:
                raise HTTPException(status_code=400, detail="Must provide both 'before' and 'after' image files or use ?demo=true")
            before_img = Image.open(io.BytesIO(await before.read())).convert("L")
            after_img = Image.open(io.BytesIO(await after.read())).convert("L")
            if before_img.size != after_img.size:
                after_img = after_img.resize(before_img.size)
                
            before_arr = np.array(before_img) / 255.0
            after_arr = np.array(after_img) / 255.0
            
        diff = after_arr - before_arr
        
        # Positive change (greener), Negative change (less green), No significant change
        pos_mask = diff >= CHANGE_THRESHOLD
        neg_mask = diff <= -CHANGE_THRESHOLD
        
        height, width = diff.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background: grayscale of 'after'
        bg = (after_arr * 255).astype(np.uint8)
        colored[:, :, 0] = bg
        colored[:, :, 1] = bg
        colored[:, :, 2] = bg
        
        # Positive = Green
        colored[pos_mask] = [45, 204, 113]
        # Negative = Red
        colored[neg_mask] = [231, 76, 60]
        
        img = Image.fromarray(colored)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        
        total_pixels = height * width
        stats = {
            "Positive Change": round((np.sum(pos_mask) / total_pixels) * 100, 2),
            "Negative Change": round((np.sum(neg_mask) / total_pixels) * 100, 2),
            "Unchanged": round((np.sum((~pos_mask) & (~neg_mask)) / total_pixels) * 100, 2)
        }
        
        return {
            "changemap": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount frontend
frontend_path = PROJECT_ROOT / "frontend"
frontend_path.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

def main():
    uvicorn.run("src.api.app:app", host=API_HOST, port=API_PORT, reload=True)

if __name__ == '__main__':
    main()
