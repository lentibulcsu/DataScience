"""
FastAPI web application for software defect prediction.
XGBoost-only with train/predict workflow.
"""

from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from pathlib import Path
from app.models import (
    train_xgboost_model,
    get_available_models,
    predict_with_model
)

# Initialize FastAPI app
app = FastAPI(title="Software Defect Prediction")

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
PREDICTIONS_DIR = BASE_DIR / "predictions"
UPLOAD_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

# Mount static files (create directory if it doesn't exist)
static_dir = BASE_DIR / "app" / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse(
        request=request, name="index.html"
    )


@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    """
    Train XGBoost model on labeled dataset.

    Args:
        file: CSV file with labeled data (includes 'defect' column)

    Returns:
        Training results and model info
    """
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only CSV files are allowed"}
            )

        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Train model
        results = train_xgboost_model(str(file_path))

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Training error: {str(e)}"}
        )


@app.get("/models")
async def list_models():
    """
    Get list of all available trained models.

    Returns:
        List of model info dictionaries
    """
    try:
        models = get_available_models()
        return JSONResponse(content={"models": models})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error loading models: {str(e)}"}
        )


@app.post("/predict")
async def make_predictions(
    file: UploadFile = File(...),
    model_filename: str = Form(...)
):
    """
    Make predictions on unlabeled dataset using selected model.

    Args:
        file: CSV file with unlabeled data (no 'defect' column)
        model_filename: Name of the model file to use

    Returns:
        Prediction results and statistics
    """
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only CSV files are allowed"}
            )

        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get model path
        model_path = BASE_DIR / "saved_models" / model_filename

        if not model_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": "Model not found"}
            )

        # Make predictions
        results = predict_with_model(
            str(model_path),
            str(file_path)
        )

        return JSONResponse(content=results)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction error: {str(e)}"}
        )


@app.get("/download/{filename}")
async def download_predictions(filename: str):
    """
    Download prediction results CSV file.

    Args:
        filename: Name of the predictions file

    Returns:
        CSV file download
    """
    file_path = PREDICTIONS_DIR / filename

    if not file_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='text/csv'
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
