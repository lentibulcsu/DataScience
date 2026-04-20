# Software Defect Prediction Web App - PoC

This is a Proof of Concept (PoC) web application that allows users to upload CSV files containing software metrics and train multiple machine learning models to predict software defects.

## Features

- 📤 **File Upload Interface**: Simple drag-and-drop or click-to-browse CSV upload
- 🤖 **Multiple ML Models**: Trains 5 models simultaneously
  - Random Forest
  - XGBoost
  - SVM (Linear Kernel)
  - SVM (RBF Kernel)
  - SVM (Sigmoid Kernel)
- 📊 **Results Dashboard**:
  - Model performance comparison table
  - Confusion matrices for each model
  - Best model recommendation
  - Dataset statistics
- 🎨 **Modern UI**: Clean, responsive interface with real-time progress indication

## Quick Start

### Option 1: Run with Python

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
uvicorn app.main:app --reload
```

3. Open your browser and navigate to:
```
http://localhost:8000
```

### Option 2: Run with Docker

1. Build the Docker image:
```bash
docker build -t defect-prediction-app .
```

2. Run the container:
```bash
docker run -p 8000:8000 defect-prediction-app
```

3. Open your browser and navigate to:
```
http://localhost:8000
```

## Usage

1. Click "Choose File" or drag and drop your CSV file
2. Wait for models to train (30-60 seconds)
3. View results:
   - Dataset information
   - Best performing model
   - Performance metrics for all models
   - Confusion matrices

## Expected CSV Format

The CSV should contain the following columns:
- `lines_of_code`
- `cyclomatic_complexity`
- `num_functions`
- `num_classes`
- `comment_density`
- `code_churn`
- `developer_experience_years`
- `num_developers`
- `commit_frequency`
- `bug_fix_commits`
- `past_defects`
- `test_coverage`
- `duplication_percentage`
- `avg_function_length`
- `depth_of_inheritance`
- `response_for_class`
- `coupling_between_objects`
- `lack_of_cohesion`
- `build_failures`
- `static_analysis_warnings`
- `security_vulnerabilities`
- `performance_issues`
- `defect` (target variable: 0 = non-defective, 1 = defective)

## API Endpoints

- `GET /` - Web interface
- `POST /upload` - Upload CSV and train models
- `GET /health` - Health check endpoint

## Technology Stack

- **Backend**: FastAPI
- **ML Libraries**: scikit-learn, XGBoost, imbalanced-learn
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Server**: Uvicorn

## Phase 1 Complete ✅

This PoC includes all Phase 1 features:
- ✅ File upload interface
- ✅ Train all 5 models
- ✅ Display accuracy, precision, recall, F1-score
- ✅ Show confusion matrices
- ✅ Display best model

## Future Enhancements (Phase 2 & 3)

- Feature importance charts
- Download results as PDF/CSV
- Model comparison visualizations
- Progress bar during training
- SHAP value plots
- Save/load trained models
- Make predictions on new data
