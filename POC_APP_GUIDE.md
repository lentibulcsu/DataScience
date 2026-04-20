# Software Defect Prediction - Production Ready Web Application

A production-ready web application for training XGBoost models and making software defect predictions. Features model persistence, prediction mode, and CSV export.

## 🚀 Key Features

### 🎯 **Production Workflow**

1. **Train Once, Use Many Times**
   - Train XGBoost model on labeled data
   - Automatically save trained models
   - Reuse saved models for multiple predictions

2. **Model Management**
   - Browse all previously trained models
   - Select models by training date
   - Option to retrain when data changes

3. **Prediction Mode**
   - Upload unlabeled CSV files
   - Get instant predictions
   - Download results with probability scores

4. **CSV Export**
   - Predictions included in output
   - Confidence scores (probabilities) for each prediction
   - Easy integration with other tools

### 💪 **Technical Capabilities**

- **Fast Training**: ~3-5 seconds on 60k samples
- **Model Persistence**: Save/load models with joblib
- **XGBoost Only**: Best performing model (99.75% accuracy)
- **Scalable**: Handle large datasets efficiently
- **Production Ready**: Proper error handling and validation

## 🛠️ Technical Stack

- **Backend**: FastAPI (Python)
- **ML Model**: XGBoost (optimized, parallelized)
- **Data Processing**:
  - scikit-learn (preprocessing, metrics)
  - imbalanced-learn (SMOTE balancing)
  - pandas, numpy
- **Model Persistence**: joblib
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Server**: Uvicorn (ASGI)
- **Containerization**: Docker

## 📦 Installation & Usage

### Option 1: Local Development

```bash
# Clone and switch to PoC branch
git clone https://github.com/lentibulcsu/DataScience.git
cd DataScience
git checkout poc-web-app

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --reload

# Open browser
http://localhost:8000
```

### Option 2: Docker

```bash
# Build the image
docker build -t defect-prediction-app .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/predictions:/app/predictions \
  defect-prediction-app

# Open browser
http://localhost:8000
```

**Note**: Volume mounts preserve trained models and predictions between container restarts.

## 📁 Project Structure

```
DataScience/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # ML training & prediction logic
│   └── templates/
│       └── index.html       # Web interface
├── data/
│   └── software_defect_prediction_dataset.csv
├── saved_models/            # Trained models storage
│   ├── .gitkeep
│   └── xgboost_model_*.joblib
├── predictions/             # Prediction results
│   ├── .gitkeep
│   └── predictions_*.csv
├── uploads/                 # Temporary upload storage
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔄 Complete Workflow

### 1️⃣ Initial Training

1. Visit `http://localhost:8000`
2. Click **"Train New Model"**
3. Upload CSV with labeled data (includes 'defect' column)
4. Wait ~3-5 seconds for training
5. Review model performance metrics
6. Model is automatically saved

### 2️⃣ Making Predictions

1. Click **"Make Predictions"**
2. Select a trained model from dropdown
3. Upload CSV with unlabeled data (no 'defect' column)
4. View prediction summary:
   - Total samples processed
   - Predicted defective count
   - Predicted clean count
   - Defect rate percentage
5. Download predictions CSV

### 3️⃣ Prediction CSV Output

The output CSV includes:
- All original features
- `predicted_defect`: 0 (clean) or 1 (defective)
- `defect_probability`: Confidence score for defect
- `non_defect_probability`: Confidence score for clean

### 4️⃣ Retraining

- Option 1: Click **"New Prediction"** → train new model
- Option 2: Click **"Start Over"** → return to main menu
- Models are timestamped, you can keep multiple versions

## 🔧 API Endpoints

### `GET /`
Main web interface.

### `POST /train`
Train new XGBoost model on labeled data.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: CSV file with 'defect' column

**Response:**
```json
{
  "model_name": "XGBoost",
  "model_filename": "xgboost_model_20260420_214346.joblib",
  "timestamp": "20260420_214346",
  "accuracy": 0.9975,
  "precision": 0.995,
  "recall": 1.0,
  "f1_score": 0.9975,
  "confusion_matrix": [[597, 3], [0, 600]],
  "dataset_info": {
    "total_samples": 60000,
    "train_samples": 115246,
    "test_samples": 1200
  }
}
```

### `GET /models`
List all saved models.

**Response:**
```json
{
  "models": [
    {
      "filename": "xgboost_model_20260420_214346.joblib",
      "path": "/path/to/model",
      "timestamp": "20260420_214346",
      "created": "2026-04-20 21:43:46"
    }
  ]
}
```

### `POST /predict`
Make predictions using selected model.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - file: CSV file without 'defect' column
  - model_filename: Name of model to use

**Response:**
```json
{
  "total_samples": 60000,
  "predicted_defective": 58226,
  "predicted_non_defective": 1774,
  "defect_rate": 97.04,
  "output_file": "/path/to/predictions.csv",
  "output_filename": "predictions_20260420_214410.csv"
}
```

### `GET /download/{filename}`
Download prediction results CSV.

**Response:** CSV file download

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## 📊 Expected Data Format

### Training Data (Labeled)

CSV file with 23 columns including target:

| Column | Type | Description |
|--------|------|-------------|
| `lines_of_code` | int | Total lines of code |
| `cyclomatic_complexity` | int | Code complexity metric |
| `num_functions` | int | Number of functions |
| `num_classes` | int | Number of classes |
| `comment_density` | float | Comment ratio |
| `code_churn` | int | Code change amount |
| `developer_experience_years` | int | Developer experience |
| `num_developers` | int | Number of contributors |
| `commit_frequency` | int | Commit count |
| `bug_fix_commits` | int | Bug fix commits |
| `past_defects` | int | Historical defects |
| `test_coverage` | float | Test coverage % |
| `duplication_percentage` | float | Code duplication % |
| `avg_function_length` | int | Avg lines per function |
| `depth_of_inheritance` | int | Inheritance depth |
| `response_for_class` | int | Methods per class |
| `coupling_between_objects` | int | Class dependencies |
| `lack_of_cohesion` | float | Cohesion metric |
| `build_failures` | int | Failed builds |
| `static_analysis_warnings` | int | Linter warnings |
| `security_vulnerabilities` | int | Security issues |
| `performance_issues` | int | Performance problems |
| **`defect`** | **int** | **Target: 0=clean, 1=defective** |

### Prediction Data (Unlabeled)

Same columns as above **except** the `defect` column (or it will be ignored).

## ⚡ Performance & Optimizations

- **Training Time**: 3-5 seconds (60k samples with SMOTE)
- **Prediction Time**: < 1 second (60k samples)
- **Model Size**: ~1-2 MB per saved model
- **Memory Usage**: Efficient streaming for large files
- **Parallelization**: Uses all CPU cores (n_jobs=-1)
- **Data Balancing**: SMOTE handles class imbalance

## 🎯 Model Performance

**XGBoost Results** (on test set):
- **Accuracy**: 99.75%
- **Precision**: 99.50%
- **Recall**: 100%
- **F1-Score**: 99.75%

**Why XGBoost Only?**
- Highest accuracy among all tested models
- Fast training and prediction
- Handles imbalanced data well
- Production-proven performance

## 🧪 Testing

### Test Training
```bash
curl -X POST http://localhost:8000/train \
  -F "file=@data/software_defect_prediction_dataset.csv"
```

### List Models
```bash
curl http://localhost:8000/models
```

### Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_data.csv" \
  -F "model_filename=xgboost_model_20260420_214346.joblib"
```

### Download Results
```bash
curl http://localhost:8000/download/predictions_20260420_214410.csv \
  -o my_predictions.csv
```

## 🐳 Docker Configuration

**Build with custom options:**
```bash
docker build --build-arg PYTHON_VERSION=3.11 -t defect-prediction-app .
```

**Run with persistent storage:**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/predictions:/app/predictions \
  -v $(pwd)/data:/app/data \
  defect-prediction-app
```

## 🔐 Production Considerations

### Current Features
✅ Model persistence
✅ Error handling
✅ Input validation
✅ Health checks
✅ CSV export

### Recommended Additions for Production
- [ ] User authentication
- [ ] Database for model metadata
- [ ] Model versioning system
- [ ] Batch prediction API
- [ ] Model monitoring/drift detection
- [ ] Rate limiting
- [ ] Logging and audit trails
- [ ] Unit and integration tests

## 📝 Use Cases

### 1. Software Development Teams
- Train model on historical bug data
- Predict defect probability for new code modules
- Focus QA efforts on high-risk modules

### 2. Quality Assurance
- Automate preliminary code quality assessment
- Prioritize testing based on defect predictions
- Track code quality trends over time

### 3. Project Managers
- Estimate testing effort required
- Identify problematic code areas early
- Make data-driven decisions on code reviews

### 4. Research & Analysis
- Compare different training datasets
- Analyze which features predict defects
- Experiment with different model versions

## 🔮 Future Enhancements

**Phase 2 Features:**
- [ ] Feature importance visualizations (SHAP)
- [ ] Model comparison dashboard
- [ ] Batch prediction mode
- [ ] PDF report generation
- [ ] Real-time prediction API
- [ ] Model A/B testing
- [ ] Scheduled retraining
- [ ] Email notifications

## 📄 License

This project is part of a Data Science course project.

## 🤝 Contributing

Pull requests welcome! For major changes:
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📞 Support

For issues or questions:
- GitHub Issues: https://github.com/lentibulcsu/DataScience/issues
- Documentation: This file + inline code comments
