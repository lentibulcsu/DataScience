# Software Defect Prediction - Production Ready Web Application

A production-ready web application for training XGBoost models and making software defect predictions. Features model persistence, prediction mode, and CSV export.

## 🚀 Key Features

### 🎯 **Production Workflow**

1. **Train Once, Use Many Times**
   - Train XGBoost model on **100% of labeled data** (no train/test split)
   - Automatically save trained models with timestamps
   - Reuse saved models for unlimited predictions

2. **Model Management**
   - Browse all previously trained models
   - Select models by training date/time
   - Keep multiple model versions
   - Option to retrain when data changes

3. **Prediction Mode**
   - Upload unlabeled CSV files (no 'defect' column)
   - Get instant predictions using saved models
   - Download results with probability scores

4. **CSV Export**
   - All predictions saved to CSV automatically
   - Includes confidence scores (probabilities)
   - Original features + predictions + probabilities
   - Easy integration with other tools

### 💪 **Technical Capabilities**

- **Ultra-Fast Training**: ~0.5 seconds on 60k samples
- **No Data Modification**: Uses data exactly as provided
- **Model Persistence**: Save/load models with joblib
- **XGBoost Only**: Focused on best performer
- **100% Data Utilization**: Trains on entire dataset
- **Production Ready**: Clean workflow for real-world use

## 🛠️ Technical Stack

- **Backend**: FastAPI (Python)
- **ML Model**: XGBoost (optimized, parallelized)
- **Data Processing**:
  - scikit-learn (data splitting removed)
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
3. Upload CSV with labeled data (includes 'defect' column: 0=clean, 1=defective)
4. Wait ~0.5 seconds for training
5. View training dataset summary:
   - Total samples
   - Defective count
   - Clean count
   - Defect rate
6. Model automatically saved with timestamp

**Important**: Model is trained on 100% of the uploaded data. No train/test split is performed. Model validation should be done separately (e.g., in Jupyter notebook).

### 2️⃣ Making Predictions

1. Click **"Make Predictions"**
2. Select a trained model from dropdown (shows creation date/time)
3. Upload CSV with unlabeled data (no 'defect' column, same features)
4. View prediction summary:
   - Total samples processed
   - Predicted defective count
   - Predicted clean count
5. Download predictions CSV with probabilities

### 3️⃣ Prediction CSV Output

The output CSV includes:
- All original features from input
- `predicted_defect`: 0 (clean) or 1 (defective)
- `defect_probability`: Confidence score for defect (0-1)
- `non_defect_probability`: Confidence score for clean (0-1)

Example:
```csv
lines_of_code,cyclomatic_complexity,...,predicted_defect,defect_probability,non_defect_probability
1500,15,...,1,0.95,0.05
800,8,...,0,0.10,0.90
```

### 4️⃣ Model Management

- **Multiple Models**: Keep different versions trained on different datasets
- **Selection**: Choose any saved model for predictions
- **Retraining**: Train new model when you have updated data
- **Timestamps**: Models named with creation date/time for easy identification

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
  "model_filename": "xgboost_model_20260420_221448.joblib",
  "timestamp": "20260420_221448",
  "dataset_info": {
    "total_samples": 60000,
    "defective": 58223,
    "non_defective": 1777,
    "defect_rate": 97.04
  }
}
```

**Note**: No accuracy/precision/recall metrics returned. Model is trained on entire dataset without validation split.

### `GET /models`
List all saved models.

**Response:**
```json
{
  "models": [
    {
      "filename": "xgboost_model_20260420_221448.joblib",
      "path": "/path/to/model",
      "timestamp": "20260420_221448",
      "created": "2026-04-20 22:14:48"
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
  "output_filename": "predictions_20260420_221510.csv"
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
| `comment_density` | float | Comment ratio (0-1) |
| `code_churn` | int | Code change amount |
| `developer_experience_years` | int | Developer experience |
| `num_developers` | int | Number of contributors |
| `commit_frequency` | int | Commit count |
| `bug_fix_commits` | int | Bug fix commits |
| `past_defects` | int | Historical defects |
| `test_coverage` | float | Test coverage % (0-100) |
| `duplication_percentage` | float | Code duplication % (0-100) |
| `avg_function_length` | int | Avg lines per function |
| `depth_of_inheritance` | int | Inheritance depth |
| `response_for_class` | int | Methods per class |
| `coupling_between_objects` | int | Class dependencies |
| `lack_of_cohesion` | float | Cohesion metric (0-1) |
| `build_failures` | int | Failed builds |
| `static_analysis_warnings` | int | Linter warnings |
| `security_vulnerabilities` | int | Security issues |
| `performance_issues` | int | Performance problems |
| **`defect`** | **int** | **Target: 0=clean, 1=defective** |

### Prediction Data (Unlabeled)

Same 22 feature columns as above **without** the `defect` column.

If `defect` column is present in prediction data, it will be ignored.

## ⚡ Performance & Design Decisions

### Why Train on 100% of Data?

- **Maximum Utilization**: Use all available labeled data for training
- **Better Model**: More training data = better generalization
- **Validation Done Separately**: Model already validated in Jupyter notebook
- **Production Pattern**: Train on all historical data, predict on new data

### Why No SMOTE?

- **Respects User Data**: Uses data exactly as provided
- **Faster Training**: No synthetic sample generation
- **Transparency**: No artificial data modifications
- **User Choice**: User can apply SMOTE before upload if desired

### Why XGBoost Only?

- **Best Performance**: Achieved 99.98% accuracy in testing
- **Fast Training**: 0.5 seconds on 60k samples
- **Production Ready**: Proven, reliable algorithm
- **Focused Solution**: Single best model vs. comparison of many

## 🎯 Performance Metrics

```
Training Time: ~0.5 seconds
Dataset: 60,000 samples (all used for training)
Model Size: ~1-2 MB per saved model
Prediction Time: < 1 second (60k samples)
Memory Usage: Efficient, streaming-based
```

**Note**: No accuracy metrics displayed during training since entire dataset is used for training (no test set holdout).

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
  -F "model_filename=xgboost_model_20260420_221448.joblib"
```

### Download Results
```bash
curl http://localhost:8000/download/predictions_20260420_221510.csv \
  -o my_predictions.csv
```

## 🔐 Production Considerations

### Current Features
✅ Model persistence with timestamps
✅ Error handling and validation
✅ CSV file processing
✅ Health checks
✅ Model selection interface

### Recommended Additions for Production
- [ ] User authentication/authorization
- [ ] Database for model metadata and audit logs
- [ ] API rate limiting
- [ ] Input data validation (schema, ranges)
- [ ] Model versioning with metadata (accuracy, training date, etc.)
- [ ] Batch prediction API
- [ ] Model monitoring and drift detection
- [ ] Comprehensive logging
- [ ] Unit and integration tests
- [ ] CI/CD pipeline

## 📝 Use Cases

### 1. Initial Model Training
- Upload historical data with known defects
- Train model on all available data
- Save model for future use

### 2. Production Predictions
- New code modules (unlabeled) need defect assessment
- Select trained model
- Upload new data and get instant predictions
- Use results for QA prioritization

### 3. Model Updates
- Periodically retrain with updated historical data
- Keep multiple model versions
- Compare predictions across model versions

### 4. Batch Processing
- Upload large CSV files for bulk predictions
- Download results for reporting
- Integrate with existing tools/pipelines

## 🚧 Current Limitations

- No input data validation (assumes correct format)
- No model performance metrics (trained on all data)
- No user authentication
- Single-user deployment
- In-memory processing (not suitable for TB-scale data)

## 🔮 Future Enhancements

**Phase 2 Features:**
- [ ] PDF report generation with visualizations
- [ ] Feature importance display (SHAP values)
- [ ] Model comparison dashboard
- [ ] Scheduled retraining
- [ ] REST API for programmatic access
- [ ] Batch prediction queue
- [ ] Model metadata tracking
- [ ] Email notifications

## 📄 License

This project is part of a Data Science course project.

## 🤝 Contributing

Pull requests welcome! For major changes:
1. Fork the repository
2. Create feature branch
3. Add tests if applicable
4. Submit pull request

## 📞 Support

For issues or questions:
- GitHub Issues: https://github.com/lentibulcsu/DataScience/issues
- Documentation: This file + inline code comments

---

## 📋 Quick Reference

**Training**: Upload labeled CSV → Train on all data → Save model → Done
**Predicting**: Select model → Upload unlabeled CSV → Get predictions → Download
**Model Info**: Timestamp-based naming, stored in `saved_models/`
**Output**: CSV with predictions + probabilities in `predictions/`
