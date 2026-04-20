# Software Defect Prediction - Web Application

A lightweight, fast web application for training and evaluating multiple machine learning models on software defect prediction data.

## 🚀 Features

### Core Functionality

- **File Upload Interface**
  - Drag-and-drop or click to browse
  - CSV file validation
  - Real-time progress indicator during training

- **Multi-Model Training**
  - Trains 5 models simultaneously:
    - Random Forest
    - XGBoost
    - SVM (Linear kernel)
    - SVM (RBF kernel)
    - SVM (Sigmoid kernel)
  - Training completes in ~7-9 seconds

- **Results Dashboard**
  - Performance metrics for all models:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
  - Confusion matrix for each model
  - Automatic best model identification
  - Dataset statistics display

- **Modern UI**
  - Responsive design
  - Clean, professional interface
  - Purple gradient theme
  - Smooth animations and transitions

## 🛠️ Technical Stack

- **Backend**: FastAPI (Python)
- **ML Libraries**:
  - scikit-learn (Random Forest, SVM)
  - XGBoost
  - imbalanced-learn (SMOTE)
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
docker run -p 8000:8000 defect-prediction-app

# Open browser
http://localhost:8000
```

### Option 3: Docker with Jupyter (Optional)

```bash
# Run Jupyter notebook instead
docker run -p 8888:8888 defect-prediction-app \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 📁 Project Structure

```
DataScience/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # ML training pipeline
│   └── templates/
│       └── index.html       # Web interface
├── data/
│   └── software_defect_prediction_dataset.csv
├── uploads/                 # Temporary upload storage
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
└── README.md
```

## 🔧 API Endpoints

### `GET /`
Returns the main web interface (HTML page).

### `POST /upload`
Accepts CSV file upload and trains all models.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: CSV file (field name: "file")

**Response:**
```json
{
  "best_model": "XGBoost",
  "dataset_info": {
    "total_samples": 60000,
    "train_samples": 115246,
    "test_samples": 1200,
    "defective_in_test": 600,
    "non_defective_in_test": 600
  },
  "models": [
    {
      "model_name": "Random Forest",
      "accuracy": 0.985,
      "precision": 0.970873786407767,
      "recall": 1.0,
      "f1_score": 0.9852216748768473,
      "confusion_matrix": [[582, 18], [0, 600]]
    },
    ...
  ]
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## 📊 Expected CSV Format

The CSV file should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `lines_of_code` | int | Total lines of code in the module |
| `cyclomatic_complexity` | int | Code complexity metric |
| `num_functions` | int | Number of functions/methods |
| `num_classes` | int | Number of classes |
| `comment_density` | float | Ratio of comments to code |
| `code_churn` | int | Amount of code changes |
| `developer_experience_years` | int | Developer experience level |
| `num_developers` | int | Number of contributors |
| `commit_frequency` | int | Number of commits |
| `bug_fix_commits` | int | Commits marked as bug fixes |
| `past_defects` | int | Historical defect count |
| `test_coverage` | float | Percentage of code tested |
| `duplication_percentage` | float | Code duplication percentage |
| `avg_function_length` | int | Average lines per function |
| `depth_of_inheritance` | int | Class inheritance depth |
| `response_for_class` | int | Methods executable per class |
| `coupling_between_objects` | int | Class dependencies |
| `lack_of_cohesion` | float | Class cohesion metric |
| `build_failures` | int | Failed build count |
| `static_analysis_warnings` | int | Linter warnings |
| `security_vulnerabilities` | int | Security issues found |
| `performance_issues` | int | Performance bottlenecks |
| `defect` | int | Target: 0 (clean) or 1 (defective) |

## ⚡ Performance Optimizations

- **Parallel Processing**: Models use all CPU cores (n_jobs=-1)
- **Smart Sampling**: SVMs trained on stratified subsets (5k-10k samples)
- **Data Scaling**: StandardScaler improves SVM convergence
- **Fast SVM**: LinearSVC used for linear kernel (much faster than SVC)
- **SMOTE Balancing**: Handles class imbalance efficiently

## 🎯 Typical Results

```
Training Time: ~7-9 seconds
Best Model: XGBoost

Model Performance:
├─ XGBoost:       99.75% accuracy
├─ Random Forest: 98.50% accuracy
├─ SVM (Linear):  94.58% accuracy
├─ SVM (Sigmoid): 94.17% accuracy
└─ SVM (RBF):     92.17% accuracy
```

## 🔄 Workflow

1. **Upload**: User uploads CSV file via web interface
2. **Validation**: System validates file format
3. **Data Prep**: SMOTE oversampling balances classes
4. **Training**: All 5 models train in parallel
5. **Evaluation**: Models tested on hold-out set
6. **Results**: Metrics displayed in dashboard
7. **Best Model**: System identifies top performer

## 🧪 Testing

Test the API directly with curl:

```bash
# Health check
curl http://localhost:8000/health

# Upload and train
curl -X POST http://localhost:8000/upload \
  -F "file=@data/software_defect_prediction_dataset.csv"
```

## 🐳 Docker Configuration

The Dockerfile supports two modes:

**Default (Web App)**:
```bash
docker run -p 8000:8000 defect-prediction-app
```

**Jupyter Notebook**:
```bash
docker run -p 8888:8888 defect-prediction-app \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 🚧 Current Limitations

- No data validation (assumes correct format)
- No persistent model storage
- Single-user (no authentication)
- In-memory processing only

## 🔮 Future Enhancements (Phase 2)

- Feature importance visualizations
- SHAP value plots
- Download results as PDF/CSV
- Model comparison charts
- Progress bar with real-time updates
- Model save/load functionality
- Prediction mode for new data
- Hyperparameter tuning interface

## 📝 License

This project is part of a Data Science course project.

## 🤝 Contributing

This is a Proof of Concept. For production use, consider:
- Adding input validation
- Implementing user authentication
- Adding database for result persistence
- Implementing caching
- Adding comprehensive error handling
- Writing unit tests
