# AI Engineering Assessment

This repository contains implementations for two AI systems:
1. **Student Dropout Prediction**: Early identification of at-risk students
2. **Hospital Readmission Prediction**: 30-day readmission risk assessment

## Features
- Complete ML pipelines with data preprocessing, feature engineering, and model training
- RESTful APIs for real-time predictions
- Comprehensive monitoring and drift detection
- Docker containerization and Kubernetes deployment
- Extensive unit tests and documentation

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/ai-engineering-assessment.git
cd ai-engineering-assessment

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start API server
python api/app.py
Project Structure

src/: Core implementation modules
notebooks/: Jupyter notebooks for exploration and analysis
api/: REST API implementation
deployment/: Containerization and deployment scripts
monitoring/: Model monitoring and drift detection
tests/: Unit and integration tests
Core ML libraries
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
Deep learning
tensorflow==2.13.0
torch==2.0.1
Data processing
scipy==1.10.1
imbalanced-learn==0.10.1
Feature engineering
feature-engine==1.6.2
Model interpretation
shap==0.42.1
lime==0.2.0.1
API and web
flask==2.3.2
flask-restx==1.1.0
pydantic==2.0.3
Data visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
Monitoring and logging
mlflow==2.5.0
wandb==0.15.8
prometheus-client==0.17.1
Database
sqlalchemy==2.0.19
psycopg2-binary==2.9.7
Utilities
pyyaml==6.0.1
python-dotenv==1.0.0
joblib==1.3.1
Testing
pytest==7.4.0
pytest-cov==4.1.0
pytest-mock==3.11.1
Code quality
black==23.7.0
flake8==6.0.0
pre-commit==3.3.3

### config/config.yaml
```yaml
# Configuration file for AI Engineering Assessment

# Database Configuration
database:
  host: "localhost"
  port: 5432
  name: "ai_assessment_db"
  user: "ai_user"
  password: "${DB_PASSWORD}"

# Model Configuration
models:
  student_dropout:
    algorithm: "xgboost"
    hyperparameters:
      learning_rate: 0.1
      max_depth: 6
      n_estimators: 100
      random_state: 42
    
  hospital_readmission:
    algorithm: "ensemble"
    hyperparameters:
      rf_n_estimators: 100
      rf_max_depth: 10
      lr_c: 1.0
      random_state: 42

# Data Configuration
data:
  student_dropout:
    train_ratio: 0.7
    validation_ratio: 0.15
    test_ratio: 0.15
    
  hospital_readmission:
    train_ratio: 0.7
    validation_ratio: 0.15
    test_ratio: 0.15

# API Configuration
api:
  host: "0.0.0.0"
  port: 5000
  debug: false

# Monitoring Configuration
monitoring:
  drift_threshold: 0.05
  performance_threshold: 0.8
  alert_email: "admin@hospital.com"
