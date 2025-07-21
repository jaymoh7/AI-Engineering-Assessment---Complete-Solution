# src/hospital_readmission/model_training.py
"""
Hospital Readmission Prediction - Model Training Module

This module implements ensemble models for predicting 30-day hospital readmissions
with emphasis on interpretability and clinical relevance.

Author: AI Engineering Assessment
Date: 2025-07-21
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
import logging
import pickle
import json
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Model interpretation
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HospitalReadmissionPredictor:
    """
    Ensemble model for hospital readmission prediction with clinical interpretability.
    
    Combines Random Forest and Logistic Regression for balanced performance
    between accuracy and interpretability, crucial for healthcare applications.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the readmission predictor with configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration parameters for models
        """
        self.config = config or self._get_default_config()
        
        # Initialize models
        self.rf_model = RandomForestClassifier(**self.config['random_forest'])
        self.lr_model = LogisticRegression(**self.config['logistic_regression'])
        
        # Ensemble weights (learned during training)
        self.ensemble_weights = None
        
        # Model metadata
        self.feature_names = None
        self.training_metrics = {}
        self.feature_importance = {}
        self.is_trained = False
        
        # SHAP explainers for interpretability
        self.rf_explainer = None
        self.lr_explainer = None
        
        logger.info("Hospital Readmission Predictor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters for models.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            },
            'logistic_regression': {
                'C': 1.0,
                'penalty': 'l2',
                'random_state': 42,
                'max_iter': 1000,
                'class_weight': 'balanced'
            },
            'ensemble': {
                'use_smote': True,
                'cv_folds': 5,
                'calibrate_probabilities': True
            }
        }
    
    def prepare_clinical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinically relevant features for readmission prediction.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Enhanced feature set with clinical indicators
        """
        X_enhanced = X.copy()
        
        # Comorbidity Index (simplified Charlson score)
        comorbidity_conditions = [
            'diabetes', 'hypertension', 'heart_disease', 'kidney_disease',
            'liver_disease', 'copd', 'cancer', 'stroke'
        ]
        
        available_conditions = [col for col in comorbidity_conditions if col in X.columns]
        if available_conditions:
            X_enhanced['comorbidity_score'] = X[available_conditions].sum(axis=1)
            logger.info(f"Created comorbidity score using {len(available_conditions)} conditions")
        
        # Length of stay categories (clinical significance)
        if 'length_of_stay' in X.columns:
            X_enhanced['los_category'] = pd.cut(
                X['length_of_stay'],
                bins=[0, 3, 7, 14, float('inf')],
                labels=['short', 'medium', 'long', 'extended']
            )
            logger.info("Created length of stay categories")
        
        # Medication complexity score
        medication_cols = [col for col in X.columns if 'medication' in col.lower()]
        if medication_cols:
            X_enhanced['medication_complexity'] = X[medication_cols].sum(axis=1)
            logger.info("Created medication complexity score")
        
        # Emergency admission flag (higher risk)
        if 'admission_type' in X.columns:
            X_enhanced['emergency_admission'] = (X['admission_type'] == 'emergency').astype(int)
            logger.info("Created emergency admission indicator")
        
        # Discharge destination risk (home vs facility)
        if 'discharge_destination' in X.columns:
            high_risk_destinations = ['skilled_nursing', 'rehabilitation', 'hospice']
            X_enhanced['high_risk_discharge'] = X['discharge_destination'].isin(high_risk_destinations).astype(int)
            logger.info("Created high-risk discharge indicator")
        
        return X_enhanced
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE for minority class oversampling.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Balanced dataset
        """
        # Check class distribution
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.min() / class_counts.max()
        
        logger.info(f"Original class distribution: {dict(class_counts)}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.3f}")
        
        if imbalance_ratio < 0.3 and self.config['ensemble']['use_smote']:
            logger.info("Applying SMOTE to balance classes")
            
            # Use SMOTE with careful parameter selection for medical data
            smote = SMOTE(
                sampling_strategy='minority',  # Only oversample minority class
                k_neighbors=min(5, class_counts.min() - 1),  # Ensure enough neighbors
                random_state=42
            )
            
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Log new distribution
            new_class_counts = pd.Series(y_balanced).value_counts()
            logger.info(f"Balanced class distribution: {dict(new_class_counts)}")
            
            return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
        
        return X, y
    
    def train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train individual models (Random Forest and Logistic Regression).
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, Any]: Training metrics for individual models
        """
        metrics = {}
        
        # Train Random Forest
        logger.info("Training Random Forest model")
        self.rf_model.fit(X_train, y_train)
        
        # Cross-validation for Random Forest
        rf_cv_scores = cross_val_score(
            self.rf_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.config['ensemble']['cv_folds'], shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        metrics['random_forest'] = {
            'cv_auc_mean': rf_cv_scores.mean(),
            'cv_auc_std': rf_cv_scores.std(),
            'feature_importance': dict(zip(X_train.columns, self.rf_model.feature_importances_))
        }
        
        # Train Logistic Regression
        logger.info("Training Logistic Regression model")
        self.lr_model.fit(X_train, y_train)
        
        # Cross-validation for Logistic Regression
        lr_cv_scores = cross_val_score(
            self.lr_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.config['ensemble']['cv_folds'], shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        metrics['logistic_regression'] = {
            'cv_auc_mean': lr_cv_scores.mean(),
            'cv_auc_std': lr_cv_scores.std(),
            'coefficients': dict(zip(X_train.columns, self.lr_model.coef_[0]))
        }
        
        logger.info(f"RF CV AUC: {metrics['random_forest']['cv_auc_mean']:.3f} ± {metrics['random_forest']['cv_auc_std']:.3f}")
        logger.info(f"LR CV AUC: {metrics['logistic_regression']['cv_auc_mean']:.3f} ± {metrics['logistic_regression']['cv_auc_std']:.3f}")
        
        return metrics
    
    def optimize_ensemble_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> np.ndarray:
        """
        Optimize ensemble weights using validation set performance.
        
        Args:
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            
        Returns:
            np.ndarray: Optimal ensemble weights
        """
        # Get predictions from individual models
        rf_probs = self.rf_model.predict_proba(X_val)[:, 1]
        lr_probs = self.lr_model.predict_proba(X_val)[:, 1]
        
        # Grid search for optimal weights
        best_auc = 0
        best_weights = None
        
        for rf_weight in np.arange(0.1, 1.0, 0.1):
            lr_weight = 1.0 - rf_weight
            ensemble_probs = rf_weight * rf_probs + lr_weight * lr_probs
            auc = roc_auc_score(y_val, ensemble_probs)
            
            if auc > best_auc:
                best_auc = auc
                best_weights = np.array([rf_weight, lr_weight])
        
        logger.info(f"Optimal ensemble weights: RF={best_weights[0]:.2f}, LR={best_weights[1]:.2f}")
        logger.info(f"Ensemble validation AUC: {best_auc:.3f}")
        
        return best_weights
    
    def calibrate_probabilities(self, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Calibrate probability predictions for clinical reliability.
        
        Args:
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
        """
        if self.config['ensemble']['calibrate_probabilities']:
            logger.info("Calibrating probability predictions")
            
            # Calibrate Random Forest (typically overconfident)
            self.rf_calibrated = CalibratedClassifierCV(
                self.rf_model, method='isotonic', cv=3
            )
            self.rf_calibrated.fit(X_val, y_val)
            
            # Calibrate Logistic Regression (usually well-calibrated, but double-check)
            self.lr_calibrated = CalibratedClassifierCV(
                self.lr_model, method='platt', cv=3
            )
            self.lr_calibrated.fit(X_val, y_val)
            
            logger.info("Probability calibration completed")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Complete training pipeline for the ensemble model.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training target
            
        Returns:
            Dict[str, Any]: Complete training metrics
        """
        logger.info("Starting ensemble model training pipeline")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Create clinical features
        X_enhanced = self.prepare_clinical_features(X)
        
        # Split data for ensemble optimization
        X_train, X_val, y_train, y_val = train_test_split(
            X_enhanced, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # Train individual models
        individual_metrics = self.train_individual_models(X_train_balanced, y_train_balanced)
        
        # Optimize ensemble weights
        self.ensemble_weights = self.optimize_ensemble_weights(X_val, y_val)
        
        # Calibrate probabilities
        self.calibrate_probabilities(X_val, y_val)
        
        # Initialize SHAP explainers for model interpretation
        self.initialize_explainers(X_train_balanced.sample(n=min(100, len(X_train_balanced))))
        
        # Final evaluation
        final_metrics = self.evaluate_ensemble(X_val, y_val)
        
        # Combine all metrics
        self.training_metrics = {
            'individual_models': individual_metrics,
            'ensemble_performance': final_metrics,
            'training_info': {
                'training_samples': len(X_train_balanced),
                'validation_samples': len(X_val),
                'feature_count': X_enhanced.shape[1],
                'training_timestamp': datetime.now().isoformat()
            }
        }
        
        self.is_trained = True
        logger.info("Ensemble model training completed successfully")
        
        return self.training_metrics
    
    def initialize_explainers(self, X_sample: pd.DataFrame):
        """
        Initialize SHAP explainers for model interpretability.
        
        Args:
            X_sample (pd.DataFrame): Sample data for explainer initialization
        """
        try:
            logger.info("Initializing SHAP explainers")
            
            # Random Forest explainer
            self.rf_explainer = shap.TreeExplainer(self.rf_model)
            
            # Logistic Regression explainer
            self.lr_explainer = shap.LinearExplainer(self.lr_model, X_sample)
            
            logger.info("SHAP explainers initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainers: {e}")
            self.rf_explainer = None
            self.lr_explainer = None
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble probability predictions.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Probability predictions [prob_negative, prob_positive]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Enhance features
        X_enhanced = self.prepare_clinical_features(X)
        
        # Get individual model predictions
        if hasattr(self, 'rf_calibrated'):
            rf_probs = self.rf_calibrated.predict_proba(X_enhanced)[:, 1]
        else:
            rf_probs = self.rf_model.predict_proba(X_enhanced)[:, 1]
            
        if hasattr(self, 'lr_calibrated'):
            lr_probs = self.lr_calibrated.predict_proba(X_enhanced)[:, 1]
        else:
            lr_probs = self.lr_model.predict_proba(X_enhanced)[:, 1]
        
        # Ensemble prediction
        ensemble_probs = (
            self.ensemble_weights[0] * rf_probs + 
            self.ensemble_weights[1] * lr_probs
        )
        
        # Return in sklearn format [prob_class_0, prob_class_1]
        return np.column_stack([1 - ensemble_probs, ensemble_probs])
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Generate ensemble class predictions.
        
        Args:
            X (pd.DataFrame): Input features
            threshold (float): Decision threshold
            
        Returns:
            np.ndarray: Class predictions
        """
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)
    
    def get_feature_importance(self, importance_type: str = 'both') -> Dict[str, float]:
        """
        Get feature importance from ensemble models.
        
        Args:
            importance_type (str): Type of importance ('rf', 'lr', or 'both')
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if importance_type == 'rf' or importance_type == 'both':
            rf_importance = dict(zip(self.feature_names, self.rf_model.feature_importances_))
            
        if importance_type == 'lr' or importance_type == 'both':
            lr_importance = dict(zip(self.feature_names, np.abs(self.lr_model.coef_[0])))
            
        if importance_type == 'both':
            # Combine importances using ensemble weights
            combined_importance = {}
            for feature in self.feature_names:
                combined_importance[feature] = (
                    self.ensemble_weights[0] * rf_importance.get(feature, 0) +
                    self.ensemble_weights[1] * lr_importance.get(feature, 0)
                )
            return combined_importance
        elif importance_type == 'rf':
            return rf_importance
        else:
            return lr_importance
    
    def explain_prediction(self, X: pd.DataFrame, patient_idx: int = 0) -> Dict[str, Any]:
        """
        Explain individual prediction using SHAP values.
        
        Args:
            X (pd.DataFrame): Input features
            patient_idx (int): Index of patient to explain
            
        Returns:
            Dict[str, Any]: Explanation including SHAP values and clinical insights
        """
        if not self.is_trained or self.rf_explainer is None:
            logger.warning("SHAP explainers not available")
            return {"error": "Explainers not initialized"}
        
        X_enhanced = self.prepare_clinical_features(X)
        patient_data = X_enhanced.iloc[patient_idx:patient_idx+1]
        
        # Get SHAP values
        rf_shap_values = self.rf_explainer.shap_values(patient_data)
        
        # Get prediction and probability
        prediction_prob = self.predict_proba(patient_data)[0, 1]
        prediction = self.predict(patient_data)[0]
        
        # Create explanation
        explanation = {
            'patient_id': patient_idx,
            'prediction': int(prediction),
            'readmission_probability': float(prediction_prob),
            'risk_level': self._categorize_risk(prediction_prob),
            'shap_values': dict(zip(X_enhanced.columns, rf_shap_values[1][0])),
            'top_risk_factors': self._get_top_risk_factors(
                dict(zip(X_enhanced.columns, rf_shap_values[1][0]))
            ),
            'clinical_recommendations': self._generate_clinical_recommendations(
                patient_data.iloc[0], prediction_prob
            )
        }
        
        return explanation
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level based on probability."""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _get_top_risk_factors(self, shap_values: Dict[str, float], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top risk factors from SHAP values."""
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = []
        for feature, value in sorted_features[:top_n]:
            top_factors.append({
                'feature': feature,
                'shap_value': float(value),
                'impact': 'increases' if value > 0 else 'decreases'
            })
        
        return top_factors
    
    def _generate_clinical_recommendations(self, patient_data: pd.Series, probability: float) -> List[str]:
        """Generate clinical recommendations based on risk factors."""
        recommendations = []
        
        if probability > 0.7:
            recommendations.append("Consider intensive discharge planning and follow-up")
            recommendations.append("Evaluate for home health services or skilled nursing facility")
        
        if 'comorbidity_score' in patient_data and patient_data['comorbidity_score'] > 3:
            recommendations.append("Coordinate care among multiple specialists")
            recommendations.append("Review medication management and potential interactions")
        
        if 'emergency_admission' in patient_data and patient_data['emergency_admission'] == 1:
            recommendations.append("Address underlying causes of emergency admission")
            recommendations.append("Consider preventive care strategies")
        
        return recommendations
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the ensemble model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, Any]: Detailed evaluation metrics
        """
        # Get predictions
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Get classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Clinical metrics
        sensitivity = class_report['1']['recall']  # True positive rate
        specificity = class_report['0']['recall']  # True negative rate
        ppv = class_report['1']['precision']  # Positive predictive value
        npv = class_report['0']['precision']  # Negative predictive value
        
        metrics = {
            'auc_score': auc_score,
            'average_precision': avg_precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        logger.info(f"Ensemble Model Performance:")
        logger.info(f"  AUC Score: {auc_score:.3f}")
        logger.info(f"  Sensitivity: {sensitivity:.3f}")
        logger.info(f"  Specificity: {specificity:.3f}")
        logger.info(f"  PPV: {ppv:.3f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained ensemble model."""
        model_data = {
            'rf_model': self.rf_model,
            'lr_model': self.lr_model,
            'ensemble_weights': self.ensemble_weights,
            'feature_names': self.feature_names,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'is_trained': self.is_trained
        }
        
        # Save calibrated models if available
        if hasattr(self, 'rf_calibrated'):
            model_data['rf_calibrated'] = self.rf_calibrated
        if hasattr(self, 'lr_calibrated'):
            model_data['lr_calibrated'] = self.lr_calibrated
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained ensemble model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rf_model = model_data['rf_model']
        self.lr_model = model_data['lr_model']
        self.ensemble_weights = model_data['ensemble_weights']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.training_metrics = model_data['training_metrics']
        self.is_trained = model_data['is_trained']
        
        # Load calibrated models if available
        if 'rf_calibrated' in model_data:
            self.rf_calibrated = model_data['rf_calibrated']
        if 'lr_calibrated' in model_data:
            self.lr_calibrated = model_data['lr_calibrated']
        
        logger.info(f"Model loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Create sample hospital data
    np.random.seed(42)
    n_patients = 5000
    
    sample_data = {
        'patient_id': range(n_patients),
        'age': np.random.normal(65, 15, n_patients),
        'length_of_stay': np.random.exponential(5, n_patients),
        'num_procedures': np.random.poisson(2, n_patients),
        'num_medications': np.random.poisson(8, n_patients),
        'diabetes': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'hypertension': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
        'heart_disease': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'admission_type': np.random.choice(['elective', 'emergency', 'urgent'], n_patients),
        'discharge_destination': np.random.choice(['home', 'skilled_nursing', 'rehabilitation'], n_patients),
        'readmission_30_day': np.random.choice([0, 1], n_patients, p=[0.85, 0.15])
    }
    
    df = pd.DataFrame(sample_data)
    
    # Split features and target
    X = df.drop(['patient_id', 'readmission_30_day'], axis=1)
    y = df['readmission_30_day']
    
    # Initialize and train model
    predictor = HospitalReadmissionPredictor()
    training_metrics = predictor.train(X, y)
    
    # Test predictions
    sample_patient = X.iloc[[0]]
    prediction = predictor.predict(sample_patient)
    probability = predictor.predict_proba(sample_patient)
    explanation = predictor.explain_prediction(X, 0)
    
    print(f"Sample prediction: {prediction[0]}")
    print(f"Sample probability: {probability[0, 1]:.3f}")
    print(f"Training AUC: {training_metrics['ensemble_performance']['auc_score']:.3f}")
