"""
Hospital Readmission Prediction Routes

This module provides Flask-RESTx API endpoints for hospital readmission prediction.
It includes routes for single predictions, batch predictions, model information,
and health monitoring.

Author: AI Engineering Assessment
Date: 2024
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any

from flask import Blueprint, request, jsonify
from flask_restx import Namespace, Resource, fields
import pandas as pd
import numpy as np

from src.hospital_readmission.model_training import HospitalReadmissionModel
from src.hospital_readmission.data_preprocessing import HospitalDataPreprocessor
from src.hospital_readmission.feature_engineering import HospitalFeatureEngineer
from api.schemas.hospital_schema import (
    hospital_input_schema,
    hospital_prediction_response_schema,
    hospital_batch_response_schema,
    hospital_model_info_schema
)

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint for hospital prediction routes
hospital_bp = Blueprint('hospital_prediction', __name__)

# Create Flask-RESTx namespace for API documentation
api = Namespace(
    'hospital',
    description='Hospital Readmission Prediction API',
    path='/api/v1/hospital'
)

# Global variables to store loaded models and preprocessors
# These will be initialized when the application starts
_hospital_model = None
_hospital_preprocessor = None
_hospital_feature_engineer = None
_model_metadata = {}

def initialize_hospital_models():
    """
    Initialize hospital readmission models and preprocessors.
    
    This function loads the trained model, preprocessor, and feature engineer
    from saved files. It should be called during application startup.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global _hospital_model, _hospital_preprocessor, _hospital_feature_engineer, _model_metadata
    
    try:
        logger.info("Initializing hospital readmission models...")
        
        # Load the trained model
        _hospital_model = HospitalReadmissionModel()
        _hospital_model.load_model('models/hospital_readmission/hospital_model.joblib')
        
        # Load the data preprocessor
        _hospital_preprocessor = HospitalDataPreprocessor()
        _hospital_preprocessor.load_preprocessor('models/hospital_readmission/preprocessor.joblib')
        
        # Load the feature engineer
        _hospital_feature_engineer = HospitalFeatureEngineer()
        _hospital_feature_engineer.load_feature_engineer('models/hospital_readmission/feature_engineer.joblib')
        
        # Load model metadata
        import joblib
        try:
            _model_metadata = joblib.load('models/hospital_readmission/model_metadata.joblib')
        except FileNotFoundError:
            _model_metadata = {
                'model_version': '1.0.0',
                'training_date': datetime.now().isoformat(),
                'model_type': 'ensemble',
                'features': [],
                'performance_metrics': {}
            }
        
        logger.info("Hospital readmission models initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize hospital readmission models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def validate_hospital_input(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate input data for hospital readmission prediction.
    
    Args:
        data (Dict[str, Any]): Input data dictionary
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    required_fields = [
        'age', 'gender', 'admission_type', 'discharge_disposition',
        'admission_source', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
        'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',
        'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
        'troglitazone', 'tolazamide', 'examide', 'citoglipton',
        'insulin', 'glyburide_metformin', 'glipizide_metformin',
        'glimepiride_pioglitazone', 'metformin_rosiglitazone',
        'metformin_pioglitazone', 'change', 'diabetesMed'
    ]
    
    # Check for required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate data types and ranges
    try:
        # Age should be between 0 and 150
        age = float(data['age'])
        if age < 0 or age > 150:
            return False, "Age must be between 0 and 150"
        
        # Time in hospital should be positive
        time_in_hospital = int(data['time_in_hospital'])
        if time_in_hospital < 0:
            return False, "Time in hospital must be non-negative"
        
        # Validate categorical fields
        valid_genders = ['Male', 'Female', 'Unknown/Invalid']
        if data['gender'] not in valid_genders:
            return False, f"Gender must be one of: {', '.join(valid_genders)}"
        
        # Validate numeric fields are non-negative
        numeric_fields = [
            'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'number_diagnoses'
        ]
        
        for field in numeric_fields:
            value = int(data[field])
            if value < 0:
                return False, f"{field} must be non-negative"
        
        return True, ""
        
    except (ValueError, TypeError) as e:
        return False, f"Invalid data type: {str(e)}"

@api.route('/predict')
class HospitalPrediction(Resource):
    """Hospital readmission prediction endpoint for single patient."""
    
    @api.doc('predict_hospital_readmission')
    @api.expect(hospital_input_schema)
    @api.marshal_with(hospital_prediction_response_schema)
    def post(self):
        """
        Predict hospital readmission risk for a single patient.
        
        This endpoint accepts patient data and returns the probability of
        30-day hospital readmission along with risk factors and recommendations.
        
        Returns:
            dict: Prediction results including probability, risk level, and explanations
        """
        try:
            # Check if models are initialized
            if _hospital_model is None or _hospital_preprocessor is None:
                logger.error("Hospital models not initialized")
                return {
                    'success': False,
                    'error': 'Models not initialized. Please contact system administrator.',
                    'prediction': None
                }, 500
            
            # Get input data from request
            data = request.get_json()
            
            if not data:
                return {
                    'success': False,
                    'error': 'No input data provided',
                    'prediction': None
                }, 400
            
            # Validate input data
            is_valid, error_message = validate_hospital_input(data)
            if not is_valid:
                logger.warning(f"Invalid input data: {error_message}")
                return {
                    'success': False,
                    'error': error_message,
                    'prediction': None
                }, 400
            
            # Convert input data to DataFrame
            input_df = pd.DataFrame([data])
            
            # Preprocess the data
            logger.info("Preprocessing hospital patient data...")
            preprocessed_data = _hospital_preprocessor.transform(input_df)
            
            # Engineer features
            logger.info("Engineering features for hospital prediction...")
            features = _hospital_feature_engineer.transform(preprocessed_data)
            
            # Make prediction
            logger.info("Making hospital readmission prediction...")
            prediction_proba = _hospital_model.predict_proba(features)[0]
            prediction_binary = _hospital_model.predict(features)[0]
            
            # Get feature importance for explanation
            feature_importance = None
            if hasattr(_hospital_model.model, 'feature_importances_'):
                feature_names = _hospital_feature_engineer.get_feature_names()
                importance_scores = _hospital_model.model.feature_importances_
                
                # Get top 10 most important features
                feature_importance = [
                    {
                        'feature': feature_names[i],
                        'importance': float(importance_scores[i])
                    }
                    for i in np.argsort(importance_scores)[-10:][::-1]
                ]
            
            # Determine risk level based on probability
            readmission_probability = float(prediction_proba[1])  # Probability of readmission
            
            if readmission_probability < 0.3:
                risk_level = 'Low'
                risk_color = 'green'
            elif readmission_probability < 0.7:
                risk_level = 'Medium'
                risk_color = 'yellow'
            else:
                risk_level = 'High'
                risk_color = 'red'
            
            # Generate recommendations based on risk factors
            recommendations = generate_hospital_recommendations(data, readmission_probability)
            
            # Prepare response
            result = {
                'success': True,
                'error': None,
                'prediction': {
                    'readmission_probability': readmission_probability,
                    'predicted_readmission': bool(prediction_binary),
                    'risk_level': risk_level,
                    'risk_color': risk_color,
                    'confidence_interval': [
                        max(0.0, readmission_probability - 0.1),
                        min(1.0, readmission_probability + 0.1)
                    ],
                    'feature_importance': feature_importance,
                    'recommendations': recommendations,
                    'prediction_timestamp': datetime.now().isoformat(),
                    'model_version': _model_metadata.get('model_version', '1.0.0')
                }
            }
            
            logger.info(f"Hospital prediction completed. Risk level: {risk_level}, "
                       f"Probability: {readmission_probability:.3f}")
            
            return result, 200
            
        except Exception as e:
            logger.error(f"Error in hospital prediction: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'prediction': None
            }, 500

@api.route('/predict/batch')
class HospitalBatchPrediction(Resource):
    """Hospital readmission batch prediction endpoint."""
    
    @api.doc('batch_predict_hospital_readmission')
    @api.expect([hospital_input_schema])
    @api.marshal_with(hospital_batch_response_schema)
    def post(self):
        """
        Predict hospital readmission risk for multiple patients.
        
        This endpoint accepts a list of patient data and returns predictions
        for all patients in a single batch operation.
        
        Returns:
            dict: Batch prediction results with individual predictions for each patient
        """
        try:
            # Check if models are initialized
            if _hospital_model is None or _hospital_preprocessor is None:
                return {
                    'success': False,
                    'error': 'Models not initialized',
                    'predictions': [],
                    'summary': {}
                }, 500
            
            # Get input data
            data = request.get_json()
            
            if not data or not isinstance(data, list):
                return {
                    'success': False,
                    'error': 'Input must be a list of patient records',
                    'predictions': [],
                    'summary': {}
                }, 400
            
            if len(data) > 1000:  # Limit batch size
                return {
                    'success': False,
                    'error': 'Batch size cannot exceed 1000 records',
                    'predictions': [],
                    'summary': {}
                }, 400
            
            logger.info(f"Processing batch prediction for {len(data)} hospital patients...")
            
            predictions = []
            successful_predictions = 0
            failed_predictions = 0
            
            # Process each patient record
            for i, patient_data in enumerate(data):
                try:
                    # Validate individual record
                    is_valid, error_message = validate_hospital_input(patient_data)
                    if not is_valid:
                        predictions.append({
                            'record_index': i,
                            'success': False,
                            'error': error_message,
                            'prediction': None
                        })
                        failed_predictions += 1
                        continue
                    
                    # Convert to DataFrame
                    input_df = pd.DataFrame([patient_data])
                    
                    # Preprocess and engineer features
                    preprocessed_data = _hospital_preprocessor.transform(input_df)
                    features = _hospital_feature_engineer.transform(preprocessed_data)
                    
                    # Make prediction
                    prediction_proba = _hospital_model.predict_proba(features)[0]
                    prediction_binary = _hospital_model.predict(features)[0]
                    
                    readmission_probability = float(prediction_proba[1])
                    
                    # Determine risk level
                    if readmission_probability < 0.3:
                        risk_level = 'Low'
                    elif readmission_probability < 0.7:
                        risk_level = 'Medium'
                    else:
                        risk_level = 'High'
                    
                    predictions.append({
                        'record_index': i,
                        'success': True,
                        'error': None,
                        'prediction': {
                            'readmission_probability': readmission_probability,
                            'predicted_readmission': bool(prediction_binary),
                            'risk_level': risk_level,
                            'patient_id': patient_data.get('patient_id', f'patient_{i}')
                        }
                    })
                    successful_predictions += 1
                    
                except Exception as e:
                    logger.error(f"Error processing record {i}: {str(e)}")
                    predictions.append({
                        'record_index': i,
                        'success': False,
                        'error': f'Processing error: {str(e)}',
                        'prediction': None
                    })
                    failed_predictions += 1
            
            # Calculate summary statistics
            successful_preds = [p['prediction'] for p in predictions if p['success']]
            
            if successful_preds:
                probabilities = [p['readmission_probability'] for p in successful_preds]
                risk_distribution = {
                    'low': sum(1 for p in successful_preds if p['risk_level'] == 'Low'),
                    'medium': sum(1 for p in successful_preds if p['risk_level'] == 'Medium'),
                    'high': sum(1 for p in successful_preds if p['risk_level'] == 'High')
                }
                
                summary = {
                    'total_records': len(data),
                    'successful_predictions': successful_predictions,
                    'failed_predictions': failed_predictions,
                    'average_readmission_probability': float(np.mean(probabilities)),
                    'risk_distribution': risk_distribution,
                    'high_risk_count': risk_distribution['high'],
                    'processing_timestamp': datetime.now().isoformat()
                }
            else:
                summary = {
                    'total_records': len(data),
                    'successful_predictions': 0,
                    'failed_predictions': failed_predictions,
                    'processing_timestamp': datetime.now().isoformat()
                }
            
            logger.info(f"Batch prediction completed. Success: {successful_predictions}, "
                       f"Failed: {failed_predictions}")
            
            return {
                'success': True,
                'error': None,
                'predictions': predictions,
                'summary': summary
            }, 200
            
        except Exception as e:
            logger.error(f"Error in hospital batch prediction: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'predictions': [],
                'summary': {}
            }, 500

@api.route('/model/info')
class HospitalModelInfo(Resource):
    """Hospital readmission model information endpoint."""
    
    @api.doc('get_hospital_model_info')
    @api.marshal_with(hospital_model_info_schema)
    def get(self):
        """
        Get information about the hospital readmission prediction model.
        
        Returns model metadata, performance metrics, and feature information.
        
        Returns:
            dict: Model information and metadata
        """
        try:
            if _hospital_model is None:
                return {
                    'success': False,
                    'error': 'Model not initialized',
                    'model_info': {}
                }, 500
            
            # Get model information
            model_info = {
                'model_version': _model_metadata.get('model_version', '1.0.0'),
                'model_type': _model_metadata.get('model_type', 'ensemble'),
                'training_date': _model_metadata.get('training_date'),
                'features': _model_metadata.get('features', []),
                'feature_count': len(_model_metadata.get('features', [])),
                'performance_metrics': _model_metadata.get('performance_metrics', {}),
                'model_parameters': _hospital_model.get_model_params() if hasattr(_hospital_model, 'get_model_params') else {},
                'last_updated': datetime.now().isoformat(),
                'deployment_status': 'active',
                'supported_predictions': ['single', 'batch'],
                'input_features': {
                    'demographic': ['age', 'gender'],
                    'administrative': ['admission_type', 'discharge_disposition', 'admission_source'],
                    'clinical': ['time_in_hospital', 'num_lab_procedures', 'num_procedures'],
                    'medications': ['num_medications', 'diabetesMed', 'change'],
                    'diagnoses': ['diag_1', 'diag_2', 'diag_3', 'number_diagnoses'],
                    'utilization': ['number_outpatient', 'number_emergency', 'number_inpatient']
                },
                'output_format': {
                    'readmission_probability': 'float [0-1]',
                    'predicted_readmission': 'boolean',
                    'risk_level': 'string [Low, Medium, High]',
                    'recommendations': 'list of strings'
                }
            }
            
            return {
                'success': True,
                'error': None,
                'model_info': model_info
            }, 200
            
        except Exception as e:
            logger.error(f"Error getting hospital model info: {str(e)}")
            
            return {
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'model_info': {}
            }, 500

@api.route('/health')
class HospitalHealthCheck(Resource):
    """Hospital readmission service health check endpoint."""
    
    @api.doc('hospital_health_check')
    def get(self):
        """
        Check the health status of the hospital readmission prediction service.
        
        Returns:
            dict: Service health status and diagnostic information
        """
        try:
            # Check model initialization status
            models_ready = all([
                _hospital_model is not None,
                _hospital_preprocessor is not None,
                _hospital_feature_engineer is not None
            ])
            
            # Perform basic model test if models are loaded
            test_passed = False
            if models_ready:
                try:
                    # Create minimal test data
                    test_data = pd.DataFrame([{
                        'age': 65, 'gender': 'Male', 'admission_type': 'Elective',
                        'discharge_disposition': 'Discharged to home',
                        'admission_source': 'Physician Referral',
                        'time_in_hospital': 3, 'num_lab_procedures': 5,
                        'num_procedures': 2, 'num_medications': 8,
                        'number_outpatient': 0, 'number_emergency': 0,
                        'number_inpatient': 0, 'diag_1': '250.00',
                        'diag_2': '401.9', 'diag_3': '272.4',
                        'number_diagnoses': 9, 'max_glu_serum': 'None',
                        'A1Cresult': '>7', 'metformin': 'No',
                        'repaglinide': 'No', 'nateglinide': 'No',
                        'chlorpropamide': 'No', 'glimepiride': 'No',
                        'acetohexamide': 'No', 'glipizide': 'No',
                        'glyburide': 'No', 'tolbutamide': 'No',
                        'pioglitazone': 'No', 'rosiglitazone': 'No',
                        'acarbose': 'No', 'miglitol': 'No',
                        'troglitazone': 'No', 'tolazamide': 'No',
                        'examide': 'No', 'citoglipton': 'No',
                        'insulin': 'Down', 'glyburide_metformin': 'No',
                        'glipizide_metformin': 'No', 'glimepiride_pioglitazone': 'No',
                        'metformin_rosiglitazone': 'No', 'metformin_pioglitazone': 'No',
                        'change': 'Ch', 'diabetesMed': 'Yes'
                    }])
                    
                    # Test preprocessing and prediction
                    preprocessed = _hospital_preprocessor.transform(test_data)
                    features = _hospital_feature_engineer.transform(preprocessed)
                    prediction = _hospital_model.predict_proba(features)
                    
                    test_passed = len(prediction) > 0 and len(prediction[0]) == 2
                    
                except Exception as test_error:
                    logger.warning(f"Health check test failed: {str(test_error)}")
                    test_passed = False
            
            status = 'healthy' if (models_ready and test_passed) else 'unhealthy'
            
            health_info = {
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'service': 'hospital_readmission_prediction',
                'version': _model_metadata.get('model_version', '1.0.0'),
                'models_loaded': models_ready,
                'test_passed': test_passed,
                'uptime': 'N/A',  # Would need to track actual uptime
                'dependencies': {
                    'database': 'not_checked',  # Would need actual DB health check
                    'model_files': 'available' if models_ready else 'missing'
                }
            }
            
            # Return appropriate HTTP status based on health
            status_code = 200 if status == 'healthy' else 503
            
            return health_info, status_code
            
        except Exception as e:
            logger.error(f"Error in hospital health check: {str(e)}")
            
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'service': 'hospital_readmission_prediction',
                'error': str(e)
            }, 500

def generate_hospital_recommendations(patient_data: Dict[str, Any], readmission_prob: float) -> List[str]:
    """
    Generate personalized recommendations based on patient data and readmission risk.
    
    Args:
        patient_data (Dict[str, Any]): Patient information
        readmission_prob (float): Predicted readmission probability
        
    Returns:
        List[str]: List of recommendation strings
    """
    recommendations = []
    
    # High-risk recommendations
    if readmission_prob > 0.7:
        recommendations.append("Schedule follow-up appointment within 7 days of discharge")
        recommendations.append("Consider discharge planning with social services")
        recommendations.append("Provide comprehensive medication reconciliation")
    
    # Age-based recommendations
    age = patient_data.get('age', 0)
    if age > 70:
        recommendations.append("Ensure fall prevention measures are in place")
        recommendations.append("Review polypharmacy and drug interactions")
    
    # Diabetes medication recommendations
    if patient_data.get('diabetesMed') == 'Yes':
        recommendations.append("Provide diabetes self-management education")
        recommendations.append("Ensure blood glucose monitoring supplies are available")
        
        if patient_data.get('A1Cresult') in ['>7', '>8']:
            recommendations.append("Consider diabetes medication adjustment")
    
    # Length of stay recommendations
    time_in_hospital = patient_data.get('time_in_hospital', 0)
    if time_in_hospital > 7:
        recommendations.append("Assess for deconditioning and mobility issues")
        recommendations.append("Consider physical therapy evaluation")
    
    # Emergency/outpatient utilization recommendations
    if patient_data.get('number_emergency', 0) > 2:
        recommendations.append("Address frequent emergency department use patterns")
        recommendations.append("Ensure adequate primary care follow-up")
    
    # Medication recommendations
    num_medications = patient_data.get('num_medications', 0)
    if num_medications > 10:
        recommendations.append("Conduct medication reconciliation and review")
        recommendations.append("Assess for potential drug interactions")
    
    # Diagnosis-based recommendations
    diagnoses_count = patient_data.get('number_diagnoses', 0)
    if diagnoses_count > 8:
        recommendations.append("Coordinate care across multiple specialties")
        recommendations.append("Ensure clear communication between providers")
    
    # Default recommendations for medium/high risk
    if readmission_prob > 0.5:
        recommendations.append("Provide clear discharge instructions and emergency contacts")
        recommendations.append("Arrange home health services if appropriate")
    
    # Ensure we have at least basic recommendations
    if not recommendations:
        recommendations.extend([
            "Follow standard discharge protocols",
            "Schedule appropriate follow-up care",
            "Provide patient education materials"
        ])
    
    return recommendations[:8]  # Limit to top 8 recommendations
