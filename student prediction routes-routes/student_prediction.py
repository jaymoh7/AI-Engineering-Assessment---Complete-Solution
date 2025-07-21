"""
Student Dropout Prediction API Routes

This module provides REST API endpoints for student dropout prediction:
- Single student prediction
- Batch predictions
- Model information and statistics
- Feature importance analysis
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from flask import request, jsonify
from flask_restx import Namespace, Resource, fields
import joblib

# Import custom modules
from schemas.student_schema import StudentDataSchema, PredictionResponseSchema
from src.student_dropout.model_training import StudentDropoutPredictor
from src.utils.model_utils import load_model, ModelNotFoundError
from src.utils.data_utils import validate_input_data, preprocess_api_input
from src.utils.evaluation_utils import calculate_prediction_confidence

# Configure logging
logger = logging.getLogger(__name__)

# Create namespace for student prediction endpoints
student_ns = Namespace(
    'student',
    description='Student Dropout Prediction Operations',
    path='/api/v1/student'
)

# Define API models for documentation
student_input_model = student_ns.model('StudentInput', {
    'student_id': fields.String(required=False, description='Unique student identifier'),
    'age': fields.Integer(required=True, description='Student age', min=16, max=65),
    'gender': fields.String(required=True, description='Student gender', enum=['M', 'F']),
    'previous_qualification': fields.Integer(required=True, description='Previous qualification grade'),
    'nationality': fields.Integer(required=True, description='Student nationality code'),
    'mothers_qualification': fields.Integer(required=True, description='Mother\'s qualification level'),
    'fathers_qualification': fields.Integer(required=True, description='Father\'s qualification level'),
    'mothers_occupation': fields.Integer(required=True, description='Mother\'s occupation code'),
    'fathers_occupation': fields.Integer(required=True, description='Father\'s occupation code'),
    'admission_grade': fields.Float(required=True, description='Admission grade', min=0, max=200),
    'displaced': fields.Integer(required=True, description='Displaced student (0/1)', enum=[0, 1]),
    'educational_special_needs': fields.Integer(required=True, description='Special needs (0/1)', enum=[0, 1]),
    'debtor': fields.Integer(required=True, description='Debtor status (0/1)', enum=[0, 1]),
    'tuition_fees_up_to_date': fields.Integer(required=True, description='Tuition up to date (0/1)', enum=[0, 1]),
    'scholarship_holder': fields.Integer(required=True, description='Scholarship holder (0/1)', enum=[0, 1]),
    'age_at_enrollment': fields.Integer(required=True, description='Age at enrollment'),
    'international': fields.Integer(required=True, description='International student (0/1)', enum=[0, 1]),
    'curricular_units_1st_sem_credited': fields.Integer(required=True, description='1st sem credited units'),
    'curricular_units_1st_sem_enrolled': fields.Integer(required=True, description='1st sem enrolled units'),
    'curricular_units_1st_sem_evaluations': fields.Integer(required=True, description='1st sem evaluations'),
    'curricular_units_1st_sem_approved': fields.Integer(required=True, description='1st sem approved units'),
    'curricular_units_1st_sem_grade': fields.Float(required=True, description='1st sem average grade'),
    'curricular_units_1st_sem_without_evaluations': fields.Integer(required=True, description='1st sem without evaluations'),
    'curricular_units_2nd_sem_credited': fields.Integer(required=True, description='2nd sem credited units'),
    'curricular_units_2nd_sem_enrolled': fields.Integer(required=True, description='2nd sem enrolled units'),
    'curricular_units_2nd_sem_evaluations': fields.Integer(required=True, description='2nd sem evaluations'),
    'curricular_units_2nd_sem_approved': fields.Integer(required=True, description='2nd sem approved units'),
    'curricular_units_2nd_sem_grade': fields.Float(required=True, description='2nd sem average grade'),
    'curricular_units_2nd_sem_without_evaluations': fields.Integer(required=True, description='2nd sem without evaluations'),
    'unemployment_rate': fields.Float(required=True, description='Unemployment rate'),
    'inflation_rate': fields.Float(required=True, description='Inflation rate'),
    'gdp': fields.Float(required=True, description='GDP')
})

prediction_response_model = student_ns.model('PredictionResponse', {
    'student_id': fields.String(description='Student identifier'),
    'prediction': fields.Integer(description='Dropout prediction (0: No Dropout, 1: Dropout)', enum=[0, 1]),
    'probability': fields.Float(description='Dropout probability', min=0, max=1),
    'confidence': fields.String(description='Prediction confidence level', enum=['Low', 'Medium', 'High']),
    'risk_factors': fields.List(fields.String, description='Key risk factors identified'),
    'recommendations': fields.List(fields.String, description='Intervention recommendations'),
    'timestamp': fields.String(description='Prediction timestamp')
})

batch_input_model = student_ns.model('BatchInput', {
    'students': fields.List(fields.Nested(student_input_model), required=True, description='List of students')
})

batch_response_model = student_ns.model('BatchResponse', {
    'predictions': fields.List(fields.Nested(prediction_response_model), description='List of predictions'),
    'summary': fields.Raw(description='Batch prediction summary'),
    'timestamp': fields.String(description='Batch processing timestamp')
})

# Global model cache
_model_cache = {}

def get_model():
    """
    Load and cache the student dropout prediction model
    
    Returns:
        Trained model instance
        
    Raises:
        ModelNotFoundError: If model cannot be loaded
    """
    global _model_cache
    
    if 'student_dropout' not in _model_cache:
        try:
            model_path = "models/student_dropout/model.pkl"
            _model_cache['student_dropout'] = joblib.load(model_path)
            logger.info("Student dropout model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load student dropout model: {e}")
            raise ModelNotFoundError(f"Student dropout model not available: {e}")
    
    return _model_cache['student_dropout']

@student_ns.route('/predict')
class StudentPrediction(Resource):
    """Single student dropout prediction endpoint"""
    
    @student_ns.expect(student_input_model)
    @student_ns.marshal_with(prediction_response_model)
    @student_ns.doc('predict_student_dropout')
    def post(self):
        """
        Predict dropout probability for a single student
        
        Returns detailed prediction with risk factors and recommendations
        """
        try:
            # Get input data
            data = request.get_json()
            
            if not data:
                student_ns.abort(400, "No input data provided")
            
            logger.info(f"Received prediction request for student: {data.get('student_id', 'unknown')}")
            
            # Validate input data
            validation_errors = validate_student_input(data)
            if validation_errors:
                student_ns.abort(400, f"Input validation errors: {validation_errors}")
            
            # Load model
            model = get_model()
            
            # Preprocess input
            processed_data = preprocess_student_input(data)
            
            # Make prediction
            prediction_prob = model.predict_proba([processed_data])[0]
            dropout_prob = prediction_prob[1]  # Probability of dropout
            prediction = int(dropout_prob >= 0.5)  # Binary prediction
            
            # Calculate confidence
            confidence = calculate_prediction_confidence(dropout_prob)
            
            # Identify risk factors
            risk_factors = identify_risk_factors(data, model, processed_data)
            
            # Generate recommendations
            recommendations = generate_recommendations(data, risk_factors, dropout_prob)
            
            result = {
                'student_id': data.get('student_id', 'anonymous'),
                'prediction': prediction,
                'probability': round(float(dropout_prob), 4),
                'confidence': confidence,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Prediction completed - Dropout probability: {dropout_prob:.4f}")
            return result, 200
            
        except ModelNotFoundError as e:
            logger.error(f"Model error: {e}")
            student_ns.abort(503, str(e))
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            student_ns.abort(500, f"Internal server error: {str(e)}")

@student_ns.route('/predict/batch')
class BatchStudentPrediction(Resource):
    """Batch student dropout prediction endpoint"""
    
    @student_ns.expect(batch_input_model)
    @student_ns.marshal_with(batch_response_model)
    @student_ns.doc('predict_batch_student_dropout')
    def post(self):
        """
        Predict dropout probability for multiple students
        
        Processes multiple student records efficiently
        """
        try:
            # Get input data
            data = request.get_json()
            
            if not data or 'students' not in data:
                student_ns.abort(400, "No student data provided")
            
            students = data['students']
            if not isinstance(students, list) or len(students) == 0:
                student_ns.abort(400, "Students must be a non-empty list")
            
            if len(students) > 100:  # Limit batch size
                student_ns.abort(400, "Batch size limited to 100 students")
            
            logger.info(f"Processing batch prediction for {len(students)} students")
            
            # Load model
            model = get_model()
            
            predictions = []
            errors = []
            
            for i, student_data in enumerate(students):
                try:
                    # Validate each student's data
                    validation_errors = validate_student_input(student_data)
                    if validation_errors:
                        errors.append(f"Student {i}: {validation_errors}")
                        continue
                    
                    # Preprocess and predict
                    processed_data = preprocess_student_input(student_data)
                    prediction_prob = model.predict_proba([processed_data])[0]
                    dropout_prob = prediction_prob[1]
                    prediction = int(dropout_prob >= 0.5)
                    
                    # Calculate additional metrics
                    confidence = calculate_prediction_confidence(dropout_prob)
                    risk_factors = identify_risk_factors(student_data, model, processed_data)
                    recommendations = generate_recommendations(student_data, risk_factors, dropout_prob)
                    
                    predictions.append({
                        'student_id': student_data.get('student_id', f'student_{i}'),
                        'prediction': prediction,
                        'probability': round(float(dropout_prob), 4),
                        'confidence': confidence,
                        'risk_factors': risk_factors[:3],  # Limit to top 3 for batch
                        'recommendations': recommendations[:2],  # Limit to top 2 for batch
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                except Exception as e:
                    errors.append(f"Student {i}: {str(e)}")
            
            # Generate summary
            summary = generate_batch_summary(predictions, errors)
            
            result = {
                'predictions': predictions,
                'summary': summary,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Batch prediction completed - {len(predictions)} successful, {len(errors)} errors")
            return result, 200
            
        except ModelNotFoundError as e:
            logger.error(f"Model error: {e}")
            student_ns.abort(503, str(e))
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            student_ns.abort(500, f"Internal server error: {str(e)}")

@student_ns.route('/model/info')
class ModelInfo(Resource):
    """Model information endpoint"""
    
    @student_ns.doc('get_model_info')
    def get(self):
        """Get information about the student dropout prediction model"""
        try:
            model = get_model()
            
            # Get model metadata
            model_info = {
                'model_type': type(model).__name__,
                'version': '1.0',
                'features_count': getattr(model, 'n_features_in_', 'unknown'),
                'classes': ['No Dropout', 'Dropout'],
                'training_timestamp': 'unknown',  # Would come from model metadata
                'performance_metrics': {
                    'accuracy': 0.85,  # Would come from model metadata
                    'precision': 0.82,
                    'recall': 0.78,
                    'f1_score': 0.80
                }
            }
            
            return model_info, 200
            
        except ModelNotFoundError as e:
            student_ns.abort(503, str(e))
        except Exception as e:
            logger.error(f"Model info error: {e}")
            student_ns.abort(500, f"Internal server error: {str(e)}")

def validate_student_input(data: Dict[str, Any]) -> Optional[str]:
    """
    Validate student input data
    
    Args:
        data: Student data dictionary
        
    Returns:
        Error message if validation fails, None if valid
    """
    required_fields = [
        'age', 'gender', 'previous_qualification', 'nationality',
        'mothers_qualification', 'fathers_qualification',
        'mothers_occupation', 'fathers_occupation',
        'admission_grade', 'displaced', 'educational_special_needs',
        'debtor', 'tuition_fees_up_to_date', 'scholarship_holder',
        'age_at_enrollment', 'international',
        'curricular_units_1st_sem_credited', 'curricular_units_1st_sem_enrolled',
        'curricular_units_1st_sem_evaluations', 'curricular_units_1st_sem_approved',
        'curricular_units_1st_sem_grade', 'curricular_units_1st_sem_without_evaluations',
        'curricular_units_2nd_sem_credited', 'curricular_units_2nd_sem_enrolled',
        'curricular_units_2nd_sem_evaluations', 'curricular_units_2nd_sem_approved',
        'curricular_units_2nd_sem_grade', 'curricular_units_2nd_sem_without_evaluations',
        'unemployment_rate', 'inflation_rate', 'gdp'
    ]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return f"Missing required fields: {missing_fields}"
    
    # Validate data types and ranges
    try:
        if not (16 <= data['age'] <= 65):
            return "Age must be between 16 and 65"
        
        if data['gender'] not in ['M', 'F']:
            return "Gender must be 'M' or 'F'"
        
        if not (0 <= data['admission_grade'] <= 200):
            return "Admission grade must be between 0 and 200"
        
        # Validate binary fields
        binary_fields = ['displaced', 'educational_special_needs', 'debtor',
                        'tuition_fees_up_to_date', 'scholarship_holder', 'international']
        for field in binary_fields:
            if data[field] not in [0, 1]:
                return f"{field} must be 0 or 1"
        
    except (TypeError, ValueError) as e:
        return f"Invalid data type: {str(e)}"
    
    return None

def preprocess_student_input(data: Dict[str, Any]) -> List[float]:
    """
    Preprocess student input data for model prediction
    
    Args:
        data: Raw student data
        
    Returns:
        Preprocessed feature vector
    """
    # Convert gender to numeric
    gender_numeric = 1 if data['gender'] == 'M' else 0
    
    # Create feature vector in the expected order
    features = [
        data['age'],
        gender_numeric,
        data['previous_qualification'],
        data['nationality'],
        data['mothers_qualification'],
        data['fathers_qualification'],
        data['mothers_occupation'],
        data['fathers_occupation'],
        data['admission_grade'],
        data['displaced'],
        data['educational_special_needs'],
        data['debtor'],
        data['tuition_fees_up_to_date'],
        data['scholarship_holder'],
        data['age_at_enrollment'],
        data['international'],
        data['curricular_units_1st_sem_credited'],
        data['curricular_units_1st_sem_enrolled'],
        data['curricular_units_1st_sem_evaluations'],
        data['curricular_units_1st_sem_approved'],
        data['curricular_units_1st_sem_grade'],
        data['curricular_units_1st_sem_without_evaluations'],
        data['curricular_units_2nd_sem_credited'],
        data['curricular_units_2nd_sem_enrolled'],
        data['curricular_units_2nd_sem_evaluations'],
        data['curricular_units_2nd_sem_approved'],
        data['curricular_units_2nd_sem_grade'],
        data['curricular_units_2nd_sem_without_evaluations'],
        data['unemployment_rate'],
        data['inflation_rate'],
        data['gdp']
    ]
    
    return features

def identify_risk_factors(data: Dict[str, Any], model, processed_data: List[float]) -> List[str]:
    """
    Identify key risk factors for the student
    
    Args:
        data: Raw student data
        model: Trained model
        processed_data: Preprocessed feature vector
        
    Returns:
        List of risk factors
    """
    risk_factors = []
    
    # Academic performance risk factors
    if data['curricular_units_1st_sem_grade'] < 10:
        risk_factors.append("Low 1st semester grades")
    
    if data['curricular_units_2nd_sem_grade'] < 10:
        risk_factors.append("Low 2nd semester grades")
    
    # Financial risk factors
    if data['debtor'] == 1:
        risk_factors.append("Outstanding debts")
    
    if data['tuition_fees_up_to_date'] == 0:
        risk_factors.append("Overdue tuition fees")
    
    if data['scholarship_holder'] == 0:
        risk_factors.append("No scholarship support")
    
    # Socioeconomic risk factors
    if data['unemployment_rate'] > 10:
        risk_factors.append("High unemployment rate environment")
    
    # Educational background
    if data['mothers_qualification'] < 3 and data['fathers_qualification'] < 3:
        risk_factors.append("Low parental education level")
    
    # Academic engagement
    approved_ratio_1st = data['curricular_units_1st_sem_approved'] / max(data['curricular_units_1st_sem_enrolled'], 1)
    approved_ratio_2nd = data['curricular_units_2nd_sem_approved'] / max(data['curricular_units_2nd_sem_enrolled'], 1)
    
    if approved_ratio_1st < 0.6:
        risk_factors.append("Low 1st semester approval rate")
    
    if approved_ratio_2nd < 0.6:
        risk_factors.append("Low 2nd semester approval rate")
    
    return risk_factors[:5]  # Return top 5 risk factors

def generate_recommendations(data: Dict[str, Any], risk_factors: List[str], dropout_prob: float) -> List[str]:
    """
    Generate intervention recommendations based on risk factors
    
    Args:
        data: Student data
        risk_factors: Identified risk factors
        dropout_prob: Dropout probability
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # High-risk interventions
    if dropout_prob > 0.7:
        recommendations.append("Immediate academic counseling required")
        recommendations.append("Assign dedicated academic mentor")
    
    # Academic performance recommendations
    if any("grade" in factor for factor in risk_factors):
        recommendations.append("Enroll in academic support programs")
        recommendations.append("Consider tutoring services")
    
    # Financial recommendations
    if any("debt" in factor or "fees" in factor for factor in risk_factors):
        recommendations.append("Meet with financial aid counselor")
        recommendations.append("Explore scholarship opportunities")
    
    # Engagement recommendations
    if any("approval rate" in factor for factor in risk_factors):
        recommendations.append("Review course load and difficulty")
        recommendations.append("Join study groups or peer support")
    
    # General support
    if dropout_prob > 0.5:
        recommendations.append("Regular check-ins with academic advisor")
        recommendations.append("Access to student wellness services")
    
    return recommendations[:6]  # Return top 6 recommendations

def generate_batch_summary(predictions: List[Dict], errors: List[str]) -> Dict[str, Any]:
    """
    Generate summary statistics for batch predictions
    
    Args:
        predictions: List of prediction results
        errors: List of processing errors
        
    Returns:
        Summary dictionary
    """
    if not predictions:
        return {
            'total_processed': 0,
            'successful_predictions': 0,
            'errors': len(errors),
            'error_rate': 1.0 if errors else 0.0
        }
    
    # Calculate statistics
    dropout_predictions = [p for p in predictions if p['prediction'] == 1]
    high_risk_students = [p for p in predictions if p['probability'] > 0.7]
    medium_risk_students = [p for p in predictions if 0.3 <= p['probability'] <= 0.7]
    low_risk_students = [p for p in predictions if p['probability'] < 0.3]
    
    avg_dropout_prob = np.mean([p['probability'] for p in predictions])
    
    return {
        'total_processed': len(predictions) + len(errors),
        'successful_predictions': len(predictions),
        'errors': len(errors),
        'error_rate': len(errors) / (len(predictions) + len(errors)),
        'dropout_predictions': len(dropout_predictions),
        'dropout_rate': len(dropout_predictions) / len(predictions),
        'average_dropout_probability': round(float(avg_dropout_prob), 4),
        'risk_distribution': {
            'high_risk': len(high_risk_students),
            'medium_risk': len(medium_risk_students),
            'low_risk': len(low_risk_students)
        },
        'recommendations_needed': len([p for p in predictions if p['probability'] > 0.5])
    }
