"""
Flask API Application for AI Engineering Assessment
Provides REST endpoints for student dropout and hospital readmission predictions

This module sets up the main Flask application with:
- API documentation using Flask-RESTX
- Error handling and logging
- CORS configuration
- Health checks and monitoring endpoints
"""

import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_restx import Api, Resource
from flask_cors import CORS
import yaml
import joblib
import pandas as pd
from typing import Dict, Any, Optional

# Import custom modules
from routes.student_prediction import student_ns
from routes.hospital_prediction import hospital_ns
from src.utils.data_utils import validate_input_data, log_prediction_request
from src.utils.model_utils import load_model, ModelNotFoundError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_path: str = 'config/config.yaml') -> Flask:
    """
    Create and configure Flask application
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        app.config.update(config)
        logger.info(f"Configuration loaded from {config_path}")
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        app.config['api'] = {'host': '0.0.0.0', 'port': 5000, 'debug': False}
    
    # Enable CORS for cross-origin requests
    CORS(app, origins="*")
    
    # Initialize Flask-RESTX API with documentation
    api = Api(
        app,
        version='1.0',
        title='AI Engineering Assessment API',
        description='REST API for Student Dropout and Hospital Readmission Predictions',
        doc='/docs/',
        authorizations={
            'Bearer': {
                'type': 'apiKey',
                'in': 'header',
                'name': 'Authorization',
                'description': 'Add "Bearer " before your token'
            }
        }
    )
    
    # Register namespaces (route groups)
    api.add_namespace(student_ns, path='/api/v1/student')
    api.add_namespace(hospital_ns, path='/api/v1/hospital')
    
    # Global error handlers
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors"""
        return jsonify({
            'error': 'Bad Request',
            'message': 'Invalid input data or request format',
            'timestamp': datetime.utcnow().isoformat()
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle not found errors"""
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'timestamp': datetime.utcnow().isoformat()
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server errors"""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred. Please try again later.',
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    
    @app.errorhandler(ModelNotFoundError)
    def model_not_found(error):
        """Handle model not found errors"""
        return jsonify({
            'error': 'Model Not Found',
            'message': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }), 503
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """
        Health check endpoint for monitoring
        Returns system status and model availability
        """
        try:
            # Check if models are available
            student_model_status = check_model_health('student_dropout')
            hospital_model_status = check_model_health('hospital_readmission')
            
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0',
                'models': {
                    'student_dropout': student_model_status,
                    'hospital_readmission': hospital_model_status
                }
            }
            
            # If any model is unhealthy, mark overall status as degraded
            if not all([student_model_status['available'], hospital_model_status['available']]):
                health_status['status'] = 'degraded'
            
            return jsonify(health_status), 200
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }), 503
    
    # Metrics endpoint for monitoring
    @app.route('/metrics', methods=['GET'])
    def metrics():
        """
        Metrics endpoint for Prometheus monitoring
        Returns basic application metrics
        """
        # This would typically integrate with prometheus_client
        # For now, return basic metrics
        return jsonify({
            'predictions_total': get_prediction_count(),
            'models_loaded': count_loaded_models(),
            'uptime_seconds': get_uptime_seconds(),
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    return app

def check_model_health(model_type: str) -> Dict[str, Any]:
    """
    Check if a specific model is available and healthy
    
    Args:
        model_type: Type of model ('student_dropout' or 'hospital_readmission')
        
    Returns:
        Dictionary with model health status
    """
    try:
        model_path = f"models/{model_type}/model.pkl"
        if os.path.exists(model_path):
            # Try to load the model
            model = joblib.load(model_path)
            return {
                'available': True,
                'loaded_at': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
                'model_type': type(model).__name__
            }
        else:
            return {
                'available': False,
                'error': 'Model file not found'
            }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }

def get_prediction_count() -> int:
    """Get total number of predictions made (placeholder implementation)"""
    # This would typically read from a database or metrics store
    return 0

def count_loaded_models() -> int:
    """Count number of loaded models"""
    count = 0
    for model_type in ['student_dropout', 'hospital_readmission']:
        if check_model_health(model_type)['available']:
            count += 1
    return count

def get_uptime_seconds() -> int:
    """Get application uptime in seconds (placeholder implementation)"""
    # This would typically track actual uptime
    return 0

# Request logging middleware
@app.before_request
def log_request_info():
    """Log request information for monitoring and debugging"""
    logger.info(f"{request.method} {request.url} - IP: {request.remote_addr}")
    
    # Log request payload for prediction endpoints
    if request.endpoint and 'predict' in request.endpoint:
        logger.info(f"Prediction request to {request.endpoint}")

@app.after_request
def log_response_info(response):
    """Log response information"""
    logger.info(f"Response status: {response.status_code}")
    return response

if __name__ == '__main__':
    # Create application
    app = create_app()
    
    # Get configuration
    config = app.config.get('api', {})
    host = config.get('host', '0.0.0.0')
    port = config.get('port', 5000)
    debug = config.get('debug', False)
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs/")
    
    # Run the application
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True  # Enable multi-threading for better performance
    )
