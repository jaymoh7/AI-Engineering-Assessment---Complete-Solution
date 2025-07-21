# src/student_dropout/data_preprocessing.py
"""
Student Dropout Prediction - Data Preprocessing Module

This module handles data cleaning, transformation, and preparation
for the student dropout prediction model.

Author: AI Engineering Assessment
Date: 2025-07-21
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudentDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for student dropout prediction.
    
    This class handles missing values, outliers, feature scaling, and
    categorical encoding for student academic data.
    """
    
    def __init__(self):
        """Initialize preprocessor with default parameters."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numerical_imputer = KNNImputer(n_neighbors=5)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.is_fitted = False
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load student data from CSV file with error handling.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Empty file: {file_path}")
            raise
            
    def identify_data_types(self, df: pd.DataFrame) -> Dict[str, list]:
        """
        Categorize columns by data type for appropriate preprocessing.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, list]: Dictionary with categorized column names
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if present
        if 'dropout' in numerical_cols:
            numerical_cols.remove('dropout')
        if 'dropout' in categorical_cols:
            categorical_cols.remove('dropout')
            
        logger.info(f"Numerical columns: {len(numerical_cols)}")
        logger.info(f"Categorical columns: {len(categorical_cols)}")
        
        return {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }
    
    def handle_missing_values(self, df: pd.DataFrame, columns_dict: Dict[str, list]) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies for different data types.
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns_dict (Dict[str, list]): Categorized column names
            
        Returns:
            pd.DataFrame: DataFrame with imputed missing values
        """
        df_processed = df.copy()
        
        # Log missing value statistics
        missing_stats = df.isnull().sum()
        logger.info(f"Missing values before imputation:\n{missing_stats[missing_stats > 0]}")
        
        # Impute numerical columns using KNN
        if columns_dict['numerical']:
            logger.info("Imputing numerical columns using KNN imputation")
            df_processed[columns_dict['numerical']] = self.numerical_imputer.fit_transform(
                df_processed[columns_dict['numerical']]
            )
        
        # Impute categorical columns using mode
        if columns_dict['categorical']:
            logger.info("Imputing categorical columns using mode")
            df_processed[columns_dict['categorical']] = self.categorical_imputer.fit_transform(
                df_processed[columns_dict['categorical']]
            )
        
        # Log final missing value statistics
        final_missing = df_processed.isnull().sum().sum()
        logger.info(f"Total missing values after imputation: {final_missing}")
        
        return df_processed
    
    def detect_outliers(self, df: pd.DataFrame, numerical_cols: list, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers in numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_cols (list): List of numerical column names
            method (str): Outlier detection method ('iqr' or 'zscore')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        df_processed = df.copy()
        outlier_counts = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                # Interquartile Range method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Count outliers
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_counts[col] = outliers.sum()
                
                # Cap outliers instead of removing them
                df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
                outlier_counts[col] = outliers.sum()
                
                # Replace outliers with median
                median_val = df[col].median()
                df_processed.loc[outliers, col] = median_val
        
        logger.info(f"Outliers detected and handled: {outlier_counts}")
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
        """
        Encode categorical variables using appropriate techniques.
        
        Args:
            df (pd.DataFrame): Input dataframe
            categorical_cols (list): List of categorical column names
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical variables
        """
        df_processed = df.copy()
        
        for col in categorical_cols:
            # Check cardinality to decide encoding strategy
            unique_values = df[col].nunique()
            
            if unique_values <= 10:  # Low cardinality - use one-hot encoding
                logger.info(f"One-hot encoding {col} ({unique_values} unique values)")
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                df_processed.drop(col, axis=1, inplace=True)
                
            else:  # High cardinality - use label encoding
                logger.info(f"Label encoding {col} ({unique_values} unique values)")
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df_processed[col] = self.label_encoders[col].transform(df[col])
        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            numerical_cols (list): List of numerical column names
            
        Returns:
            pd.DataFrame: DataFrame with scaled numerical features
        """
        df_processed = df.copy()
        
        if numerical_cols:
            logger.info(f"Scaling {len(numerical_cols)} numerical features")
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
        
        return df_processed
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features based on domain knowledge.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: DataFrame with additional derived features
        """
        df_processed = df.copy()
        
        # Create GPA trend feature (if multiple semester GPAs available)
        if all(col in df.columns for col in ['gpa_sem1', 'gpa_sem2']):
            df_processed['gpa_trend'] = df_processed['gpa_sem2'] - df_processed['gpa_sem1']
            logger.info("Created GPA trend feature")
        
        # Create engagement score (if engagement metrics available)
        engagement_cols = [col for col in df.columns if 'engagement' in col.lower()]
        if engagement_cols:
            df_processed['engagement_score'] = df_processed[engagement_cols].mean(axis=1)
            logger.info("Created engagement score feature")
        
        # Create attendance rate feature
        if 'total_classes' in df.columns and 'attended_classes' in df.columns:
            df_processed['attendance_rate'] = (
                df_processed['attended_classes'] / df_processed['total_classes']
            )
            logger.info("Created attendance rate feature")
        
        # Create credit load ratio
        if 'enrolled_credits' in df.columns and 'completed_credits' in df.columns:
            df_processed['completion_rate'] = (
                df_processed['completed_credits'] / df_processed['enrolled_credits']
            )
            logger.info("Created completion rate feature")
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'dropout') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed features and target
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Separate features and target
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df.copy()
            y = None
            
        # Identify data types
        columns_dict = self.identify_data_types(X)
        
        # Preprocessing steps
        X = self.handle_missing_values(X, columns_dict)
        X = self.detect_outliers(X, columns_dict['numerical'])
        X = self.create_derived_features(X)
        
        # Update column categorization after feature creation
        columns_dict = self.identify_data_types(X)
        
        X = self.encode_categorical_variables(X, columns_dict['categorical'])
        X = self.scale_features(X, columns_dict['numerical'])
        
        self.is_fitted = True
        logger.info(f"Preprocessing completed. Final shape: {X.shape}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to new data (inference time).
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Processed features
            
        Raises:
            ValueError: If preprocessor hasn't been fitted
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Applying preprocessing to new data")
        X = df.copy()
        
        # Apply same preprocessing steps (without fitting)
        columns_dict = self.identify_data_types(X)
        
        # Note: In production, you'd need to handle new categorical values
        # and ensure column consistency more carefully
        
        return X


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        'student_id': range(1000),
        'gpa_sem1': np.random.normal(3.0, 0.5, 1000),
        'gpa_sem2': np.random.normal(3.1, 0.6, 1000),
        'attendance_rate': np.random.beta(8, 2, 1000),  # Skewed towards higher values
        'engagement_score': np.random.gamma(2, 2, 1000),
        'major': np.random.choice(['Engineering', 'Business', 'Arts', 'Science'], 1000),
        'financial_aid': np.random.choice(['Yes', 'No'], 1000),
        'dropout': np.random.choice([0, 1], 1000, p=[0.85, 0.15])  # 15% dropout rate
    }
    
    # Add some missing values for testing
    df = pd.DataFrame(sample_data)
    df.loc[np.random.choice(df.index, 50, replace=False), 'gpa_sem1'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'major'] = np.nan
    
    # Test preprocessing
    preprocessor = StudentDataPreprocessor()
    X_processed, y = preprocessor.fit_transform(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Columns: {list(X_processed.columns)}")
