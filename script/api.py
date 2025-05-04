from flask import Flask, request, jsonify
import pandas as pd
import json
import os
import sys

# Add the current directory to sys.path to import predict.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Data manipulation
import pandas as pd
import joblib
import os

# Scikit-learn base classes và scaler/encoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Oversampling
from imblearn.over_sampling import SMOTE

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=['Tenure', 'Monthly Charges', 'Total Charges']):
        self.feat_with_outliers = feat_with_outliers

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.feat_with_outliers).issubset(df.columns):
            Q1 = df[self.feat_with_outliers].quantile(0.25)
            Q3 = df[self.feat_with_outliers].quantile(0.75)
            IQR = Q3 - Q1
            condition = ~((df[self.feat_with_outliers] < (Q1 - 3 * IQR)) | 
                          (df[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)
            df_cleaned = df[condition].copy()
            return df_cleaned
        else:
            missing_features = list(set(self.feat_with_outliers) - set(df.columns))
            print(f"Missing features in dataframe: {missing_features}")
            return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=['Customer ID', 'Total Charges']):
        self.feature_to_drop = feature_to_drop

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if set(self.feature_to_drop).issubset(df.columns):
            df = df.drop(self.feature_to_drop, axis=1)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

# Min-Max scaling
class MinMaxScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features=['Tenure', 'Monthly Charges']):
        self.features = features
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.features])
        return self

    def transform(self, X):
        X[self.features] = self.scaler.transform(X[self.features])
        return X

# One-hot encoding
class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, features=['Internet Service', 'Payment Method', 'Multiple Lines', 'Streaming TV', 'Streaming Movies', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Contract']):
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        self.encoder.fit(X[self.features])
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X[self.features])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.features), index=X.index)
        X = X.drop(columns=self.features)
        return pd.concat([X, encoded_df], axis=1)

# Binary mapping (e.g., Yes/No, Y/N)
class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_features={
        'Gender': {'Female': 1, 'Male': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'Phone Service': {'Yes': 1, 'No': 0},
        'Paperless Billing': {'Yes': 1, 'No': 0},
        'Churn': {'Yes': 1, 'No': 0}
    }):
        self.mapping_features = mapping_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature, mapping in self.mapping_features.items():
            if feature in X.columns:
                X[feature] = X[feature].map(mapping)
        return X

# Imputer for missing values
class ImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', features=['Total Charges']):
        self.strategy = strategy
        self.features = features

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.fill_values = X[self.features].mean()
        elif self.strategy == 'median':
            self.fill_values = X[self.features].median()
        else:
            self.fill_values = X[self.features].mode().iloc[0]
        return self

    def transform(self, X):
        X[self.features] = X[self.features].fillna(self.fill_values)
        return X

# Oversampling minority class using SMOTE
class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if 'Churn' in df.columns:
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'Churn'], df['Churn'])
            df_bal = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal, columns=['Churn'])], axis=1)
            return df_bal
        else:
            print("Churn is not in the dataframe")
            return df

def predict_churn(data):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    data = data.rename(columns={
        "customerID": "Customer ID",
        "gender": "Gender",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "tenure": "Tenure",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "Contract": "Contract",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges"
    })

    # Define paths relative to the current directory
    pipeline_path = os.path.join(current_dir, 'saved_models', 'pipeline', 'pipeline_preprocessing.sav')
    model_path = os.path.join(current_dir, 'saved_models', 'xgboost', 'xgboost_model.sav')
    
    pipeline = joblib.load(pipeline_path)

    df = data.copy()

    # Add Churn column if it's not present (for prediction mode)
    if 'Churn' not in df.columns:
        df['Churn'] = 'No'  # Default value, will be ignored during prediction

    data_processed = pipeline.transform(df)

    X_data_processed = data_processed.drop(columns='Churn')
    
    model = joblib.load(model_path)

    y_pred = model.predict(X_data_processed)
    y_proba = model.predict_proba(X_data_processed)

    # Add prediction results to original dataframe
    data['Predicted Churn'] = y_pred
    data['Prob Non-Churn'] = y_proba[:, 0]
    data['Prob Churn'] = y_proba[:, 1]

    # # Map numerical predictions to Yes/No for better readability
    # data['Predicted Churn'] = data['Predicted Churn'].map({1: 'Yes', 0: 'No'})
    
    # # If Senior Citizen was numeric, map it back to Yes/No
    # if 'Senior Citizen' in data.columns and data['Senior Citizen'].dtype == 'int64':
    #     data['Senior Citizen'] = data['Senior Citizen'].map({1: 'Yes', 0: 'No'})

    return data

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        content_type = request.headers.get('Content-Type')
        
        if content_type == 'application/json':
            # Get JSON data from request
            json_data = request.json
            
            # Check if multiple customers or single customer
            if isinstance(json_data, list):
                # Multiple customers
                df = pd.DataFrame(json_data)
            else:
                # Single customer
                df = pd.DataFrame([json_data])
                
            # Make predictions using the predict_churn function
            result_df = predict_churn(df)
            
            # Convert DataFrame to dict for JSON response
            if len(result_df) == 1:
                # Return single customer result as object
                result = result_df.iloc[0].to_dict()
            else:
                # Return multiple customers result as array
                result = result_df.to_dict(orient='records')
            
            # Return the predictions
            return jsonify({
                'status': 'success',
                'predictions': result,
                'message': 'Churn prediction completed successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported Content-Type: {content_type}. Please send JSON data.'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'API is running'
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
