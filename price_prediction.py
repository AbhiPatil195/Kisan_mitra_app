"""
KisanMitra Price Prediction Module
This module handles market price prediction using historical data and machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class PricePredictionEngine:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """
        Prepares features for price prediction.
        
        Args:
            data (pd.DataFrame): Historical price data
            
        Returns:
            pd.DataFrame: Processed features
        """
        df = data.copy()
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Market features
        df['price_ma7'] = df['price'].rolling(window=7).mean()
        df['price_ma30'] = df['price'].rolling(window=30).mean()
        df['price_std7'] = df['price'].rolling(window=7).std()
        
        # Handle missing values
        df = df.fillna(method='ffill')
        
        return df
        
    def train_model(self, historical_data):
        """
        Trains the price prediction model.
        
        Args:
            historical_data (pd.DataFrame): Historical market data
            
        Returns:
            float: Model accuracy score
        """
        features = self.prepare_features(historical_data)
        
        X = features[['day_of_week', 'month', 'year', 'is_weekend',
                     'price_ma7', 'price_ma30', 'price_std7']]
        y = features['price']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        return self.model.score(X_scaled, y)
        
    def predict_price(self, current_data, days_ahead=7):
        """
        Predicts future prices.
        
        Args:
            current_data (pd.DataFrame): Current market data
            days_ahead (int): Number of days to predict
            
        Returns:
            dict: Predicted prices with confidence intervals
        """
        features = self.prepare_features(current_data)
        
        # Prepare prediction dates
        last_date = features['date'].max()
        future_dates = [last_date + timedelta(days=x) 
                       for x in range(1, days_ahead + 1)]
        
        predictions = []
        confidence_intervals = []
        
        for future_date in future_dates:
            # Create feature set for prediction
            pred_features = self._create_prediction_features(
                features, future_date)
            
            # Scale features
            pred_scaled = self.scaler.transform(pred_features)
            
            # Make prediction
            pred = self.model.predict(pred_scaled)[0]
            
            # Calculate confidence interval
            pred_std = np.std([tree.predict(pred_scaled)
                             for tree in self.model.estimators_])
            ci = (pred - 1.96 * pred_std, pred + 1.96 * pred_std)
            
            predictions.append(pred)
            confidence_intervals.append(ci)
            
        return {
            'dates': future_dates,
            'predictions': predictions,
            'confidence_intervals': confidence_intervals
        }
        
    def _create_prediction_features(self, historical_features, target_date):
        """
        Creates feature set for a future date.
        
        Args:
            historical_features (pd.DataFrame): Historical features
            target_date (datetime): Date to predict for
            
        Returns:
            pd.DataFrame: Features for prediction
        """
        features = pd.DataFrame({
            'day_of_week': [target_date.weekday()],
            'month': [target_date.month],
            'year': [target_date.year],
            'is_weekend': [1 if target_date.weekday() in [5, 6] else 0],
            'price_ma7': [historical_features['price'].tail(7).mean()],
            'price_ma30': [historical_features['price'].tail(30).mean()],
            'price_std7': [historical_features['price'].tail(7).std()]
        })
        
        return features 