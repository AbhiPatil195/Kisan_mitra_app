import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def analyze_market_segmentation(data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Helper function for market segmentation analysis"""
    analyzer = KisanMitraBusinessAnalysis()
    df = pd.DataFrame(data)
    clusters = analyzer.perform_market_segmentation(df)
    reduced_data = analyzer.dimensionality_reduction(df)
    return {
        "clusters": clusters.tolist(),
        "reduced_data": reduced_data.tolist()
    }

def calculate_financial_metrics(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Helper function for financial metrics calculation"""
    analyzer = KisanMitraBusinessAnalysis()
    df = pd.DataFrame(data)
    return analyzer.calculate_financial_metrics(
        df['revenue'],
        df['cost'],
        df.get('investment')
    )

def analyze_time_series(data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Helper function for time series analysis"""
    analyzer = KisanMitraBusinessAnalysis()
    df = pd.DataFrame(data)
    ts_data = pd.Series(df['value'].values, index=pd.to_datetime(df['date']))
    return analyzer.time_series_analysis(ts_data)

class KisanMitraBusinessAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.knn = None
        self.pca = PCA(n_components=2)
        
    def perform_market_segmentation(self, data: pd.DataFrame) -> np.ndarray:
        """Perform market segmentation using K-means clustering"""
        # Scale the features
        scaled_data = self.scaler.fit_transform(data)
        
        # Apply PCA for dimensionality reduction
        reduced_data = self.pca.fit_transform(scaled_data)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(reduced_data)
        
        return clusters
    
    def customer_classification(self, X_train, y_train, X_test):
        """
        Classify farmers/customers using KNN
        Parameters:
        - X_train: Training features (e.g., farm size, crop type, income)
        - y_train: Training labels (e.g., service subscription level)
        - X_test: Test features for prediction
        """
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.knn.fit(X_train, y_train)
        predictions = self.knn.predict(X_test)
        return predictions
    
    def dimensionality_reduction(self, data: pd.DataFrame) -> np.ndarray:
        """Reduce dimensionality of the data using PCA"""
        scaled_data = self.scaler.fit_transform(data)
        return self.pca.fit_transform(scaled_data)
    
    def time_series_analysis(self, data: pd.Series) -> Dict[str, np.ndarray]:
        """Perform time series analysis and forecasting"""
        # Decompose the time series
        decomposition = seasonal_decompose(data, period=12)
        
        # Fit ARIMA model for forecasting
        model = ARIMA(data, order=(1,1,1))
        results = model.fit()
        
        # Generate forecast for next 12 periods
        forecast = results.forecast(steps=12)
        
        return {
            "trend": decomposition.trend.values,
            "seasonal": decomposition.seasonal.values,
            "forecast": forecast.values
        }
    
    def calculate_financial_metrics(self, revenue: pd.Series, costs: pd.Series, 
                                  investments: pd.Series = None) -> Dict[str, float]:
        """Calculate various financial metrics"""
        # Calculate basic metrics
        gross_profit = revenue - costs
        profit_margin = (gross_profit / revenue) * 100
        
        # Calculate ROI if investment data is provided
        roi = None
        if investments is not None and not investments.empty:
            net_profit = gross_profit - investments
            roi = (net_profit / investments) * 100
        
        # Calculate customer metrics (example values)
        marketing_cost = 50000  # Example fixed cost
        num_customers = 100     # Example number of customers
        customer_lifespan = 12  # months
        
        cac = marketing_cost / num_customers
        clv = (revenue.mean() / num_customers) * customer_lifespan
        mrr = revenue.mean()  # Monthly Recurring Revenue
        
        return {
            "gross_profit": float(gross_profit.mean()),
            "profit_margin": float(profit_margin.mean()),
            "roi": float(roi.mean()) if roi is not None else None,
            "cac": float(cac),
            "clv": float(clv),
            "mrr": float(mrr)
        }
    
    def visualize_results(self, data: np.ndarray, plot_type: str = 'clusters', **kwargs) -> None:
        """Visualize analysis results"""
        plt.figure(figsize=(12, 6))
        
        if plot_type == 'clusters':
            clusters = kwargs.get('clusters')
            plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
            plt.title("Market Segments Analysis")
            plt.xlabel("Feature 1 (PCA)")
            plt.ylabel("Feature 2 (PCA)")
            
        elif plot_type == 'time_series':
            plt.plot(data, label='Actual')
            if 'forecast' in kwargs:
                plt.plot(kwargs['forecast'], label='Forecast', linestyle='--')
            plt.title("Market Trend Analysis")
            plt.xlabel("Time Period")
            plt.ylabel("Value")
            plt.legend()
            
        elif plot_type == 'financial':
            metrics = kwargs.get('metrics', {})
            plt.bar(metrics.keys(), metrics.values())
            plt.title("Financial Metrics")
            plt.xticks(rotation=45)
            plt.ylabel("Value (₹)")
            
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    """
    Example usage of the business analysis module
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample market data (features: farm size, income, tech adoption, age)
    market_data = pd.DataFrame({
        'farm_size': np.random.uniform(1, 50, n_samples),
        'income': np.random.uniform(10000, 100000, n_samples),
        'tech_adoption': np.random.uniform(0, 1, n_samples),
        'age': np.random.uniform(25, 65, n_samples)
    })
    
    # Sample time series data (monthly service usage)
    dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
    time_series_data = pd.Series(
        np.random.normal(1000, 100, 24) + np.linspace(0, 500, 24),  # Trending upward
        index=dates
    )
    
    # Initialize analysis
    analyzer = KisanMitraBusinessAnalysis()
    
    # Market segmentation
    clusters = analyzer.perform_market_segmentation(market_data)
    reduced_data = analyzer.dimensionality_reduction(market_data)
    
    # Time series analysis
    ts_results = analyzer.time_series_analysis(time_series_data)
    
    # Financial analysis
    revenue = pd.Series(np.random.uniform(1000, 5000, n_samples))
    costs = pd.Series(np.random.uniform(500, 3000, n_samples))
    investments = pd.Series(np.random.uniform(10000, 50000, 10))
    financial_metrics = analyzer.calculate_financial_metrics(revenue, costs, investments)
    
    # Visualize results
    analyzer.visualize_results(reduced_data, 'clusters', clusters=clusters)
    analyzer.visualize_results(time_series_data, 'time_series', forecast=ts_results['forecast'])
    analyzer.visualize_results(None, 'financial', metrics=financial_metrics)

if __name__ == "__main__":
    main() 