import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from typing import Dict, Any, List, Optional

# Base URL
BASE_URL = "http://127.0.0.1:8000"

def print_section(title: str) -> None:
    """Print a section header for test output
    
    Args:
        title: Title of the test section
    """
    print("\n" + "="*50)
    print(f"Testing: {title}")
    print("="*50)

def test_root() -> None:
    """Test the root endpoint"""
    print_section("Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error testing root endpoint: {str(e)}")
        raise

def test_health_check() -> None:
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to Kisan Mitra API"}
        print("Health check passed successfully")
    except (requests.exceptions.RequestException, AssertionError) as e:
        print(f"Error in health check: {str(e)}")
        raise

def test_market_segmentation() -> None:
    """Test the market segmentation endpoint"""
    data: List[Dict[str, Any]] = [
        {"farm_size": 5.0, "income": 100000, "tech_adoption": 0.7, "age": 45},
        {"farm_size": 8.0, "income": 150000, "tech_adoption": 0.8, "age": 52},
        {"farm_size": 3.0, "income": 80000, "tech_adoption": 0.5, "age": 38},
        {"farm_size": 10.0, "income": 200000, "tech_adoption": 0.9, "age": 55},
        {"farm_size": 6.0, "income": 120000, "tech_adoption": 0.6, "age": 42}
    ]
    try:
        response = requests.post(f"{BASE_URL}/market-segmentation", json=data)
        response.raise_for_status()
        result = response.json()
        assert "clusters" in result
        assert "reduced_data" in result
        print("Market segmentation test passed successfully")
    except (requests.exceptions.RequestException, AssertionError) as e:
        print(f"Error in market segmentation test: {str(e)}")
        raise

def test_financial_metrics() -> None:
    """Test the financial metrics endpoint"""
    data: List[Dict[str, Any]] = [
        {"revenue": 100000, "cost": 70000, "investment": 50000},
        {"revenue": 150000, "cost": 100000, "investment": 75000}
    ]
    try:
        response = requests.post(f"{BASE_URL}/financial-metrics", json=data)
        response.raise_for_status()
        result = response.json()
        assert "gross_profit" in result
        assert "profit_margin" in result
        assert "roi" in result
        assert "cac" in result
        assert "clv" in result
        assert "mrr" in result
        print("Financial metrics test passed successfully")
    except (requests.exceptions.RequestException, AssertionError) as e:
        print(f"Error in financial metrics test: {str(e)}")
        raise

def test_time_series_analysis() -> None:
    """Test the time series analysis endpoint"""
    try:
        # Generate sample time series data
        dates = pd.date_range(start='2022-01-01', periods=12, freq='MS')
        values = np.random.normal(100, 10, 12)
        data = [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in zip(dates, values)]
        
        response = requests.post(f"{BASE_URL}/time-series-analysis", json=data)
        response.raise_for_status()
        result = response.json()
        assert "trend" in result
        assert "seasonal" in result
        assert "forecast" in result
        print("Time series analysis test passed successfully")
    except (requests.exceptions.RequestException, AssertionError) as e:
        print(f"Error in time series analysis test: {str(e)}")
        raise

def test_direct_sales() -> None:
    """Test the direct sales endpoints"""
    try:
        # Test adding a product
        product_data: Dict[str, Any] = {
            "name": "Organic Wheat",
            "category": "Grains",
            "quantity": 1000.0,
            "unit": "kg",
            "price_per_unit": 25.0,
            "quality_grade": "A",
            "harvest_date": "2024-03-15",
            "location": "Punjab",
            "farmer_id": "FARM001"
        }
        response = requests.post(f"{BASE_URL}/direct-sales/product", json=product_data)
        response.raise_for_status()
        product_id = response.json()["product_id"]

        # Test adding a wholesaler
        wholesaler_data: Dict[str, Any] = {
            "id": "WH001",
            "name": "John Doe",
            "business_name": "Agro Wholesale",
            "contact": "+91-9876543210",
            "location": "Punjab",
            "preferred_products": ["Grains", "Vegetables"],
            "min_order_quantity": 500.0,
            "max_order_quantity": 5000.0,
            "payment_terms": "Net 30"
        }
        response = requests.post(f"{BASE_URL}/direct-sales/wholesaler", json=wholesaler_data)
        response.raise_for_status()
        wholesaler_id = response.json()["wholesaler_id"]

        # Test creating a sale
        sale_data: Dict[str, Any] = {
            "product_id": product_id,
            "wholesaler_id": wholesaler_id,
            "quantity": 800.0,
            "price_per_unit": 30.0,
            "total_amount": 24000.0,
            "sale_date": datetime.now().strftime("%Y-%m-%d"),
            "payment_status": "pending",
            "delivery_status": "scheduled"
        }
        response = requests.post(f"{BASE_URL}/direct-sales/sale", json=sale_data)
        response.raise_for_status()

        # Test getting farmer's products
        response = requests.get(f"{BASE_URL}/direct-sales/farmer/FARM001/products")
        response.raise_for_status()
        assert len(response.json()["products"]) > 0

        # Test getting matching wholesalers
        response = requests.get(f"{BASE_URL}/direct-sales/product/{product_id}/matching-wholesalers")
        response.raise_for_status()
        assert len(response.json()["matching_wholesalers"]) > 0

        # Test revenue increase calculation
        response = requests.get(f"{BASE_URL}/direct-sales/farmer/FARM001/revenue-increase")
        response.raise_for_status()
        result = response.json()
        assert "current_revenue" in result
        assert "potential_revenue" in result
        assert "revenue_increase" in result
        assert "increase_percentage" in result

        # Test sales analytics
        response = requests.get(f"{BASE_URL}/direct-sales/farmer/FARM001/analytics")
        response.raise_for_status()
        result = response.json()
        assert "total_sales" in result
        assert "total_quantity" in result
        assert "average_price" in result
        assert "number_of_sales" in result
        assert "payment_status" in result
        
        print("Direct sales tests passed successfully")
    except (requests.exceptions.RequestException, AssertionError) as e:
        print(f"Error in direct sales tests: {str(e)}")
        raise

def main() -> None:
    """Main function to run all tests"""
    print("\nStarting Kisan Mitra API Tests")
    print("="*50)
    
    try:
        # Test each endpoint
        test_root()
        time.sleep(1)  # Small delay between requests
        
        test_health_check()
        time.sleep(1)
        
        test_market_segmentation()
        time.sleep(1)
        
        test_financial_metrics()
        time.sleep(1)
        
        test_time_series_analysis()
        time.sleep(1)
        
        test_direct_sales()
        
        print("\n" + "="*50)
        print("All tests completed successfully!")
        print("="*50)
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"Error occurred during testing: {str(e)}")
        print("="*50)
        raise

if __name__ == "__main__":
    main() 