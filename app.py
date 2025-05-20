from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from business_analysis import KisanMitraBusinessAnalysis
import io
import json
from direct_sales import (
    Product,
    Wholesaler,
    DirectSale,
    direct_sales_system
)

app = FastAPI(
    title="Kisan Mitra Business Analysis API",
    description="API for analyzing agricultural market data and business metrics",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the business analyzer
analyzer = KisanMitraBusinessAnalysis()

class MarketData(BaseModel):
    farm_size: float
    income: float
    tech_adoption: float
    age: float

class FinancialData(BaseModel):
    revenue: float
    cost: float
    investment: Optional[float] = None

@app.post("/market-segmentation")
async def perform_segmentation(data: List[MarketData]) -> Dict[str, Any]:
    """
    Perform market segmentation on farmer data
    
    Args:
        data: List of market data points
        
    Returns:
        Dict containing clusters and reduced data
        
    Raises:
        HTTPException: If segmentation fails
    """
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([d.dict() for d in data])
        
        # Perform segmentation
        clusters = analyzer.perform_market_segmentation(df)
        reduced_data = analyzer.dimensionality_reduction(df)
        
        return {
            "clusters": clusters.tolist(),
            "reduced_data": reduced_data.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/time-series-analysis")
async def analyze_time_series(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze time series data from uploaded CSV file
    
    Args:
        file: CSV file containing time series data
        
    Returns:
        Dict containing trend, seasonal, and forecast data
        
    Raises:
        HTTPException: If analysis fails or data is invalid
    """
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Ensure data has date and value columns
        if 'date' not in df.columns or 'value' not in df.columns:
            raise ValueError("CSV must contain 'date' and 'value' columns")
        
        # Convert to time series and handle missing values
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        
        if len(df) < 24:
            raise ValueError(f"Need at least 24 data points for analysis. Got {len(df)} points.")
        
        ts_data = pd.Series(df['value'].values, index=df['date'])
        
        # Perform analysis
        results = analyzer.time_series_analysis(ts_data)
        
        # Convert numpy arrays to lists and handle NaN values
        return {
            "trend": [float(x) if not np.isnan(x) else None for x in results['trend']],
            "seasonal": [float(x) if not np.isnan(x) else None for x in results['seasonal']],
            "forecast": [float(x) if not np.isnan(x) else None for x in results['forecast']]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/financial-metrics")
async def calculate_metrics(data: List[FinancialData]) -> Dict[str, Any]:
    """
    Calculate financial metrics for the business
    
    Args:
        data: List of financial data points
        
    Returns:
        Dict containing calculated financial metrics
        
    Raises:
        HTTPException: If calculation fails
    """
    try:
        revenue = pd.Series([d.revenue for d in data])
        costs = pd.Series([d.cost for d in data])
        investments = pd.Series([d.investment for d in data if d.investment is not None])
        
        metrics = analyzer.calculate_financial_metrics(
            revenue,
            costs,
            investments if not investments.empty else None
        )
        
        # Convert numpy types to Python native types for JSON serialization
        return {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
               for k, v in metrics.items()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information
    
    Returns:
        Dict containing API information and available endpoints
    """
    return {
        "name": "Kisan Mitra Business Analysis API",
        "version": "1.0.0",
        "endpoints": [
            "/market-segmentation",
            "/time-series-analysis",
            "/financial-metrics",
            "/direct-sales/product",
            "/direct-sales/wholesaler",
            "/direct-sales/sale",
            "/direct-sales/farmer/{farmer_id}/products",
            "/direct-sales/product/{product_id}/matching-wholesalers",
            "/direct-sales/farmer/{farmer_id}/revenue-increase",
            "/direct-sales/farmer/{farmer_id}/analytics"
        ]
    }

# Direct Sales endpoints
@app.post("/direct-sales/product")
async def add_product(product: Product) -> Dict[str, Any]:
    """
    Add a new product to the direct sales system
    
    Args:
        product: Product details
        
    Returns:
        Dict containing success message and product ID
        
    Raises:
        HTTPException: If product addition fails
    """
    try:
        product_id = direct_sales_system.add_product(product)
        return {"message": "Product added successfully", "product_id": product_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/direct-sales/wholesaler")
async def add_wholesaler(wholesaler: Wholesaler) -> Dict[str, Any]:
    """
    Add a new wholesaler to the direct sales system
    
    Args:
        wholesaler: Wholesaler details
        
    Returns:
        Dict containing success message and wholesaler ID
        
    Raises:
        HTTPException: If wholesaler addition fails
    """
    try:
        wholesaler_id = direct_sales_system.add_wholesaler(wholesaler)
        return {"message": "Wholesaler added successfully", "wholesaler_id": wholesaler_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/direct-sales/sale")
async def create_sale(sale: DirectSale) -> Dict[str, Any]:
    """
    Create a new sale in the direct sales system
    
    Args:
        sale: Sale details
        
    Returns:
        Dict containing success message and sale ID
        
    Raises:
        HTTPException: If sale creation fails
    """
    try:
        sale_id = direct_sales_system.create_sale(sale)
        return {"message": "Sale recorded successfully", "sale_id": sale_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/direct-sales/farmer/{farmer_id}/products")
async def get_farmer_products(farmer_id: str) -> Dict[str, Any]:
    """
    Get all products listed by a farmer
    
    Args:
        farmer_id: ID of the farmer
        
    Returns:
        Dict containing list of products
    """
    products = direct_sales_system.get_farmer_products(farmer_id)
    return {"products": products}

@app.get("/direct-sales/product/{product_id}/matching-wholesalers")
async def get_matching_wholesalers(product_id: str) -> Dict[str, Any]:
    """
    Get matching wholesalers for a product
    
    Args:
        product_id: ID of the product
        
    Returns:
        Dict containing list of matching wholesalers
        
    Raises:
        HTTPException: If product is not found
    """
    product = next((p for p in direct_sales_system.products if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    matching_wholesalers = direct_sales_system.get_matching_wholesalers(Product(**product))
    return {"matching_wholesalers": matching_wholesalers}

@app.get("/direct-sales/farmer/{farmer_id}/revenue-increase")
async def get_revenue_increase(farmer_id: str) -> Dict[str, Any]:
    """
    Calculate potential revenue increase for a farmer
    
    Args:
        farmer_id: ID of the farmer
        
    Returns:
        Dict containing revenue metrics
    """
    return direct_sales_system.calculate_revenue_increase(farmer_id)

@app.get("/direct-sales/farmer/{farmer_id}/analytics")
async def get_sales_analytics(farmer_id: str) -> Dict[str, Any]:
    """
    Get sales analytics for a farmer
    
    Args:
        farmer_id: ID of the farmer
        
    Returns:
        Dict containing sales analytics
    """
    return direct_sales_system.get_sales_analytics(farmer_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 