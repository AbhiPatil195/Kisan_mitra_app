from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

class Product(BaseModel):
    name: str
    category: str
    quantity: float
    unit: str  # kg, quintal, ton
    price_per_unit: float
    quality_grade: str  # A, B, C
    harvest_date: str
    location: str
    farmer_id: str

class Wholesaler(BaseModel):
    id: str
    name: str
    business_name: str
    contact: str
    location: str
    preferred_products: List[str]
    min_order_quantity: float
    max_order_quantity: float
    payment_terms: str

class DirectSale(BaseModel):
    product_id: str
    wholesaler_id: str
    quantity: float
    price_per_unit: float
    total_amount: float
    sale_date: str
    payment_status: str
    delivery_status: str

class DirectSalesSystem:
    def __init__(self):
        self.products: List[Dict[str, Any]] = []
        self.wholesalers: List[Dict[str, Any]] = []
        self.sales: List[Dict[str, Any]] = []
        
    def add_product(self, product: Product) -> str:
        """Add a new product listing
        
        Args:
            product: Product object containing product details
            
        Returns:
            str: Generated product ID
        """
        product_id = f"PROD_{len(self.products) + 1}"
        self.products.append({
            "id": product_id,
            **product.dict()
        })
        return product_id
    
    def add_wholesaler(self, wholesaler: Wholesaler) -> str:
        """Add a new wholesaler
        
        Args:
            wholesaler: Wholesaler object containing wholesaler details
            
        Returns:
            str: Wholesaler ID
        """
        self.wholesalers.append(wholesaler.dict())
        return wholesaler.id
    
    def create_sale(self, sale: DirectSale) -> str:
        """Create a new direct sale
        
        Args:
            sale: DirectSale object containing sale details
            
        Returns:
            str: Product ID associated with the sale
        """
        self.sales.append(sale.dict())
        return sale.product_id
    
    def get_farmer_products(self, farmer_id: str) -> List[Dict[str, Any]]:
        """Get all products listed by a farmer
        
        Args:
            farmer_id: ID of the farmer
            
        Returns:
            List[Dict[str, Any]]: List of products
        """
        return [p for p in self.products if p["farmer_id"] == farmer_id]
    
    def get_matching_wholesalers(self, product: Product) -> List[Dict[str, Any]]:
        """Find matching wholesalers for a product
        
        Args:
            product: Product object to find matches for
            
        Returns:
            List[Dict[str, Any]]: List of matching wholesalers
        """
        return [
            w for w in self.wholesalers
            if product.category in w["preferred_products"]
            and w["min_order_quantity"] <= product.quantity <= w["max_order_quantity"]
            and w["location"] == product.location
        ]
    
    def calculate_revenue_increase(self, farmer_id: str) -> Dict[str, Any]:
        """Calculate potential revenue increase from direct sales
        
        Args:
            farmer_id: ID of the farmer
            
        Returns:
            Dict[str, Any]: Dictionary containing revenue metrics
        """
        farmer_products = self.get_farmer_products(farmer_id)
        if not farmer_products:
            return {"message": "No products found for this farmer"}
        
        total_revenue = sum(p["quantity"] * p["price_per_unit"] for p in farmer_products)
        # Assuming 20% higher price for direct sales (no middleman)
        potential_revenue = total_revenue * 1.2
        revenue_increase = potential_revenue - total_revenue
        
        return {
            "current_revenue": total_revenue,
            "potential_revenue": potential_revenue,
            "revenue_increase": revenue_increase,
            "increase_percentage": (revenue_increase / total_revenue) * 100
        }
    
    def get_sales_analytics(self, farmer_id: str) -> Dict[str, Any]:
        """Get sales analytics for a farmer
        
        Args:
            farmer_id: ID of the farmer
            
        Returns:
            Dict[str, Any]: Dictionary containing sales analytics
        """
        farmer_sales = [s for s in self.sales if s["product_id"].startswith(f"PROD_{farmer_id}")]
        
        if not farmer_sales:
            return {"message": "No sales data found for this farmer"}
        
        total_sales = sum(s["total_amount"] for s in farmer_sales)
        total_quantity = sum(s["quantity"] for s in farmer_sales)
        avg_price = total_sales / total_quantity if total_quantity > 0 else 0
        
        return {
            "total_sales": total_sales,
            "total_quantity": total_quantity,
            "average_price": avg_price,
            "number_of_sales": len(farmer_sales),
            "payment_status": {
                "completed": len([s for s in farmer_sales if s["payment_status"] == "completed"]),
                "pending": len([s for s in farmer_sales if s["payment_status"] == "pending"])
            }
        }

# Initialize the system
direct_sales_system = DirectSalesSystem() 