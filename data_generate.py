import pandas as pd
import random
from datetime import datetime, timedelta

def generate_recent_sales(num_rows=5000):
    data = []
    
    # Define date range: last 5 months (July 1, 2025 to Nov 30, 2025)
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 11, 30)
    days_range = (end_date - start_date).days
    
    products = {
        'PROD-X10': {'cat': 'Electronics', 'price': 120.00},
        'PROD-A55': {'cat': 'Office Supplies', 'price': 15.00},
        'PROD-Z99': {'cat': 'Furniture', 'price': 450.00},
        'PROD-C33': {'cat': 'Software', 'price': 5000.00},
        'PROD-B20': {'cat': 'Peripherals', 'price': 40.00}
    }
    
    # Generate 50 unique Customer IDs with different buying behaviors
    customer_ids = [f"CUST-{random.randint(1000, 9999)}" for _ in range(50)]
    
    for i in range(1, num_rows + 1):
        invoice_id = f"INV-2025-{str(i).zfill(5)}"
        cust_id = random.choice(customer_ids)
        
        # Random date within the last 5 months
        days_offset = random.randint(0, days_range)
        inv_date = start_date + timedelta(days=days_offset)
        
        sku = random.choice(list(products.keys()))
        category = products[sku]['cat']
        base_price = products[sku]['price']
        
        # Simulate "bulk order" spikes
        quantity = random.randint(50, 150) if random.random() > 0.90 else random.randint(1, 20)
        
        # Volume discount logic
        final_price = base_price * 0.9 if quantity > 50 else base_price
        total = round(quantity * final_price, 2)
        
        data.append([
            invoice_id, cust_id, inv_date.strftime('%Y-%m-%d'), 
            sku, category, quantity, final_price, total, "Net30"
        ])
        
    df = pd.DataFrame(data, columns=[
        'Invoice_ID', 'Customer_ID', 'Invoice_Date', 'SKU_ID', 
        'Product_Category', 'Quantity', 'Unit_Price', 'Total_Revenue', 'Payment_Terms'
    ])
    
    # Sort by date
    df = df.sort_values(by='Invoice_Date')
    df.to_csv('last_5_months_sales_data.csv', index=False)
    print(f"Generated {num_rows} rows for period July-Nov 2025.")

if __name__ == "__main__":
    generate_recent_sales()