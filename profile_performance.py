"""
Performance profiling script for Data Sentinel V2.
Measures time spent in each stage of the analysis pipeline.
"""
import time
import io
import pandas as pd
import sys
sys.path.insert(0, 'C:/Users/nicke/Downloads/Data Sentinel V2')

from app.services.analysis_service import analyze_dataset

def test_performance():
    # Test with ecommerce dirty data (medium size - 228 rows, 13 columns)
    df = pd.read_csv('data/sample/ecommerce_dirty_data.csv')
    
    # Warm up
    print("Warming up...")
    from app.services.analysis_service import analyze_dataset
    _ = analyze_dataset(df)
    
    # Benchmark
    print("Benchmarking...")
    start = time.time()
    result = analyze_dataset(df)
    elapsed = time.time() - start
    print(f"Total analysis time: {elapsed:.3f}s")
    
    # Test with larger dataset
    print("\nTesting with larger dataset...")
    df_large = pd.DataFrame({f'col_{i}': range(10000) for i in range(20)})
    df_large['target'] = [i % 2 for i in range(10000)]
    
    start = time.time()
    result = analyze_dataset(df_large)
    elapsed = time.time() - start
    print(f"Large CSV (10000 rows, 20 cols): {elapsed:.3f}s")
    
    # Test Excel file
    print("\nTesting Excel file...")
    df_excel = pd.DataFrame({f'col_{i}': range(1000) for i in range(15)})
    df_excel['target'] = [i % 3 for i in range(1000)]
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_content = excel_buffer.getvalue()
    
    start = time.time()
    from app.services.analysis_service import analyze_dataset
    result = analyze_dataset(pd.read_excel(io.BytesIO(excel_content), engine='openpyxl'))
    elapsed = time.time() - start
    print(f"Excel analysis (1000 rows, 15 cols): {elapsed:.3f}s")

if __name__ == "__main__":
    import time
    import io
    import pandas as pd
    from sklearn.datasets import load_iris
    import requests
    
    test_performance()