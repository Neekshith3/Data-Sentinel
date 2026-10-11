import sys
sys.path.insert(0, 'C:/Users/nicke/Downloads/Data Sentinel V2')
import uvicorn
uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=False, log_level='debug')