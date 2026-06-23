import pandas as pd
import pickle
import json
import os

def generate_real_cache():
    print("Starting Intelligence Cache Generation from Model...")
    # Load model and encoders
    with open('v_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('h3_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    meta = pd.read_csv('h3_metadata.csv')
    
    cache = {"weekday": [], "weekend": []}
    
    # Pre-encode all H3 indices
    meta['h3_encoded'] = le.transform(meta['h3_index'])
    
    # Generate 24h forecast for Weekday (Mon=0) and Weekend (Sat=5)
    for day_type, day_val in [("weekday", 0), ("weekend", 5)]:
        print(f"Baking risk patterns for {day_type}...")
        for hour in range(24):
            X = meta[['h3_encoded']].copy()
            X['hour'] = hour
            X['day_of_week'] = day_val
            
            # Run Batch Inference
            preds = model.predict(X[['h3_encoded', 'hour', 'day_of_week']])
            
            # We use the total city-wide risk sum to represent the cycle
            city_sum = float(preds.sum())
            cache[day_type].append(round(city_sum, 2))
            
    # Save the authentic model footprint
    with open('predictions_cache.json', 'w') as f:
        json.dump(cache, f)
    print("✅ AUTHENTIC SUM-BASED CACHE RESTORED.")

if __name__ == "__main__":
    generate_real_cache()
