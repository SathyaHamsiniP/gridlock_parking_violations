import pandas as pd
import h3
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np

print("Loading dataset...")
df = pd.read_csv('jan to may police violation_anonymized791b166.csv')

def get_h3_cell(lat, lon, res):
    try:
        return h3.latlng_to_cell(lat, lon, res)
    except AttributeError:
        return h3.geo_to_h3(lat, lon, res)

print("Engineering features...")
df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
df = df.dropna(subset=['created_datetime', 'latitude', 'longitude'])

df['hour'] = df['created_datetime'].dt.hour
df['day_of_week'] = df['created_datetime'].dt.dayofweek

print("Calculating H3 indices (Optimized)...")
unique_coords = df[['latitude', 'longitude']].drop_duplicates()
coord_to_h3 = { (lat, lon): get_h3_cell(lat, lon, 9) for lat, lon in unique_coords.itertuples(index=False)}
df['h3_index'] = df.apply(lambda x: coord_to_h3[(x['latitude'], x['longitude'])], axis=1)

print("Saving metadata...")
# Safer metadata aggregation
def get_mode(x):
    m = x.mode()
    return str(m.iloc[0]) if not m.empty else "Unknown"

h3_meta = df.groupby('h3_index').agg({
    'latitude': 'mean',
    'longitude': 'mean',
    'location': get_mode
}).reset_index()

print("Preparing training data...")
train_df = df.groupby(['h3_index', 'hour', 'day_of_week']).size().reset_index(name='violation_count')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['h3_encoded'] = le.fit_transform(train_df['h3_index'])

X = train_df[['h3_encoded', 'hour', 'day_of_week']]
y = train_df['violation_count']

print(f"Training on {len(X)} records...")
model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X, y)

print("Saving artifacts...")
with open('v_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('h3_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
h3_meta.to_csv('h3_metadata.csv', index=False)

print("✅ AI Forecasting Model Trained Successfully.")
