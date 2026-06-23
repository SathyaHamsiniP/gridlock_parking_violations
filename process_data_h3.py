import pandas as pd
import numpy as np
import h3
import json
from datetime import datetime

# Configuration
INPUT_FILE = 'jan to may police violation_anonymized791b166.csv'
H3_RESOLUTION = 9 # Approx 170m across hex

BSU_MAPPING = {
    'SCOOTER': 1, 'MOTOR CYCLE': 1, 'MOPED': 1,
    'CAR': 2, 'JEEP': 2, 'VAN': 2, 'MAXI-CAB': 2, 'PASSENGER AUTO': 1.5, 'GOODS AUTO': 1.5,
    'TANKER': 4, 'BUS (BMTC/KSRTC)': 5, 'PRIVATE BUS': 5, 'TOURIST BUS': 5, 'FACTORY BUS': 5,
    'LGV': 3, 'HGV': 5, 'LORRY/GOODS VEHICLE': 4, 'TEMPO': 3, 'MINI LORRY': 3,
    'SCHOOL VEHICLE': 4, 'TRACTOR': 3, 'OTHERS': 2
}

def load_and_preprocess(filepath):
    print("Loading data...")
    cols = ['latitude', 'longitude', 'vehicle_type', 'violation_type', 'created_datetime', 'location', 'junction_name']
    df = pd.read_csv(filepath, usecols=cols)
    df = df.dropna(subset=['latitude', 'longitude'])
    df['bsu'] = df['vehicle_type'].map(BSU_MAPPING).fillna(2)
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime'])
    df['hour'] = df['created_datetime'].dt.hour
    df['month'] = df['created_datetime'].dt.month
    return df

def find_hotspots_h3(df):
    print("Aggregating by H3 hexagons...")
    # Generate H3 index for each point (v4 API)
    df['h3_index'] = df.apply(lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], H3_RESOLUTION), axis=1)
    return df

def calculate_priority(df):
    print("Calculating priority scores per hexagon...")
    # Group by H3 index
    groups = df.groupby('h3_index')
    
    report = []
    # Only process hexes with at least 50 violations for clarity
    for h3_index, group in groups:
        if len(group) < 50: continue
        
        total_violations = len(group)
        total_space_blocked = group['bsu'].sum()
        
        high_impact_count = group['violation_type'].str.contains('MAIN ROAD|CROSSING|BUS LANE', case=False, na=False).sum()
        high_impact_pct = (high_impact_count / total_violations) * 100
        
        # Trend
        may_data = len(group[group['month'] == 5])
        prev_avg = len(group[group['month'] < 5]) / 4
        growth = ((may_data - prev_avg) / prev_avg * 100) if prev_avg > 0 else 0
        
        peak_hour = group['hour'].value_counts().idxmax()
        peak_hour_share = (group['hour'].value_counts().max() / total_violations) * 100
        
        top_location = group['location'].mode().iloc[0] if not group['location'].mode().empty else "Unknown"
        top_junction = group['junction_name'].mode().iloc[0] if not group['junction_name'].mode().empty else "No Junction"
        
        # Center of the hex (v4 API)
        lat, lon = h3.cell_to_latlng(h3_index)

        # PRIORITY SCORE (Updated for better sensitivity)
        # 50% Volume/Space, 40% Main Road/Impact, 10% Growth
        raw_score = (min(total_violations/500, 1) * 30 + 
                     min(total_space_blocked/1000, 1) * 20 + 
                     (high_impact_pct/100) * 40 + 
                     min(max(growth, 0)/50, 1) * 10)
        
        report.append({
            'area_name': f"{top_junction} ({top_location[:25]}...)",
            'h3_index': h3_index,
            'latitude': lat,
            'longitude': lon,
            'violations': total_violations,
            'space_blocked_units': int(total_space_blocked),
            'high_impact_pct': round(high_impact_pct, 1),
            'growth_pct': round(growth, 1),
            'peak_hour': f"{peak_hour}:00",
            'peak_share': round(peak_hour_share, 1),
            'raw_score': raw_score,
            'action': "Deploy Tow Truck" if high_impact_pct > 25 else "Increase Bike Patrol"
        })
    
    # NORMALIZATION: Scale worst raw score to ~98
    report_df = pd.DataFrame(report)
    if not report_df.empty:
        max_raw = report_df['raw_score'].max()
        report_df['priority_score'] = report_df['raw_score'].apply(lambda x: round((x / max_raw) * 98, 1))
    else:
        report_df['priority_score'] = 0

    return report_df.sort_values(by='priority_score', ascending=False)

if __name__ == "__main__":
    df = load_and_preprocess(INPUT_FILE)
    df = find_hotspots_h3(df)
    final_report = calculate_priority(df)
    
    print("\n--- TOP 10 ENFORCEMENT PRIORITIES (H3) ---")
    print(final_report.head(10)[['area_name', 'violations', 'priority_score', 'action']].to_string(index=False))
    
    final_report.to_json('hotspot_report.json', orient='records')
    print("\nReport saved to hotspot_report.json")
