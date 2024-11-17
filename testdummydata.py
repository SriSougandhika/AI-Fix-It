# THIS CODE WORKS ON SAMPLE DUMMY DATABASE, TO SIMULATE REAL TIME PROCESSING
# THIS UPDATES THE PREDICTIVE MAINTENANCE EVERY 3 SECONDS BY TAKING A ROW FROM DB
# THE DB OF COURSE IS A DUMMY SIMILAR TO DATABASE CREATED.
# THE SCALING, LABEL ENCODING, DATA PREPROCESSING IS DONE BEFORE PREDICTION HERE
# THEN FINALLY PREDICTION IS VISIBLE AS AN UPDATE EVERY THREE SECONDS

import sqlite3
import pandas as pd
import joblib
import time

# Connect to the .db file
conn = sqlite3.connect('dummy_vehicle_maintenance.db')

# Query to fetch all data from a specific table (replace 'your_table_name' with the table name)
query = "SELECT * FROM dummy_maintenance_data"

# Convert the result of the query to a DataFrame
df = pd.read_sql_query(query, conn)

le_make_and_model = joblib.load('Make_and_Model_encoder.pkl')
le_vehicle_type = joblib.load('Vehicle_Type_encoder.pkl')
le_route_info = joblib.load('Route_Info_encoder.pkl')
le_maintenance_type = joblib.load('Maintenance_Type_encoder.pkl')
le_weather_condition = joblib.load('Weather_Conditions_encoder.pkl')
le_road_condition = joblib.load('Road_Conditions_encoder.pkl')
scaler = joblib.load('scaler.pkl')

df['Make_and_Model'] = le_make_and_model.transform(df['Make_and_Model'])
df['Vehicle_Type'] = le_vehicle_type.transform(df['Vehicle_Type'])
df['Route_Info'] = le_route_info.transform(df['Route_Info'])
df['Maintenance_Type'] = le_maintenance_type.transform(df['Maintenance_Type'])
df['Weather_Conditions'] = le_weather_condition.transform(df['Weather_Conditions'])
df['Road_Conditions'] = le_road_condition.transform(df['Road_Conditions'])
print("LABEL ENCODING DONE!")

original = pd.read_csv('logistics_dataset_with_maintenance_required.csv')
all_features = original.columns
for col in all_features:
    if col not in df.columns:
        df[col] = 0

# Extract year, month, and day into separate columns, first convert to datetime
df['Last_Maintenance_Date'] = pd.to_datetime(df['Last_Maintenance_Date'], errors='coerce')
df['Last_Maintenance_Year'] = df['Last_Maintenance_Date'].dt.year
df['Last_Maintenance_Month'] = df['Last_Maintenance_Date'].dt.month
df['Last_Maintenance_Day'] = df['Last_Maintenance_Date'].dt.day
df.drop(columns=['Last_Maintenance_Date'], inplace=True)
df.drop(columns=['Maintenance_Required'], inplace=True)

print("EXTRA FEATURES SET!")
# Select columns to scale
columns_to_scale = ['Usage_Hours', 'Load_Capacity', 'Actual_Load', 'Predictive_Score',
                    'Delivery_Times', 'Downtime_Maintenance', 'Impact_on_Efficiency']

# Apply scaler on the selected columns
df[columns_to_scale] = scaler.transform(df[columns_to_scale])
print("SCALING DONE!")

items_to_remove = ['Last_Maintenance_Date', 'Maintenance_Required']
items_to_add = ['Last_Maintenance_Year', 'Last_Maintenance_Month', 'Last_Maintenance_Day']
all_features = [item for item in all_features if item not in items_to_remove]
all_features.extend(items_to_add)

model = joblib.load('model.pkl')
maintenance_prediction = model.predict(df.iloc[0][all_features].to_frame().T)
print("Maintenance Required (1 means Yes, 0 means No):", maintenance_prediction[0])

for i in range(len(df)):
    print('UPDATE!')
    row_data = df.iloc[i][all_features].to_frame().T
    maintenance_prediction = model.predict(row_data)
    print(f"Row {i+1} - Maintenance Required (1 means Yes, 0 means No):", maintenance_prediction[0])
    time.sleep(3)

# Close the connection
conn.close()
