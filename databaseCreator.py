# THIS IS FOR SOLELY DUMMY DATA CREATION SIMILAR TO THE ORIGINAL DATA
# THIS HELPS SIMULATE REAL TIME INPUT FEED FOR PREDICTIVE MAINTENANCE

import sqlite3
import pandas as pd
import random
from datetime import datetime

# Load the original dataset
original_data = pd.read_csv('logistics_dataset_with_maintenance_required.csv')


# Generate dummy data based on the original dataset's structure
def generate_dummy_data(num_rows):
    dummy_data = []
    for _ in range(num_rows):
        dummy_row = {
            'Vehicle_ID': random.randint(1000, 9999),
            'Make_and_Model': random.choice(original_data['Make_and_Model'].unique()),
            'Year_of_Manufacture': random.randint(2000, 2023),
            'Vehicle_Type': random.choice(original_data['Vehicle_Type'].unique()),
            'Usage_Hours': round(random.uniform(100, 15000), 2),
            'Route_Info': random.choice(original_data['Route_Info'].unique()),
            'Load_Capacity': round(random.uniform(5.0, 10.0), 2),
            'Actual_Load': round(random.uniform(1.0, 9.5), 2),
            'Maintenance_Type': random.choice(original_data['Maintenance_Type'].unique()),
            'Maintenance_Cost': round(random.uniform(50, 500), 2),
            'Predictive_Score': round(random.uniform(0.0, 1.0), 2),
            'Maintenance_Required': random.choice([0, 1]),
            'Weather_Conditions': random.choice(original_data['Weather_Conditions'].unique()),
            'Road_Conditions': random.choice(original_data['Road_Conditions'].unique()),
            'Delivery_Times': round(random.uniform(20, 60), 2),
            'Downtime_Maintenance': round(random.uniform(0.0, 5.0), 2),
            'Impact_on_Efficiency': round(random.uniform(0.0, 1.0), 2),
            'Last_Maintenance_Year': random.randint(2000, 2023),
            'Last_Maintenance_Month': random.randint(1, 12),
            'Last_Maintenance_Day': random.randint(1, 28)
        }
        dummy_data.append(dummy_row)
    return pd.DataFrame(dummy_data)

# Create dummy data
dummy_data = generate_dummy_data(100)  # Generate 100 dummy rows

# Save the dummy data to a SQLite database
db_name = 'dummy_vehicle_maintenance.db'
connection = sqlite3.connect(db_name)

dummy_data.to_sql('dummy_maintenance_data', connection, if_exists='replace', index=False)

print(f"Dummy database '{db_name}' created successfully with 100 rows.")

connection.close()
