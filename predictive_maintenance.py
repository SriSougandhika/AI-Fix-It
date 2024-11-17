# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import pickle

# Load your dataset
df = pd.read_csv('logistics_dataset_with_maintenance_required.csv')

# 1. Data Preprocessing

# Convert Last_Maintenance_Date to datetime format
df['Last_Maintenance_Date'] = pd.to_datetime(df['Last_Maintenance_Date'], errors='coerce')

# Extract year, month, and day into separate columns
df['Last_Maintenance_Year'] = df['Last_Maintenance_Date'].dt.year
df['Last_Maintenance_Month'] = df['Last_Maintenance_Date'].dt.month
df['Last_Maintenance_Day'] = df['Last_Maintenance_Date'].dt.day

# Drop the original date column
df.drop(columns=['Last_Maintenance_Date'], inplace=True)

# Fill missing values if any
df.fillna(method='ffill', inplace=True)

# Encode all categorical columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Save encoders for future use if needed

# Separate features and target
X = df.drop(columns=['Maintenance_Required'])
y = df['Maintenance_Required']

# Scale numerical features
scaler = StandardScaler()
X[['Usage_Hours', 'Load_Capacity', 'Actual_Load', 'Predictive_Score', 'Delivery_Times', 'Downtime_Maintenance', 'Impact_on_Efficiency']] = scaler.fit_transform(
    X[['Usage_Hours', 'Load_Capacity', 'Actual_Load', 'Predictive_Score', 'Delivery_Times', 'Downtime_Maintenance', 'Impact_on_Efficiency']]
)

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, random_state=42, verbose=1)
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Ensure dummy_data has all necessary columns as in the original training data
# Start by creating a DataFrame with the correct columns
all_features = X.columns  # X is the original feature DataFrame from training
dummy_data_full = pd.DataFrame(columns=all_features)

# Define your dummy data, filling in only the columns you know
dummy_data = pd.DataFrame([{
    'Vehicle_ID': 999,
    'Make_and_Model': label_encoders['Make_and_Model'].transform(['Ford F-150'])[0],
    'Year_of_Manufacture': 2020,
    'Vehicle_Type': label_encoders['Vehicle_Type'].transform(['Truck'])[0],
    'Usage_Hours': 4500,
    'Route_Info': label_encoders['Route_Info'].transform(['Rural'])[0],
    'Load_Capacity': 8.5,
    'Actual_Load': 7.8,
    'Last_Maintenance_Year': 2023,
    'Last_Maintenance_Month': 6,
    'Last_Maintenance_Day': 1,
    'Maintenance_Type': label_encoders['Maintenance_Type'].transform(['Oil Change'])[0],
    'Predictive_Score': 0.2,
    'Weather_Conditions': label_encoders['Weather_Conditions'].transform(['Clear'])[0],
    'Road_Conditions': label_encoders['Road_Conditions'].transform(['Highway'])[0],
    'Delivery_Times': 40,
    'Downtime_Maintenance': 0.1,
    'Impact_on_Efficiency': 0.15
}])

# Add missing columns with default values (e.g., zero or NaN)
for col in all_features:
    if col not in dummy_data.columns:
        dummy_data[col] = 0  # Use an appropriate placeholder, like 0 or np.nan

# Select columns to scale
columns_to_scale = ['Usage_Hours', 'Load_Capacity', 'Actual_Load', 'Predictive_Score',
                    'Delivery_Times', 'Downtime_Maintenance', 'Impact_on_Efficiency']

# Apply scaler on the selected columns
dummy_data[columns_to_scale] = scaler.transform(dummy_data[columns_to_scale])

# Predict maintenance requirement
maintenance_prediction = model.predict(dummy_data[all_features])
print("Maintenance Required (1 means Yes, 0 means No):", maintenance_prediction[0])

# Saving the model using pickle
# Save the model to a file
with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)
