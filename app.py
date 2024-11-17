from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import threading
import time
import pandas as pd
import sqlite3
import joblib
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

app = Flask(__name__)

'''
# Load the trained model
model = load_model("parts_classifier.h5")
'''
# Download the model file
model_path = hf_hub_download(repo_id="SriSougandhika/parts_classifier_aifixit", filename="parts_classifier.h5")
# Load the model
model = load_model(model_path)

# Define class names
class_names = ["bolt", "locatingpin", "nut", "washer"]

# Initialize video capture
cap = cv2.VideoCapture(0)
is_streaming = False


def generate_frame():
    while is_streaming:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the bounding box dimensions (fixed size in the center)
        box_x = int((frame.shape[1] - 128) / 2)
        box_y = int((frame.shape[0] - 128) / 2)
        frame_width, frame_height = 128, 128  # Match model input size

        # Draw the bounding box
        cv2.rectangle(frame, (box_x, box_y), (box_x + frame_width, box_y + frame_height), (0, 255, 0), 2)

        # Pre-process the frame for prediction
        roi = frame[box_y:box_y + frame_height, box_x:box_x + frame_width]
        frame_resized = cv2.resize(roi, (128, 128))
        frame_normalized = frame_resized / 255.0
        img_array = np.expand_dims(frame_normalized, axis=0)

        # Get prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Add prediction label on the frame
        if confidence > 0.85:
            predicted_label = class_names[predicted_class]
            label = f"{predicted_label} ({confidence * 100:.2f}%)"
            cv2.putText(frame, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No object detected", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Encode frame to JPEG and send it as response
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


# Route for displaying the vehicle maintenance history and predictions
@app.route('/maintenance')
def maintenance():
    return render_template('maintenance.html')


@app.route('/update_maintenance', methods=['GET'])
def update_maintenance():
    # Get the index value from query parameters (default: 0)
    index_value = int(request.args.get('index', 0))

    # Connect to SQLite and fetch predictions (code as you have it)
    conn = sqlite3.connect('dummy_vehicle_maintenance.db')
    query = "SELECT * FROM dummy_maintenance_data"
    df = pd.read_sql_query(query, conn)

    # Load encoders and transform data
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

    original = pd.read_csv('logistics_dataset_with_maintenance_required.csv')
    all_features = original.columns
    for col in all_features:
        if col not in df.columns:
            df[col] = 0

    df['Last_Maintenance_Date'] = pd.to_datetime(df['Last_Maintenance_Date'], errors='coerce')
    df['Last_Maintenance_Year'] = df['Last_Maintenance_Date'].dt.year
    df['Last_Maintenance_Month'] = df['Last_Maintenance_Date'].dt.month
    df['Last_Maintenance_Day'] = df['Last_Maintenance_Date'].dt.day
    df.drop(columns=['Last_Maintenance_Date'], inplace=True)
    df.drop(columns=['Maintenance_Required'], inplace=True)

    columns_to_scale = ['Usage_Hours', 'Load_Capacity', 'Actual_Load', 'Predictive_Score', 'Delivery_Times', 'Downtime_Maintenance', 'Impact_on_Efficiency']
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    items_to_remove = ['Last_Maintenance_Date', 'Maintenance_Required']
    items_to_add = ['Last_Maintenance_Year', 'Last_Maintenance_Month', 'Last_Maintenance_Day']
    all_features = [item for item in all_features if item not in items_to_remove]
    all_features.extend(items_to_add)

    model_pred = joblib.load('model.pkl')
    prediction = model_pred.predict(df.iloc[index_value][all_features].to_frame().T)

    input_to_table = df.iloc[index_value][all_features].to_dict()
    input_to_table['Maintenance_Required'] = int(prediction[0])
    # Convert any numpy.int64 or numpy.float64 to native Python types
    input_to_table = {key: (value.item() if hasattr(value, "item") else value) for key, value in input_to_table.items()}
    return jsonify(input_to_table)


@app.route('/start_camera')
def start_camera():
    global is_streaming
    is_streaming = True
    return render_template('camera.html')


@app.route('/stop_camera')
def stop_camera():
    global is_streaming
    is_streaming = False
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Inventory data (dummy values)
inventory = [
    {"product": "Gears", "quantity": 50, "threshold": 20},
    {"product": "Metal sheets", "quantity": 70, "threshold": 30},
    {"product": "Engine motors", "quantity": 40, "threshold": 15},
    {"product": "Valves", "quantity": 35, "threshold": 10},
    {"product": "Ball bearings", "quantity": 25, "threshold": 10},
]

# Lock for thread-safe operations
inventory_lock = threading.Lock()


def decrement_inventory():
    """Decrement inventory every 5 seconds."""
    while True:
        with inventory_lock:
            for item in inventory:
                if item["quantity"] > 0:
                    item["quantity"] -= 10
        time.sleep(5)


# Start inventory decrement thread
threading.Thread(target=decrement_inventory, daemon=True).start()


@app.route('/inventory')
def inventory_page():
    """Render the inventory page."""
    return render_template('inventory.html')


@app.route('/get_inventory', methods=['GET'])
def get_inventory():
    """Return the current inventory status as JSON."""
    with inventory_lock:
        return jsonify(inventory)


@app.route('/replenish_inventory', methods=['POST'])
def replenish_inventory():
    """Replenish the inventory for a specific product."""
    product = request.json.get("product")
    with inventory_lock:
        for item in inventory:
            if item["product"] == product:
                item["quantity"] += 50  # Replenish 50 units
                break
    return jsonify({"message": f"{product} inventory replenished!"})


@app.route('/update_inventory')
def update_inventory():
    global inventory
    for item in inventory:
        if item['quantity'] > 0:
            item['quantity'] -= 10  # Decrease by 10 units
            if item['quantity'] < 0:
                item['quantity'] = 0  # Ensure it doesn't go below 0

    return jsonify(inventory)


if __name__ == '__main__':
    app.run(debug=True)
