# AI-Fix-It
The proposed framework leverages the recent advancements in AI and ML and use it to enhance the effectiveness in the ever-growing and advancing automobile industry.

## Purpose and Aim:
The proposed framework integrates the following things:
 1. **Classification** - Identify & divide various parts of the machines that need to be assembled.
 2. **Predictive Maintenance** - After being assembled, it needs to have a check done on it - for defaults. This model will be trained through data input.
 3. **Inventory management** - Statistics on which part is in abundance and which is not.

## Datasets used:
Since our domain is automotive industries, to get a successful basic prototype working, we have used the following two Kaggle datasets. Links to respective datasets are provided as well.
 1. Mechanical parts data - This consists of mechanical parts Bolts, Nuts, Washers, Locating Pin. [DATASET-LINK](https://www.kaggle.com/datasets/krishna8338/mechanical-parts-data)
 2. Logistics Vehicle Maintenance History Dataset - This dataset consists of 92,000 records related to vehicle maintenance predictions within logistics operations across California. [DATASET-LINK](https://www.kaggle.com/datasets/datasetengineer/logistics-vehicle-maintenance-history-dataset)

## Code resources:
This repository contains many files. For ease of navigation, this subsection contains file name, with the purpose or the result of code in it.
 1. [**aifixit-classifier.ipynb**](aifixit-classifier.ipynb) - This notebook guides to the building and training of a simple convolutional neural network or CNN, on the mechanical parts dataset. It thus classifies the part to four categories for maintaining simplicity. Later on, in the application, it is equipped with computer vision for real time image procesing and part classification. 
 2. [**aifixit-predictivemaintenance.ipynb**](aifixit-predictivemaintenance.ipynb) - This notebook guides to the building and training of the logistic regression model. The rationale behind choosing the simple model is our main taks being binary classification (0 if the vehicle does not require maintenance else 1). The dataset chosen here is Logistics Vehicle Maintenance History.
 3. [**app.py**](app.py) - Main running file. Contains flask as backend, HTML as frontend. It has further routes for camera (parts classification), predictive maintenance (maintenance required or not) and inventory (for parts available in stock)
 4. [**databaseCreator.py**](databaseCreator.py) - This is a helper file, that helped create the dummy data input for predictive maintenance. To simulate real time data feed, a timer of 3 seconds is added and a row is read from the .db file. Then the prediction is made on it using the regression model from above and is displayed on the application screen with a badge "Yes" or "No" accordingly.
 5. [**dummy_vehicle_maintenance.db**](dummy_vehicle_maintenance.db) - This is the resultant file of running the above code, with dummy values similar to the dataset chosen to train the regression model.
 6. [**logistics_dataset_with_maintenance_required.csv**](logistics_dataset_with_maintenance_required.csv) - This is the same file as the dataset, which has been downloaded from kaggle for ease of use.
 7. [**parts_classifier_realtime.py**](parts_classifier_realtime.py) - This file contains integration of computer vision through opencv and cv2 modules. Using bounding box at the center of camera screen we can put any of the equipment to identify it.
 8. [**testdummydata.py**](testdummydata.py) - This file consist of accepting each row from dummy database created above, to predict. Accordingly a timer is added too.
 9. [Templates](templates) - This folder consists of the frontend files of the app.py. It has HTML codes for landing page, maintenance page, camera page, and inventory page.

Other files:
[model.pkl](model.pkl) is the regression model.
[Brake_Condition_encoder.pkl](Brake_Condition_encoder.pkl), [Maintenance_Type_encoder.pkl](Maintenance_Type_encoder.pkl), [Make_and_Model_encoder.pkl](Make_and_Model_encoder.pkl), [Road_Conditions_encoder.pkl](Road_Conditions_encoder.pkl), [Route_Info_encoder.pkl](Route_Info_encoder.pkl), [Vehicle_Type_encoder.pkl](Vehicle_Type_encoder.pkl) and [Weather_Conditions_encoder.pkl](Weather_Conditions_encoder.pkl) are all encoders used in pre-processing the dummy data.
[scaler.pkl](scaler.pkl) is used to scale and standardize the dummy data.

## How to run the code:
To run the code, kindly ensure the installation of required modules and packages mentioned in the below section. Then simply run the app.py file. It should show you the localhost link, mostly at 5000 port. On clicking the link, we see the main page. With this, navigation further can be done.

## Resources and requirement list:
 1. Flask - 3.1.0
 2. Huggingface-hub - 0.26.2
 3. Joblib - 1.4.2
 4. Keras - 3.6.0
 5. Matplotlib - 3.9.2
 6. Numpy - 2.0.2
 7. Opencv-python - 4.10.0.84
 8. Pandas - 2.2.3
 9. Requests - 2.32.3
 10. Scikit-learn - 1.5.2
 11. Scipy - 1.14.1
 12. Tensorflow - 2.18.0
 13. Sqlite3 - 3.47.0

## Conclusion:
The presented prototype has the following three independent models:
 1. Classification of the parts
 2. Predictive maintenance of the vehicles
 3. Inventory management
Currently, these models are working well as per the datasets considered and the AI algorithms implemented. This working prototype can be further enhanced to a fully automated and integrated model to help the OEMs. It can be performed considering a few layers on top of this basic prototype to make more out of it

The classification can be further involved in defect detection. The predictive maintenance model will not only mention the vehicle requiring attention but also the specific parts that are to be looked into. Finally, the inventory of the parts, will be integrated to ensure the execution of the predictive maintenance. If the parts are defective and cannot be repaired, then the inventory search will be triggered for that part. The products that are to be replaced will be ordered if the inventory runs out beyond the set threshold at any point of time. The products that are to be repaired will not trigger the inventory but contact the corresponding person (Alert/ message for the same can be displayed).
