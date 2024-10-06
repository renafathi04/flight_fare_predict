from src.components.data_loader import load_object
from src.components.feature_engineering import FeatureEngineering
from src.components.model_training import ModelTraining
from src.logger import logging


# Step 1: Load the dataset
data_filepath = 'data/Flight_Fare.csv'
data = load_object(data_filepath)

# Step 2: Feature Engineering

# Initialize feature engineering class
fe = FeatureEngineering(data)

# Drop unwanted columns
unwanted_columns = ['Route','Additional_Info']  
fe = fe.drop_unwanted_cols(unwanted_columns)

# Handle missing values
categorical_columns=['Airline','Source','Destination','Total_Stops','Date_of_Journey','Dep_Time','Arrival_Time','Duration']
fe = fe.handle_missing_value(categorical_columns,strategy='most_frequent')

# numerical_columns=['Date_of_Journey','Dep_Time','Arrival_Time','Duration']
# fe = fe.handle_missing_value(numerical_columns,strategy='median')

# Drop duplicates
fe = fe.drop_duplicate()

# Convert date columns
date_columns = ['Date_of_Journey']  
fe = fe.convert_date(date_columns)

# Convert time columns
time_columns = ['Dep_Time', 'Arrival_Time'] 
fe = fe.convert_time(time_columns)

fe = fe.convert_duration(duration_column='Duration') 
# Apply label encoding
label_columns = ['Total_Stops'] 
fe = fe.label_encoding(label_columns)

# Apply one-hot encoding
specific_cols = ['Airline', 'Source', 'Destination']  # replace with actual columns for one-hot encoding
fe = fe.one_hot_encoding(specific_cols)

# Get the processed data
processed_data = fe.get_processed_data()

# Step 3: Separate features (X) and target (y)
X = processed_data.drop('Price', axis=1)
y = processed_data['Price']

# Step 4: Train models and save the best one
model_trainer = ModelTraining(X, y)
best_model = model_trainer.train_best_model(metric='r2')
model_trainer.save_best_model('best_model.pkl')

logging.info("Flight price prediction pipeline completed.")
