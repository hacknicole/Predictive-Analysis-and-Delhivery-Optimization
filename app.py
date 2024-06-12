from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
from sklearn.decomposition import PCA
import sklearn
import pytz

print(sklearn.__version__)

app = Flask(__name__)

# Load data and models once
data = pd.read_csv('segment.csv')
with open('encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            ist = pytz.timezone('Asia/Kolkata')
            trip_creation_time = datetime.now(ist).strftime("%Y-%m-%dT%H:%M")
            print(trip_creation_time)
            #trip_creation_time = request.form['trip_creation_time']
            source_states = request.form['source_states']
            source_cities = request.form['source_cities'].replace(' ', '_')
            destination_states = request.form['destination_states']
            destination_cities = request.form['destination_cities'].replace(' ', '_')
            rout_type = int(request.form['rout_type'])
            source_name = f"{source_cities} ({source_states})"
            destination_name = f"{destination_cities} ({destination_states})"

            source_center = data.loc[data.source_name == source_name, 'source_center'].values[0]
            destination_center = data.loc[data.destination_name == destination_name, 'destination_center'].values[0]

            actual_distance_to_destination = data.loc[
                (data.destination_center == destination_center) & 
                (data.source_center == source_center), 
                'actual_distance_to_destination'
            ]
            if actual_distance_to_destination.empty:
                return render_template('index.html', error="There is no trip in this route")
            else:
                actual_distance_to_destination = actual_distance_to_destination.values[0]
            print(actual_distance_to_destination)
            datetime_obj = datetime.fromisoformat(trip_creation_time)
            time_part = datetime_obj.time()
            datetime_with_time = datetime.combine(datetime_obj.date(), time_part)
            minutes_to_add = 0 if rout_type == 0 else 265
            new_datetime = datetime_with_time + timedelta(minutes=minutes_to_add)
            od_start_time_hour = new_datetime.time().hour

            source_center_enc = label_encoder['source_center'].transform([source_center])
            source_name_enc = label_encoder['source_name'].transform([source_name])
            destination_center_enc = label_encoder['destination_center'].transform([destination_center])
            destination_name_enc = label_encoder['destination_name'].transform([destination_name])
            source_states_enc = label_encoder['source_state'].transform([source_states])
            destination_states_enc = label_encoder['destination_state'].transform([destination_states])
            actual_distance_to_destination_scaled = scaler.transform([[actual_distance_to_destination]])[0][0]

            pass_values = np.array([[source_center_enc[0], source_name_enc[0], destination_center_enc[0], destination_name_enc[0], float(actual_distance_to_destination_scaled), source_states_enc[0], destination_states_enc[0], int(od_start_time_hour), float(rout_type)]])
            print(pass_values)
            predicted_time = model.predict(pca.transform(pass_values))
            print(predicted_time[0])
            eta = new_datetime + timedelta(minutes=predicted_time[0])
            actual_time_hour = int(predicted_time[0] // 60)
            actual_time_hour_min = round(predicted_time[0] % 60)
            actual_time_display = f"{actual_time_hour} hours and {actual_time_hour_min} minutes"
            eta = eta.strftime('%d-%m-%Y %H:%M')
            trip_creation_time = datetime_obj.strftime('%d-%m-%Y %H:%M')
            trip_id = f"{source_center}{destination_center}{new_datetime.strftime('%Y%m%d%H%M')}"
            return render_template('result.html', trip_id=trip_id, source_cities=source_cities, source_states=source_states, destination_cities=destination_cities, destination_states=destination_states, actual_time=eta, rout_type=rout_type, trip_creation_time=trip_creation_time,actual_time_display=actual_time_display)
         
        
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', error="There is no trip in this route")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
