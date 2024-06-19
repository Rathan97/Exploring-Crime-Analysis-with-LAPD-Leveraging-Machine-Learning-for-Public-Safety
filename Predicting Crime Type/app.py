from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'C:\\Users\\LENOVO L460\\project\\crime analysis\\random_forest_classifier_model.pkl'
with open(filename, 'rb') as f:
    rf_classifier = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('mains.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        AREA = int(request.form['areaName'])
        Time_of_Day_Encoded = int(request.form['timeOfDay'])
        DATE_OCC_Day_of_Week_Encoded = int(request.form['dayOfWeek'])
        DATE_OCC_Week = int(request.form['week'])
        Season_Encoded = int(request.form['SeasonType'])
        On_Holidays_Encoded = int(request.form['holiday'])
        Hour = int(request.form['hour'])
        Weapon_Used_Category_Encoded = int(request.form['weaponCategory'])
        Vict_Age = int(request.form['victimAge'])
        Vict_Sex = int(request.form['victimSex'])
        
        data = np.array([[AREA, Time_of_Day_Encoded, DATE_OCC_Day_of_Week_Encoded, DATE_OCC_Week, Season_Encoded, On_Holidays_Encoded,
                            Hour, Weapon_Used_Category_Encoded, Vict_Age, Vict_Sex]])
        
        predictions = rf_classifier.predict(data)
        predicted_crime_clean = predictions[0].strip("[]' ")
        return render_template('results.html', predictions=predicted_crime_clean)

if __name__ == '__main__':
    app.run(debug=True)
