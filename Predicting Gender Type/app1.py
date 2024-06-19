from flask import Flask, render_template, request
import pickle
import numpy as np

filename ='C:\\Users\\saira\\OneDrive\\Desktop\\gender\\genders.pkl'
with open(filename, 'rb') as f:
    loaded_classifiers = pickle.load(f)

app = Flask(__name__)
def get_gender(prediction):
    if prediction == 3:
        return "Male"
    elif prediction == 1:
        return "Female"
    else:
        return "Unknown"


@app.route('/')
def home():
	return render_template('main1.html')

@app.route('/Test', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        Hour= int(request.form['hour'])
        Vict_Age = int(request.form['victimAge'])
        AREA = int(request.form['areaName'])
        Vict_Descent_Encoded = int(request.form['victimDescent'])
        Season_Encoded = int(request.form['season'])
        DATE_OCC_Day_of_Week_Encoded = int(request.form['dayOfWeek'])
        Time_of_Day_Encoded=int(request.form['timeOfDay'])
        On_Holidays_Encoded=int(request.form['holiday'])
        Part=int(request.form['Crime'])
        
        data = np.array([[Hour, Vict_Age,AREA,
            Vict_Descent_Encoded, Season_Encoded,
                          DATE_OCC_Day_of_Week_Encoded,Time_of_Day_Encoded,On_Holidays_Encoded,Part]])
        
        predictions = loaded_classifiers.predict(data)
        predicted_gender = get_gender(predictions)
        return render_template('result1.html', predicted_gender=predicted_gender)

        
if __name__ == '__main__':
	app.run()
