from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'C:\\Users\\saira\\OneDrive\\Desktop\\new\\random_forest_model.pkl'
with open(filename, 'rb') as f:
    loaded_classifiers = pickle.load(f)


app = Flask(__name__)
@app.route('/')
def home():
	return render_template('main.html')

@app.route('/Test', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        Weapon_Used_Category_Encoded = int(request.form['weaponCategory'])
        Vict_Age = int(request.form['victimAge'])
        Vict_Sex = int(request.form['victimSex'])
        AREA = int(request.form['areaName'])
        Vict_Descent_Encoded = int(request.form['victimDescent'])
        Season_Encoded = int(request.form['season'])
        DATE_OCC_Day_of_Week_Encoded = int(request.form['dayOfWeek'])
        
        data = np.array([[Weapon_Used_Category_Encoded, Vict_Age, Vict_Sex, AREA,
            Vict_Descent_Encoded, Season_Encoded,
                          DATE_OCC_Day_of_Week_Encoded]])
        
        predictions = loaded_classifiers.predict(data)
        return render_template('result.html', predictions=predictions)

        
if __name__ == '__main__':
	app.run()

