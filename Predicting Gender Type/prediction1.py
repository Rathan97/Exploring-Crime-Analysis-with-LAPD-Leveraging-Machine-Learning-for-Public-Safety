from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

data  = pd.read_csv("C:\\Users\\saira\\OneDrive\\Desktop\\new\\crime_cleaned_data.csv")

label_encoder = LabelEncoder()
for column in ['DATE OCC Day of Week']:
    encoded_column = column + ' Encoded'
    data[encoded_column] = label_encoder.fit_transform(data[column])
    
data1 = data.select_dtypes(['int','float'])

data1 = data1[data1['Vict Sex Encoded'] != 4]


X =data1[data1['Vict Sex Encoded'] != 4][['AREA','Vict Descent Encoded', 'Hour','Time of Day Encoded','Part 1-2',
                  'On Holidays Encoded','Vict Age','Season Encoded','DATE OCC Day of Week Encoded']] 
y =data1['Vict Sex Encoded']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()


rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)


filename = 'genders.pkl'
with open(filename, 'wb') as f:
    pickle.dump(rf_classifier, f)

