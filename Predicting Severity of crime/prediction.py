import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


data=pd.read_excel('C:\\Users\\saira\\OneDrive\\Desktop\\new\\crime_cleaned_data.csv')

label_encoder = LabelEncoder()
for column in ['DATE OCC Day of Week']:
    encoded_column = column + ' Encoded'
    data[encoded_column] = label_encoder.fit_transform(data[column])


data['Crime Type'] = data['Part 1-2'].apply(lambda x: 'Misdemeanor' if x == 1 else 'Felony')

    
X =data[['Weapon Used Category Encoded','Vict Age','Vict Sex Encoded', 'AREA',
                  'Vict Descent Encoded','Season Encoded','DATE OCC Day of Week Encoded']]
y =data['Crime Type']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


filename = 'random_forest_model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(clf, f)
