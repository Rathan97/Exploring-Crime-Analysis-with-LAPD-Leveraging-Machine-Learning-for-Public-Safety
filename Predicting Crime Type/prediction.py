from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load your dataset
# Assuming you have already loaded your dataset into a DataFrame 'df'

 # Assuming 'crime_type_encoded' is the encoded target variable
# Separate features (X) and target variable (y)
# X = data1[['AREA','Time of Day Encoded','DATE OCC Day of Week Encoded','DATE OCC Week','Vict Age','Vict Sex Encoded','On Holidays Encoded', 'Weapon Used Cd','Hour']]
# y = data1['crime_type']  # Assuming 'crime_type_encoded' is the encoded target variable
data = pd.read_csv('cleaned_data_crime1.csv')
X = data[['AREA','Time of Day Encoded','DATE OCC Day of Week Encoded','DATE OCC Week','Season Encoded','On Holidays Encoded', 'Hour','Weapon Used Cd','Vict Age','Vict Sex Encoded']]
y = data['crime_type']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize the Random Forest Classifier model
rf_classifier = RandomForestClassifier()

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
