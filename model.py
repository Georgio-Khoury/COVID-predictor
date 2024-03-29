import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('.\dataset.csv')
y=df['COVID-19']
X = df.iloc[:,0:20]
encodr = LabelEncoder()
y = encodr.fit_transform(y)
for col in X.columns[:20]:
    X.loc[:,col] = encodr.fit_transform(X[col])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)
joblib.dump(rf_classifier, 'trained_model.pkl')
# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)
print('X_test:', X_test)
print(y_pred)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)

# Generate a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Predicted No', 'Predicted Yes'])
plt.yticks([0, 1], ['Actual No', 'Actual Yes'])
plt.show()