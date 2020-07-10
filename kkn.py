#Maram Alshehri - Computer Scince Student 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#loading dataset from local file
iris_data= pd.read_csv('iris.csv')
iris_data.info()

#getting the features names only not the target
x_features= list(iris_data.columns.values.tolist())[:-1]

y=iris_data["species"]
x=iris_data[x_features]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
print(X_train.values[0])
print(X_test.values[0])


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=1)
#classifier = KNeighborsClassifier(n_neighbors=3)
#classifier = KNeighborsClassifier(n_neighbors=5)
#classifier = KNeighborsClassifier(n_neighbors=10)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))



