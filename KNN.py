import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNN:
    @staticmethod
    def model1KNN():
        data = pd.read_csv('Diabetes.csv')

        cols_to_replace = ['PGL', 'DIA', 'TSF', 'INS', 'BMI']
        data[cols_to_replace] = data[cols_to_replace].replace(0, pd.NA)
        data.fillna(data.mean(), inplace=True)

        X = data.drop(['Diabetic'], axis=1)
        y = data['Diabetic']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_train, y_train)

        predictions_knn = knn_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions_knn)

        print("\n")
        print("Accuracy Of KNN Model:", accuracy)
        print("\n")