import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

class KNN4:
    @staticmethod
    def modelKNN4():
        data = pd.read_csv('Diabetes.csv')

        cols_to_replace = ['PGL', 'DIA', 'TSF', 'INS', 'BMI']

        data[cols_to_replace] = data[cols_to_replace].replace(0, pd.NA).fillna(data.mean())

        X = data.drop(['Diabetic'], axis=1)
        y = data['Diabetic']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        k_values = [3, 6, 10, 13]

        results = {}

        for k in k_values:
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(X_train, y_train)

            predictions_knn = knn_model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions_knn)
            roc_auc = roc_auc_score(y_test, predictions_knn)
            cm = confusion_matrix(y_test, predictions_knn)

            results[k] = {'Accuracy': accuracy, 'ROC AUC': roc_auc, 'Confusion Matrix': cm}

        results_df = pd.DataFrame.from_dict(results, orient='index')
        print(results_df)