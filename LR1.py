from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
class LR1:
    @staticmethod
    def model1():
        file_path = "Diabetes.csv"
        data = pd.read_csv(file_path)

        cols_to_replace = ['PGL', 'DIA', 'TSF', 'BMI']
        data[cols_to_replace] = data[cols_to_replace].replace(0, pd.NA).fillna(data.mean())

        X = data.drop(['AGE'], axis=1)
        y = data['AGE']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        model_LR1 = LinearRegression()

        model_LR1.fit(X_train, y_train)

        y_pred = model_LR1.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r1 = r2_score(y_test, y_pred)

        print("\n")
        print(f"Mean Squared Error For Model 1: {mse}")
        print(f"R-squared Score For Model 1: {r1}")
        print("\n")