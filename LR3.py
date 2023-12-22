import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LR3:
    @staticmethod
    def model3():
        data = pd.read_csv('Diabetes.csv')

        important_features = ['NPG', 'PGL', 'DIA']

        X = data[important_features]
        y = data['AGE']

        cols_to_replace = ['PGL', 'DIA']
        data[cols_to_replace] = data[cols_to_replace].replace(0, pd.NA).fillna(data.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        model_LR3 = LinearRegression()

        model_LR3.fit(X_train, y_train)

        y_pred = model_LR3.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n")
        print(f"Mean Squared Error For Model 3: {mse}")
        print(f"R-squared Score For Model 3: {r2}")
        print("\n")

