import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LR2:
    @staticmethod
    def model2():
        data = pd.read_csv('Diabetes.csv')

        X = data[['NPG']]
        y = data['AGE']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_LR2 = LinearRegression()

        model_LR2.fit(X_train, y_train)

        y_pred = model_LR2.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n")
        print(f"Mean Squared Error For Model 2: {mse}")
        print(f"R-squared Score For Model 2: {r2}")
        print("\n")
