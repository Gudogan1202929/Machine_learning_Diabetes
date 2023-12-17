import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from KNN import KNN
from KNN4 import KNN4
from LR1 import LR1
from LR2 import LR2
from LR3 import LR3

file_path = "Diabetes.csv"
data = pd.read_csv(file_path)

cols_to_replace = ['PGL', 'DIA', 'TSF', 'INS', 'BMI']
data[cols_to_replace] = data[cols_to_replace].replace(0, pd.NA)
data.fillna(data.mean(), inplace=True)

summary_stats = data.describe()
print(summary_stats)

sns.countplot(x='Diabetic', data=data)
plt.title('Distribution of Diabetic Class')
plt.xlabel('Diabetic')
plt.ylabel('Count')
plt.show()

age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

data['Age_Group'] = pd.cut(data['AGE'], bins=age_bins, labels=age_labels, right=False)

diabetic_data = data[data['Diabetic'] == 1]

plt.figure(figsize=(10, 6))
sns.histplot(data=diabetic_data, x='Age_Group', bins=len(age_labels))
plt.title('Diabetics Distribution by Age Groups')
plt.xlabel('Age Groups')
plt.ylabel('Number of Diabetics')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.kdeplot(data['AGE'], fill=True)
plt.title('Density Plot for Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

plt.figure(figsize=(8, 6))
sns.kdeplot(data['BMI'], fill=True)
plt.title('Density Plot for BMI')
plt.xlabel('BMI')
plt.ylabel('Density')
plt.show()

numeric_data = data.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Features')
plt.show()

LR1.model1()
LR2.model2()
LR3.model3()

KNN.model1KNN()
KNN4.modelKNN4()
