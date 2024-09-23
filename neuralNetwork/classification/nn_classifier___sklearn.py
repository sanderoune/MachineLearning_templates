# Multilayer perceptron classifier for Titanic survivors dataset.
# The multi-layer perceptron model of skLearn is implemented.

import pandas as pd
import seaborn as sns # library for statistical data visualization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

print(train_data.isnull().sum())

survived = train_data['Survived'].value_counts()[1]
not_survived = train_data['Survived'].value_counts()[0]

print(survived)
print(not_survived)
survival_count = [survived, not_survived]
labels = ['Survived', 'Not survived']

plt.figure(figsize=(10, 6))
plt.bar(labels, survival_count, color ='maroon', 
        width = 0.4)
plt.title('Survival Distribution')
plt.savefig('./survivors', dpi = 300)
plt.close()

# Input and output data
X = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
y = train_data['Survived']

# Categorical and numerical columns
numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_cols = ['Sex', 'Embarked', 'Pclass']

# Transform data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# model definition
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
])

# trai/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

test_pred = model_pipeline.predict(test_data)

submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': test_pred})
submission.to_csv('submission.csv', index=False)