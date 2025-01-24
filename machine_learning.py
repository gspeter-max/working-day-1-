import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate a high-dimensional dataset
x, y = make_classification(
    n_samples=200,         # Small dataset
    n_features=100,        # High-dimensional feature space
    n_informative=15,      # Only a few informative features
    n_redundant=10,        # Some redundant features
    n_classes=2,           # Binary classification
    class_sep=0.3,         # Classes are not well-separated
    flip_y=0.1,            # Add some label noise
    random_state=42
)

# Save the dataset to a DataFrame
df = pd.DataFrame(x, columns=[f"gene_{i}" for i in range(1, 101)])
df['disease'] = y

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state =  42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


from sklearn.svm  import SVC
from sklearn.model_selection  import GridSearchCV
from sklearn.metrics import classification_report

svc = SVC(
    kernel = "rbf",
    probability = True, 
    random_state = 42 
)
svc.fit(x_train_scaled,y_train)
y_probality = svc.predict(x_test)

print(f"classification report : {classification_report(y_test,y_probality)}")
print(y_probality)

from sklearn.model_selection import GridSearchCV
param_grid  = {
    "C" : [0.1,0.5,0.3,1],
    'gamma' : [0.001,0.1,0.001,1]
}
grid_search = GridSearchCV(
   estimator = svc,  param_grid = param_grid , verbose = 2,cv = 50,scoring = 'roc_auc'
)
grid_search.fit(x_train_scaled, y_train)
bast_model = grid_search.best_params_
print(dir(grid_search))
print(f"bast parameters : {grid_search.best_params_}")
bast_estimator = grid_search.best_estimator_
y_prediction = bast_estimator.predict(x_test)
print(f"classification score : {classification_report(y_test,y_prediction)}")

