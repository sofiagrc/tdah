#%%
import numpy as np
from sklearn.model_selection import train_test_split
X,y = np.arange(10).reshape((5,2)), range(5)
X

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
X_train
y_train
X_test
y_test
# %%
train_test_split(y, shuffle=False)
# %%
from sklearn import datasets
iris = datasets.load_iris(as_frame=True)
X, y = iris['data'], iris['target']
X.head()
y.head()
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
print(X_train.head())
print(y_train.head())
print(X_test.head())
y_test.head()

# %%
# hacer lo mismo con cross validation
# otra vez con subject validation 
# CROSS VALIDATION
from sklearn.model_selection import cross_val_score
from sklearn import metrics 
from sklearn.model_selection import cross_validate

clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)  # separa en 5 bloques 
print("%0.2f accuracy with a standar deviation of %0.2f" %(scores.mean(), scores.std()))

scoring= ['precision_macro', 'recall_macro']
scores= cross_validate(clf, X, y, scoring=scoring)
print(sorted(scores.keys()))
scores['test_recall_macro']


print("Recall_macro medio: %.3f (+/- %.3f)" % (
    scores['test_recall_macro'].mean(),
    scores['test_recall_macro'].std()
))
print("Precision_macro medio: %.3f (+/- %.3f)" % (
    scores['test_precision_macro'].mean(),
    scores['test_precision_macro'].std()
))


