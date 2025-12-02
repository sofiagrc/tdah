
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

n_neighbors = 5

pipelines = [
    ("KNN", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors))
    ])),
    ("SVM", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC())
    ])),
    ("AdaBoost", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", AdaBoostClassifier(n_estimators=100, random_state=0))
    ])),
]

for name, pipe in pipelines:
    
    pipe.fit(X_train, y_train)  #entrenar

    
    y_pred = pipe.predict(X_test) #predecir

    # METRICAS EVALUACION
    acc = accuracy_score(y_test, y_pred)
    print(f"Modelo: {name} - Accuracy: {acc:.3f}")



cv =[
    KFold(n_splits=5),
    GroupKFold(n_splits=5),
    StratifiedKFold(n_splits=5),
    StratifiedGroupKFold(n_splits= 5)
    ] 

for modelo in cv:
    for name, pipe in pipelines:
        

        

    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"\nModelo: {name}")
    
    print("  Scores por fold:", scores)
    print(f"  Media accuracy: {scores.mean():.3f}")
    print(f"  Desviación típica: {scores.std():.3f}")

