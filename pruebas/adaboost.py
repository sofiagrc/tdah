
#%% -------------------------------------------------------------------------------------------------------------
#  Machine Learning con scikit-learn 
# SVM
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score   
from pathlib import Path    

# LEYENDO ARCHIVOS --------------------------------------------------------------

path_eda_x = "C:/Users/pprru/Desktop/salidas_eda_nueva/features_eda_all_users.csv"
path_eda_y = "C:/Users/pprru/Desktop/salidas_eda_nueva/labels_eda_all_users.csv"
path_pow_x = "C:/Users/pprru/Desktop/salidas_eda_nueva/features_robots_all_users.csv"
path_pow_y = "C:/Users/pprru/Desktop/salidas_eda_nueva/labels_robots_all_users.csv"

def read_archivo(path: str):
    print(f"Leyendo archivo: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df



def obtener_usuarios_pow(path: str = "C:/Users/pprru/Desktop/salidas_eda_nueva/bandpower_robots_all_users.csv"):
    print(f"Leyendo archivo: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    users = df.pop("user")
    return users

def obtener_usuarios_eda(path: str = "C:/Users/pprru/Desktop/salidas_eda_nueva/tabla_eda_con_diagnostico.csv"):
    print(f"Leyendo archivo: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    users = df.pop("username")
    return users


def normalizar_datos(X_train, X_test):
    print("NORMALIZANDO CON STANDARDSCALER")
    scaler = StandardScaler() # zscore normalization: media 0, desviacion tipica 1 (z=(x-mean)/std)
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test= scaler.transform(X_test) # nunca se usan datos de test para ajustar el scaler
    return X_train, X_test


# PREDICCION CON Clasificadores -------------------------------------------------
classifier = {
    "SVC": svm.SVC(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
}

def clasificar(X_train, X_test, y_train, y_test):
    metricas_clasificador = {}
    for name, clf in classifier.items():
                print(f"Usando el clasificador: {name}")

                # ENTRENAR 
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # EVALUAR
                accuracy = accuracy_score(y_test, y_pred)
                p1 = precision_score(y_test, y_pred, average='binary')
                r1 = recall_score(y_test, y_pred, average='binary')
                matrix= confusion_matrix(y_test, y_pred, normalize='all')
                tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel().tolist()
                specificity = tn / (tn + fp)
                precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
                auroc="-"
                if(not name=="KNN"):
                    auroc= roc_auc_score(y_test, clf.decision_function(X_test))
                metricas_clasificador[name] = {"accuracy":accuracy, 
                               "precision":precision, 
                               "recall":recall,
                                "specificity": specificity,
                                "fscore": fscore,
                                "auroc":auroc
                                }
    return metricas_clasificador

    





def clasificacion_final (path_x : Path, path_y:Path, tipo:str):

    X =read_archivo(path_x)
    y =read_archivo(path_y)

    

    if tipo=="pow":
        groups = obtener_usuarios_pow()
    else:
        groups = obtener_usuarios_eda()

    validation = {
    "kfold":KFold(n_splits=5),
    "groupkfold":GroupKFold(n_splits=5),
    "stratifiedkfold":StratifiedKFold(n_splits=5),
    "stratifiedgroupkfold":StratifiedGroupKFold(n_splits= 5),
    }
    completo = {}
    
  

    for val_name, val in validation.items():

        resultados_folds = []   # de cada fold se guarda un dataframe 

        print(f" Usando validación cruzada: {val_name}")
        if(val_name=="kfold"):
            for train_ix, test_ix in val.split(X):

                #print("%s %s"% (train_ix,test_ix))
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                
                y_train = y.iloc[train_ix].values.ravel()
                y_test  = y.iloc[test_ix].values.ravel()
                X_train, X_test = normalizar_datos(X_train, X_test)


                #print(X_train, X_test, y_train, y_test)
                metricas = clasificar(X_train, X_test, y_train, y_test)
                print("La lista se ha actualizado con K-Fold")


               

        elif(val_name=="stratifiedkfold"): 
            for fold_ix, (train_ix, test_ix) in enumerate(val.split(X,y)):
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                X_train, X_test = normalizar_datos(X_train, X_test)

                
                y_train = y.iloc[train_ix].values.ravel()
                y_test  = y.iloc[test_ix].values.ravel()
                metricas = clasificar(X_train, X_test, y_train, y_test)
                print("La lista se ha actualizado con  Stratified K-Fold")

                
        
        elif(val_name=="groupkfold" ):
            for train_ix, test_ix in val.split(X, y, groups=groups):
                #print("%s %s"% (train_ix,test_ix))
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                X_train, X_test = normalizar_datos(X_train, X_test)


                y_train = y.iloc[train_ix].values.ravel()
                y_test  = y.iloc[test_ix].values.ravel()
                
                
                #print(X_train, X_test, y_train, y_test) 
                metricas = clasificar(X_train, X_test, y_train, y_test)
                print("La lista se ha actualizado con Group K-Fold")

            
    
        elif val_name == "stratifiedgroupkfold":
            for train_ix, test_ix in val.split(X, y.values.ravel(), groups=groups):
                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                X_train, X_test = normalizar_datos(X_train, X_test)

                y_train = y.iloc[train_ix].values.ravel()
                y_test  = y.iloc[test_ix].values.ravel()

                metricas = clasificar(X_train, X_test, y_train, y_test)
                print("La lista se ha actualizado con Stratified Group K-Fold")

                
        
        




        completo[val_name].append(metricas)


    print("RESULTADOS FINALES:")
    tabla_completa = pd.DataFrame.from_dict(completo) # convierte el diccionario en un dataframe
    for val_name, resultados in completo.items():
        print(f" Validación cruzada: {val_name}")
        tabla = pd.DataFrame.from_dict(resultados) # convierte el diccionario en un dataframe
        print(tabla)    







if __name__ == "__main__":
   
    #clasificacion_final(path_eda_x, path_eda_y, "eda")
    clasificacion_final(path_pow_x, path_pow_y, "pow")
                    

   




