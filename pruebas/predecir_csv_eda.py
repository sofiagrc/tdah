
#%% -------------------------------------------------------------------------------------------------------------
#  Machine Learning con scikit-learn 
# SVM
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


from crear_csv_eda import compute_eda
X,y, salida_x, salida_y, salida= compute_eda()


#TRAIN, TEST SPLIT --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split( # divide los datos en conjunto de entrenamiento y test
    X, y, test_size=0.33, random_state=42)



#X_train_raw = X_train.copy()


# NORMALIZAR DATOS --------------------------------------------------------------

print("NORMALIZANDO CON STANDARDSCALER")
#print(f"Sin Normalizar:{X_train} ")
scaler = StandardScaler()
scaler.fit(X_train)
#print(scaler.mean_)
X_train=scaler.transform(X_train)
X_test= scaler.transform(X_test)
print(f"Normalizada:{X_train} ")


# PREDICCION CON SVC ------------------------------------------------------------

print(f" EMPIEZA LA PREDICCION CON SVC")
clf = svm.SVC()
clf.fit(X_train, y_train) 


#%% ACCURACY 
from sklearn.metrics import accuracy_score 
y_pred = clf.predict(X_test)
y_true = y_test  # convertir a array 1D
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
# %% PRECISION ---------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
from sklearn.metrics import precision_score
y_true = y_test  # convertir a array 1D
print(f"Test: {y_test}")
y_pred= clf.predict(X_test)
print(f"Predict: {y_pred}")
p1 = precision_score(y_true, y_pred, average='macro')
p2 = precision_score(y_true, y_pred, average='micro')
p3 = precision_score(y_true, y_pred, average='weighted')
p4= precision_score(y_true, y_pred, average=None)
print(f"Macro Precision: {p1}")
print(f"Micro Precision: {p2}")   
print(f"Weighted Precision: {p3}")
print(f"Precision por clase (None){p4}")

# %% RECALL -----------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
from sklearn.metrics import recall_score
y_true = y_test  # convertir a array 1D
y_pred= clf.predict(X_test)
r1 = recall_score(y_true, y_pred, average='macro')
r2 = recall_score(y_true, y_pred, average='micro')
r3 = recall_score(y_true, y_pred, average='weighted')
r4= recall_score(y_true, y_pred, average=None)
print(f"Macro Recall: {r1}")
print(f"Micro Recall: {r2}")   
print(f"Weighted Recall: {r3}")
print(f"Recall por clase (None): {r4}")
# %% SPECIFICITY -------------------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
y_true = y_test  # convertir a array 1D
y_pred= clf.predict(X_test)
matrix= confusion_matrix(y_true, y_pred, normalize='all')
tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel().tolist()
print(tn,fp,fn,tp)
specificity = tn / (tn + fp)
print(f"Specificity: {specificity}")
reca= recall_score(y_true, y_pred, average='binary')
print(f"Recall formula: {tp/(tp+fn)}")
print(f"Recall: {reca}")
#%% FSCORE --------------------------------------------------------------------------------------------------------
from sklearn.metrics import precision_recall_fscore_support
y_true = y_test  # convertir a array 1D
y_pred= clf.predict(X_test)
precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(f"Macro Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")
precision_recall_fscore_support(y_true, y_pred, average='micro')
print(f"Micro Fscore--> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")
precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"Weighted Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")


# %% AUROC -------------------------------------------------------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

#roc_auc_score(y_true, clf.predict_proba(X_test)[:, 1]) # tiene que ser un clasificador de decisión o que de probabilidades cvm no lo hace
auroc= roc_auc_score(y_true, clf.decision_function(X_test))
print(f"AUROC: {auroc}")


#===============================================================================================================
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================
#===============================================================================================================

# %% ---------------------------------------------------K-FOLD
from sklearn.model_selection import KFold
# hhacer tabla con los valores, no direcetamente la media

tabla_final_medias=[]

fila=[]
kf = KFold(n_splits=5)

acc=[]
prec=[]
rec=[]
spec=[]
fsc=[]
aur=[]

for train_ix, test_ix in kf.split(X):

    #print("%s %s"% (train_ix,test_ix))
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    
    y_train = y.iloc[train_ix].values.ravel()
    y_test  = y.iloc[test_ix].values.ravel()
    #print(X_train, X_test, y_train, y_test)
        
    clf = svm.SVC()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    acc.append(accuracy)
    #print(f"Accuracy: {accuracy}")

    p1 = precision_score(y_test, y_pred, average='binary')
    prec.append(p1)
    #print(f"Macro Precision: {p1}")
   
    r1 = recall_score(y_test, y_pred, average='binary')
    rec.append(r1)
    #print(f"Macro Recall: {r1}")

    matrix= confusion_matrix(y_test, y_pred, normalize='all')
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel().tolist()
    specificity = tn / (tn + fp)
    spec.append(specificity)
    #print(f"Specificity: {specificity}")

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    fsc.append(fscore)
    #print(f"Macro Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")

    
    
    aur.append(auroc)
    #print(f"AUROC: {auroc}")

    fila.append({
        "Accuracy": accuracy,
        "Precision": p1,
        "Recall": r1,
        "Specificity": specificity,
        "FScore": fscore,
        "AUROC": auroc
    }
    )
    df_folds = pd.DataFrame(fila)


   
print("TABLA K-FOLD")
print(df_folds)
print(f"MEDIA ACCURACY: {sum(acc)/len(acc)}")
print(f"MEDIA PRECISION: {sum(prec)/len(prec)}")
print(f"MEDIA RECALL: {sum(rec)/len(rec)}")
print(f"MEDIA SPECIFICITY: {sum(spec)/len(spec)}")
print(f"MEDIA FSCORE: {sum(fsc)/len(fsc)}")
print(f"MEDIA AUROC: {sum(aur)/len(aur)}")

tabla_final_medias.append({
    "Media Accuracy":sum(acc)/len(acc),
    "Media Precision": sum(prec)/len(prec),
    "Media Recall": sum(rec)/len(rec),
    "Media Specificity": sum(spec)/len(spec),
    "Media FScore": sum(fsc)/len(fsc),
    "Media AUROC": sum(aur)/len(aur)
})
df_folds2 = pd.DataFrame(tabla_final_medias)


# %% -----------------------------------------STRATIFIED KFOLD

from sklearn.model_selection import StratifiedKFold
skf= StratifiedKFold(n_splits=5)
skf.get_n_splits(X,y)

fila3=[]

for fold_ix, (train_ix, test_ix) in enumerate(skf.split(X,y)):
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]


    y_train = y.iloc[train_ix].values.ravel()
    y_test  = y.iloc[test_ix].values.ravel()

    clf = svm.SVC()
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy}")
    acc.append(accuracy)

    p1 = precision_score(y_test, y_pred, average='binary')
    prec.append(p1)
    #print(f"Macro Precision: {p1}")
   
    r1 = recall_score(y_test, y_pred, average='binary')
    rec.append(r1)
    #print(f"Macro Recall: {r1}")

    matrix= confusion_matrix(y_test, y_pred, normalize='all')
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel().tolist()
    specificity = tn / (tn + fp)
    spec.append(specificity)
    #print(f"Specificity: {specificity}")

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')
    fsc.append(fscore)
    #print(f"Macro Fscore --> Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")

    auroc= roc_auc_score(y_test, clf.decision_function(X_test))
    aur.append(auroc)
    #print(f"AUROC: {auroc}")
    fila3.append({
        "Accuracy": accuracy,
        "Precision": p1,
        "Recall": r1,
        "Specificity": specificity,
        "FScore": fscore,
        "AUROC": auroc
    }
    )
    df_folds = pd.DataFrame(fila3)

print("TABLA STRATIFIED K-FOLD")
print(df_folds)

print(f"MEDIA ACCURACY: {sum(acc)/len(acc)}")
print(f"MEDIA PRECISION: {sum(prec)/len(prec)}")
print(f"MEDIA RECALL: {sum(rec)/len(rec)}")
print(f"MEDIA SPECIFICITY: {sum(spec)/len(spec)}")
print(f"MEDIA FSCORE: {sum(fsc)/len(fsc)}")
print(f"MEDIA AUROC: {sum(aur)/len(aur)}")

tabla_final_medias.append({
    "Media Accuracy":sum(acc)/len(acc),
    "Media Precision": sum(prec)/len(prec),
    "Media Recall": sum(rec)/len(rec),
    "Media Specificity": sum(spec)/len(spec),
    "Media FScore": sum(fsc)/len(fsc),
    "Media AUROC": sum(aur)/len(aur)
})
df_folds2 = pd.DataFrame(tabla_final_medias)


#%% -------------------------------------------------------------------------------------------------------------------------
# PREPROCESADO 
# - NORMALIZAR
# - NULL
# -


