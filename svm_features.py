
from sklearn.model_selection import train_test_split
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


path = '###'
getdata = pd.read_csv(path, sep=',')
getdata = getdata.replace(to_replace='T2', value=2)
getdata = getdata.replace(to_replace='T1', value=1)

dataTarget = getdata['T']
data = getdata.drop('T', axis=1)

scaler = StandardScaler()
scaler.fit(data)
standardized = scaler.transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    standardized, dataTarget, test_size=0.29, random_state=173, stratify=dataTarget)  # datu sadalīšana apmacības un testēšanas kopām. 70% apmacībai un 30 testēšanai.


model = SVC(C=25,gamma=0.01, kernel='sigmoid')# modeļa izveidošana ar hiperparametriem, ar kuriem hiperparametru pielāgošanas laikā modeļa KKP bija vislielākais

y_score = model.fit(X_train, y_train) # modeļa apmacība

perm = PermutationImportance(y_score).fit(X_test, y_test)  # svarīgāko atribūtu noteikšana ar permutācijas atribūtu svarīgumu

eli5.show_weights(perm, feature_names=data.columns.values) # svaru attelojums
