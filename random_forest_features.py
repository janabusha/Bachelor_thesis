from sklearn.model_selection import train_test_split
import eli5
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance

path = '###'
getdata = pd.read_csv(path, sep=',')

dataTarget = getdata['T'] # mērķa atribūtu kopa
data = getdata.drop('T', axis=1) #atribūtu kopa

X_train, X_test, y_train, y_test = train_test_split(
    data, dataTarget, test_size=0.29, random_state=80, stratify=dataTarget) # datu sadalīšana apmacības un testēšanas kopām. 70% apmacībai un 30 testēšanai.

model = RandomForestClassifier(bootstrap = True, max_depth=80, max_features= 5, min_samples_leaf= 3, min_samples_split=12, n_estimators=200) #modeļa izveidošana ar hiperparametriem, ar kuriem hiperparametru pielāgošanas laikā modeļa KKP bija vislielākais

y_score = model.fit(X_train, y_train) # modeļa apmacība

perm = PermutationImportance(y_score).fit(X_test, y_test)  # svarīgāko atribūtu noteikšana ar permutācijas atribūtu svarīgumu

eli5.show_weights(perm, feature_names = data.columns.values) # svaru attelojums