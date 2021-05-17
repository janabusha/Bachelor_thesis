
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz


path = '###'
getdata = pd.read_csv(path, sep=',')

dataTarget = getdata['T'] #mērķa atribūtu kopa
data = getdata.drop('T', axis=1) #atribūtu kopa

X_train, X_test, y_train, y_test = train_test_split(
    data, dataTarget, test_size=0.29, random_state=124, stratify=dataTarget)  # datu sadalīšana apmacības un testēšanas kopām. 70% apmacībai un 30 testēšanai.

model = DecisionTreeClassifier(max_depth=8,max_leaf_nodes=38,min_samples_split=20) #modeļa izveidošana ar hiperparametriem, ar kuriem hiperparametru pielāgošanas laikā modeļa KKP bija vislielākais

y_score = model.fit(X_train, y_train) #modeļa apmacība

y_pred = y_score.predict(X_test) # mērķa atribūta prgnoze

# lēmuma koka vizualizācija
dot_data = tree.export_graphviz(y_score, out_file=None)
graph = graphviz.Source(dot_data)

dot_data = tree.export_graphviz(y_score, out_file=None,

                                feature_names=list(data.columns[:15].values),

                                class_names=['T1', 'T2'],
                                filled=True, rounded=True,

                                special_characters=True)
graph = graphviz.Source(dot_data)
print(dot_data)
