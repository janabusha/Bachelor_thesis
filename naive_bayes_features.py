
from sklearn.model_selection import train_test_split
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.naive_bayes import GaussianNB


path = '###'
getdata = pd.read_csv(path, sep=',')

dataTarget = getdata['T'] # merka atribūtu kopa
data = getdata.drop('T', axis=1) #atribūtu kopa

X_train, X_test, y_train, y_test = train_test_split(
    data, dataTarget, test_size=0.29, random_state=116, stratify=dataTarget)   # datu sadalīšana apmacības un testēšanas kopām. 70% apmacībai un 30 testēšanai.

model = GaussianNB() # modela izveidošana


y_score = model.fit(X_train, y_train) # modela apmacība

perm = PermutationImportance(y_score, random_state=1).fit(X_test, y_test) # svarīgāko atribūtu noteikšana ar permutācijas atribūtu svarīgumu

eli5.show_weights(perm, feature_names = X_test.columns.values) # svaru attelojums
