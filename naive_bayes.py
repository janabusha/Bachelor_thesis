
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV



path = '###'
savePath = '###'
getdata = pd.read_csv(path, sep=',')
getdata = getdata.drop('ID', axis=1)
getdata = getdata.replace(to_replace='T2', value=2)
getdata = getdata.replace(to_replace='T1', value=1)

dataTarget = getdata['T'] #mērķa atribūtu kopa
data = getdata.drop('T', axis=1) # atribūtu kopa

random_seed = [65, 97, 111, 15, 54, 27, 193, 71, 84, 198, 191, 17, 74, 172, 6, 63, 33, 194, 7, 1, 18, 16, 152, 26,
               158, 72, 52, 98, 76, 296, 163, 190, 200, 31, 93, 12, 123, 180, 181, 173, 147, 137, 3, 178, 5, 186, 88,
               20, 101, 124, 95, 41, 10, 81, 102, 80, 103, 47, 165, 121, 141, 133, 159, 34, 73, 120, 119, 75, 151, 109,
               23, 100, 115, 25, 188, 64, 155, 177, 60, 42, 96, 164, 108, 170, 45, 37, 185, 192, 99, 2, 175, 56, 148,
               14, 19, 116, 4, 162, 168, 44
               ]


newfile = pd.DataFrame({}) #tukšas datnes izveide

for seed in random_seed:

    X_train, X_test, y_train, y_test = train_test_split(
        data, dataTarget, test_size=0.29, random_state=seed, stratify=dataTarget)  # datu sadalīšana apmacības un testēšanas kopām. 70% apmacībai un 30 testēšanai.

    for scor in ['roc_auc','accuracy']: # precizitātes mērs pēc kura tiek pielāgoti hiperparametri
        model = GaussianNB() #modeļa izveide
        params = {}
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=cv, scoring=scor, error_score=0)
        grid_result = grid_search.fit(X_train, y_train)
        best_model = grid_result.best_estimator_

        y_score = best_model.fit(X_train, y_train) #modeļa apmacība

        y_pred = y_score.predict(X_test) #merķa atribūtu prognozēšana

        if scor == 'roc_auc':
            newrow = {'CV_'+scor: grid_result.best_score_,
                      'Test_'+scor: metrics.roc_auc_score(y_test, y_pred)}
            newfile = newfile.append(newrow, ignore_index=True)
        else:
            newrow = {'CV_'+scor: grid_result.best_score_,
                      'Test_'+scor: metrics.accuracy_score(y_test, y_pred)}
            newfile = newfile.append(newrow, ignore_index=True)# jaunās rindas pievienošana datnei

        newfile.to_csv(savePath, mode='w', index=False) # datnes saglabāšana
