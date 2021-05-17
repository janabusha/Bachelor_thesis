from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import eli5
from eli5.sklearn import PermutationImportance


tf.random.set_seed(1) # svaru inicializācija

path = '###'
getdata = pd.read_csv(path, sep=',')
getdata = getdata.replace(to_replace='T2', value=0)
getdata = getdata.replace(to_replace='T1', value=1)

dataTarget = getdata['T']
scaler = StandardScaler()

data = getdata.drop('T', axis=1)

scaler.fit(data)
print(data.columns.values)
standardized = scaler.transform(data)
print(standardized)

# modeļa izveidošana ar hiperparametriem, ar kuriem hiperparametru pielāgošanas laikā modeļa KKP bija vislielākais
model = keras.Sequential()
model.add(keras.layers.Dense(input_dim=15, units=96, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])



X_train, X_test, y_train, y_test = train_test_split(
    standardized, dataTarget, test_size=0.29, random_state=152, stratify=dataTarget)  # datu sadalīšana apmacības un testēšanas kopām. 70% apmacībai un 30 testēšanai.


model.fit(X_train, y_train, epochs=50) #modeļa apmācība

perm = PermutationImportance(model, random_state=1, scoring='explained_variance').fit(X_test, y_test) # svarīgāko atribūtu noteikšana ar permutācijas atribūtu svarīgumu

eli5.show_weights(perm, feature_names = data.columns.values)  # svaru attelojums