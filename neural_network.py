from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

tf.random.set_seed(1) # svaru inicializācija

path = "###"
savePath = "###"
getdata = pd.read_csv(path, sep=',')
# getdata = getdata.drop('ID', axis=1)
getdata = getdata.replace(to_replace='T2', value=0)
getdata = getdata.replace(to_replace='T1', value=1)

dataTarget = getdata['T'] # mērķa atribūtu kopa
scaler = StandardScaler()
data = getdata.drop('T', axis=1) # atribūtu kopa


scaler.fit(data)
standardized = scaler.transform(data) #datu normalizācija izmantojot ar z-novērtējuma normalizāciju


random_seed = [65, 97, 111, 15, 54, 27, 193, 71, 84, 198, 191, 17, 74, 172, 6, 63, 33, 194, 7, 1, 18, 16, 152, 26,
               158, 72, 52, 98, 76, 296, 163, 190, 200, 31, 93, 12, 123, 180, 181, 173, 147, 137, 3, 178, 5, 186, 88,
               20, 101, 124, 95, 41, 10, 81, 102, 80, 103, 47, 165, 121, 141, 133, 159, 34, 73, 120, 119, 75, 151, 109,
               23, 100, 115, 25, 188, 64, 155, 177, 60, 42, 96, 164, 108, 170, 45, 37, 185, 192, 99, 2, 175, 56, 148,
               14, 19, 116, 4, 162, 168, 44
               ]

newfile = pd.DataFrame({}) # tukšas datnes izveidošana

print(data.shape)

def model_builder(hp): # neironu tīkla izveidošana
    model = keras.Sequential()

    hp_units = hp.Int('units', min_value=32, max_value=128, step=32) # iespējamais neironu skaits slēptājā slānī.

    model.add(keras.layers.Dense(input_dim=42, units=hp_units, activation=hp.Choice(
        'dense_activation',
        values=['relu', 'tanh', 'sigmoid']))) # iespējamas aktivācijas funkcijas sleptājā slānī.
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # iespējamie apmācības koeficienti.

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy']) #modeļa kompilēšana

    return model


for seed in random_seed:
    X_train, X_test, y_train, y_test = train_test_split(
        standardized, dataTarget, test_size=0.29, random_state=seed, stratify=dataTarget)

    tuner = kt.BayesianOptimization(  # metode optimālo hiperparametru noteikšanai.
        model_builder,
        objective='val_accuracy',
        max_trials=10,
        seed=42,
        executions_per_trial=2,
        overwrite=True)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) # algoritms beigs apmācīties, ja pēdējo piecu iterāciju laikā, validācijas kļūda paliek nemainīga

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])


    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]    # optimālo hiperparametru saglabāšana
    best_activation = "Activation: " + best_hps.get('dense_activation')
    best_units = ". Units: " + str(best_hps.get('units'))
    best_l_rate = ". Learning rate: " + str(best_hps.get('learning_rate'))

    hps = best_activation + best_units + best_l_rate

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)  # modeļa veidošana ar optimālajiem hiperparametriem

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    hypermodel = tuner.hypermodel.build(best_hps)


    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2) # modeļa atkārtota apmācība.

    eval_result = hypermodel.evaluate(X_test, y_test) # modeļa precizitātes noteikšana uz testa datiem
    y_pred = hypermodel.predict(X_test)

    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred)
    auc_rf = auc(fpr_rf, tpr_rf) # laukuma zem tīkla noteikšana


    print("auc ", auc_rf)
    print("[test loss, test accuracy]:", eval_result)
    newrow = {'Best_params' : hps,
              'Validation':max(val_acc_per_epoch),
              'Test': eval_result[1],
              'AUC': auc_rf,
              }
    newfile = newfile.append(newrow, ignore_index=True) # rindas pievienošana MS Excel datnē

    newfile.to_csv(savePath, mode='w', index=False) # datnes saglabāšana
