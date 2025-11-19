############################################################################
# Código de Entrenamiento 
############################################################################

import pandas as pd
from xgboost import XGBClassifier
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df.drop(['Churn'],axis=1)
    y_train = df[['Churn']]
    print(filename, ' cargado correctamente')

    # Entrenamos el modelo con toda la muestra
    xgb_mod=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                        colsample_bytree=1, gamma=0, learning_rate=0.9, max_delta_step=0,
                        max_depth=7, min_child_weight=1, n_estimators=100,
                        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                        silent=True, subsample=1)
    xgb_mod.fit(X_train, y_train)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(xgb_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('telcom_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()