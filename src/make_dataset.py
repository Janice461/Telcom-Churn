###################################
# Script de Preparación de Datos
###################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
        #Removing correlated and unneccessary columns
    col_to_drop = ['State', 'Area code', 'Total day charge', 'Total eve charge', 
                'Total night charge', 'Total intl charge']
    
    df = df.drop(columns = col_to_drop, axis = 1)

    #target column
    target_col = ["Churn"]

    #Separating categorical and numerical columns
    #categorical columns
    cat_cols = list(set(df.nunique()[df.nunique()<6].keys().tolist() 
                        + df.select_dtypes(include='object').columns.tolist()))
    cat_cols = [x for x in cat_cols if x not in target_col]
    #numerical columns
    num_cols = [x for x in df.columns if x not in cat_cols + target_col]
    #Binary columns with 2 values
    bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()

    #Label encoding Binary columns
    le = LabelEncoder()
    for i in bin_cols:
        df[i] = le.fit_transform(df[i])

    #Scaling Numerical columns
    std = StandardScaler()
    scaled = std.fit_transform(df[num_cols])
    scaled = pd.DataFrame(scaled, columns=num_cols)

    #dropping original values merging scaled values for numerical columns
    df = df.drop(columns = num_cols, axis = 1)
    df = df.merge(scaled, left_index=True, right_index=True, how = "left")

    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df,  features,filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('churn-bigml-train.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1,[['International plan', 'Voice mail plan', 'Account length',
        'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total eve minutes', 'Total eve calls', 'Total night minutes',
        'Total night calls', 'Total intl minutes', 'Total intl calls',
        'Customer service calls', 'Churn']],'telcom_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('churn-bigml-test.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2,[['International plan', 'Voice mail plan', 'Account length',
        'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total eve minutes', 'Total eve calls', 'Total night minutes',
        'Total night calls', 'Total intl minutes', 'Total intl calls',
        'Customer service calls', 'Churn']],'telcom_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('churn-bigml-predict.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3,[['International plan', 'Voice mail plan', 'Account length',
        'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total eve minutes', 'Total eve calls', 'Total night minutes',
        'Total night calls', 'Total intl minutes', 'Total intl calls',
        'Customer service calls']], 'telcom_churn.csv')
    
if __name__ == "__main__":
    main()