"""
Toolbox for use in machine learning problems
Last modification: Aug 14 2025
@author: LupeVillegas
"""
import pandas as pd 
import numpy as np
import seaborn as sns  
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, f_oneway, kruskal, shapiro, levene
import matplotlib.pyplot as plt
import math


def describe_df(df):
    """
    Esta función describe las columnas de un DataFrame, devolviendo el porcentaje de nulos, valores únicos y cardinalidad

    Argumentos:
    df (DataFrame): DataFrame original

    Retorna:
    DataFrame: DataFrame con una descripción de cada variable de dataframe original
    """

    return pd.DataFrame([df.dtypes, np.round(df.isnull().sum() * 100 / len(df), 2), \
    df.nunique(), np.round(df.nunique()/len(df) * 100 ,2)]).rename(index = {0: "DATA_TYPE", 1: "MISSINGS (%)", 2: "UNIQUES_VALUES", 3: "CARDN (%)"})


       
def tipifica_variables(df, umbral_categoria, umbral_continua ):
    """
    Esta función da información de la variable y el tipo de variable

    Argumentos:
    df (DataFrame): DataFrame orginal
    umbral_categoria (int) : umbral de variable categorica
    umbral_continua (float): umbral de variable continua

    Retorna:
    DataFrame: DataFrame con dos columnas "nombre_variable" y "tipo_sugerido"
    """

    #funcion de clasificacion
    def clasificar_cardinalidad(c, umbral_categoria, umbral_continua):  
        if c == 2:
            return 'Binaria'
        elif c < umbral_categoria:
            return 'Categórica'
        elif c >= umbral_categoria:
            if c >= umbral_continua:
                return 'Numerica Continua'
            else:
                return 'Numerica Discreta'
        else:
            return 'Sin categoria'

    #Data frame con cardinalidad
    df_d = describe_df(df)
    df_description = df_d.T
    
    df_description['tipo_sugerido']= df_description ['UNIQUES_VALUES'].apply(clasificar_cardinalidad, args = (umbral_categoria, umbral_continua))

    return pd.DataFrame([df_description.index , df_description['tipo_sugerido']]).T.rename(columns = {0: "nombre_variable", 1: "tipo_sugerido"})


def get_features_num_regression(df, target_col, umbral_corr, pvalue = None):
    """
    Esta función retorna una lista de features con correlación mayor a umbral_corr. Si "pvalue" es distinta de None, devuelve features que además superan el test de hipótesis con el target.

    Argumentos:
    df (DataFrame)      : DataFrame original
    target_col (Series) : nombre de una de las columnas numéricas (variable numérica continua o discreta)
    umbral_corr (float) : umbral entre 0 y 1
    pvalue (float )     : pvalue de un test Pearson. Default = None

    Retorna:
    List: Lista con columnas numéricas del DataFrame 
    """
    
    # Check
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' no es columa del DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise TypeError(f"'{target_col}' debe ser una variable numérica continua o discreta.")
    
    if not isinstance(umbral_corr, (float, int)):
        raise TypeError("'umbral_corr' debe ser un float o int.")
    
    if pvalue is not None and not isinstance(pvalue, (float, int)):
        raise TypeError("'pvalue' debe ser un float o int.")

    #Correlacion
    corr_matrix = np.abs(df.corr(numeric_only= True)) 
    corr_features = corr_matrix.index[corr_matrix[target_col] > umbral_corr].tolist()
    
    if target_col in corr_features:
        corr_features.remove(target_col)
        
    if pvalue is None:
        return corr_features

    #Pvalue diferente de None
    selected_features= []
    for feature in corr_features:
        r, p = pearsonr(df[target_col], df[feature])
        if p <= pvalue:
            selected_features.append(feature)
            
    return selected_features


def plot_features_num_regression(df, target_col = "", columns = None, umbral_corr = 0.0, pvalue = None):
    """
    Esta función grafica las features con correlación mayor a umbral_corr. Si "pvalue" es distinta de None, devuelve features que además superan el test de hipótesis.

    Argumentos:
    df (DataFrame)      : DataFrame original
    target_col (Series) : nombre de una de las columnas. Default = ""
    columns (list)      : Lista de strings. Default  = [].
    umbral_corr (float) : umbral entre 0 y 1. Default = 0.
    pvalue (float )     : pvalue de un test Pearson. Default = None

    Retorna:
    List: Lista con columnas numéricas del DataFrame 
    """
    # Check
    
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' no es columa del DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise TypeError(f"'{target_col}' debe ser una variable numérica continua o discreta.")
    
    if not isinstance(umbral_corr, (float, int)):
        raise TypeError("'umbral_corr' debe ser un float o int.")
    
    if pvalue is not None and not isinstance(pvalue, (float, int)):
        raise TypeError("'pvalue' debe ser un float o int.")

  
    # Si la lista no esta vacia   
    if columns:
        features = columns
    # Si la lista esta vacia
    else:
        features = get_features_num_regression(df, target_col, umbral_corr, pvalue)
        columns = features
        
    sns.pairplot(df[features + [target_col]], hue = target_col);

    return columns
        

def get_features_cat_regression(df, target_col, pvalue = 0.05):
    """
    Esta función retorna una lista de features categóricas que superan el test de hipótesis usado.

    Argumentos:
    df (DataFrame)      : DataFrame original
    target_col (Series) : nombre de una de las columnas numéricas (variable numérica continua o discreta)
    pvalue (float )     : pvalue de un two sample T-test. Default = 0.05

    Retorna:
    List: Lista con columnas categóricas del DataFrame 
    """

    # Check
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' no es columa del DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise TypeError(f"'{target_col}' debe ser una variable numérica continua o discreta.")
    
    if not isinstance(pvalue, (float, int)):
        raise TypeError("'pvalue' debe ser un float o int.")


    # Columnas categóricas
    categorical = df.select_dtypes(exclude=[np.number]).columns
    features = []
    for cat in categorical:
        temp = df[[cat, target_col]].dropna()
        #Separar en grupos por cat
        groups = [g[target_col].values for _, g in temp.groupby(cat, observed=True)]
        
        normal = all(shapiro(g)[1] > 0.05 for g in groups if len(g) > 2)
        equal_var = levene(*groups)[1] > 0.05 if len(groups) > 1 else False
        
        if normal and equal_var:
            stat, p = f_oneway(*groups) #One way ANOVA
        else:
            stat, p = kruskal(*groups) #Kruskal-Wallis H-test
        
        if p < pvalue:
            features.append(cat)

    return features
    
    

def plot_features_cat_regression(df, target_col = "", columns = None, pvalue = 0.05, with_individual_plot = False):
    """
    Esta función grafica histogramas de las features categóricas o features numéricas que superan el test de hipótesis usado, respectivamente.

    Argumentos:
    df (DataFrame)      : DataFrame original
    target_col (Series) : nombre de una de las columnas. Default = ""
    columns (list)      : Lista de strings. Default  = [].
    pvalue (float )     : pvalue de un test Pearson. Default = None
    with_individual_plot (boolean) : opción grafica individual. Default = False.

    Retorna:
    List: Lista con columnas usadas en el grafico 
    """
    
    # Check
    # Check
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' no es columa del DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise TypeError(f"'{target_col}' debe ser una variable numérica continua o discreta.")

    if not isinstance(pvalue, (float, int)):
        raise TypeError("'pvalue' debe ser un float o int.")

    if not isinstance(with_individual_plot, (bool)):
        raise TypeError("'with_individual_plot' debe ser True or False.")
    

    # Si la lista no esta vacia   
    if columns:
        col_cat = get_features_cat_regression(df[columns + [target_col]], target_col, pvalue)
        columns = col_cat
    # Si la lista esta vacia
    else:
        columns = get_features_num_regression(df, target_col, 0.0, pvalue) 
        col_cat = []


    #Histogramas
    
    if with_individual_plot:
        
        for col in columns:
            plt.figure(figsize=(6, 4))
            if col in col_cat:
                sns.histplot(data=df, x=target_col, hue=col, kde=False, multiple="dodge")
                plt.title(f"Histograma de {target_col} por {col}")
            else:
                binned = pd.cut(df[col], bins=5)
                sns.histplot(data=df.assign(binned=binned), x=target_col, hue="binned", kde=False, multiple="dodge")
                plt.title(f"Histograma de {target_col} por {col}")
        plt.tight_layout()
        plt.show()
        
    else:
        n_cols = 3
        n_rows = math.ceil(len(columns) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
        axes = axes.flatten() 

        for i, col in enumerate(columns):
            ax = axes[i]
            if col in col_cat:
                sns.histplot(data=df, x=target_col, hue=col, kde=False,
                         multiple="dodge", ax=ax)
                ax.set_title(f"Histograma de {target_col} por {col}")
            else:
                binned = pd.cut(df[col], bins=5)
                sns.histplot(data=df.assign(binned=binned), x=target_col, hue="binned",
                         kde=False, multiple="dodge", ax=ax)
                ax.set_title(f"Histograma de {target_col} por {col})")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        

    return columns
    
    