import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import pickle


df_dt2 = pd.read_csv(r'C:\Users\oberd\OneDrive\Documentos\..BANCO EDA\diab_2001.csv',
                     encoding='ISO-8859-1', engine='python')

# PREPROCESSAMENTO
# substituir os valores 0 de['glicose','pressao_sanguinea','espessura_triceps','insulina', 'imc'] por NaN
df_dt2 = df_dt2.copy(deep=True)
df_dt2[['glicose', 'pressao_sanguinea', 'espessura_triceps', 'insulina', 'imc']] = df_dt2[
        ['glicose', 'pressao_sanguinea', 'espessura_triceps', 'insulina', 'imc']].replace(0, np.NaN)

# Substitui valores ausentes por mean,medain
df_dt2['glicose'].fillna(df_dt2['glicose'].mean(), inplace=True)
df_dt2['pressao_sanguinea'].fillna(df_dt2['pressao_sanguinea'].mean(), inplace=True)
df_dt2['espessura_triceps'].fillna(df_dt2['espessura_triceps'].median(), inplace=True)
df_dt2['insulina'].fillna(df_dt2['insulina'].mean(), inplace=True)
df_dt2['imc'].fillna(df_dt2['imc'].median(), inplace=True)

# CONSTRUÇÃO DO MODELO PARA O FEMININO

std_list = ['n_gravidez', 'glicose', 'pressao_sanguinea', 'espessura_triceps', 'insulina', 'imc',
                     'hist_familiar_D', 'idade']

def standartization(x):
    x_std = x.copy(deep=True)
    for column in std_list:
        x_std[column] = (x_std[column] - x_std[column].mean()) / x_std[column].std()
    return x_std

df_dt2 = standartization(df_dt2)

# Separar  variaveis independentes/entradas da dependente/saida
X = df_dt2.iloc[:, :-1]
y = df_dt2.iloc[:, -1]
# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Construção do modelo (Logistic Regression)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# Avaliando o modelo
log_reg = log_reg.score(X_test, y_test)
# Construção do modelo (Random forest classifier)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# Avaliando o modelo
clf = clf.score(X_test, y_test)
# Construção do modelo (XGBClassifier)
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gbc.fit(X_train, y_train)
# Avaliando o modelo
gbc = gbc.score(X_test, y_test)
# Construção do modelo (LGBM Classifier)
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
# Avaliando o modelo
lgbm = lgbm.score(X_test, y_test)
model_compare = pd.DataFrame({"Logistic Regression": log_reg,
                              "Random Forest Classifier": clf,
                              "GradientBoostingClassifier": gbc,
                              "LGBM Classifier": lgbm}, index=["accuracy"])
print(model_compare)

log_reg1 = log_reg
clf1 = clf
gbc1 = gbc
lgbm1 = lgbm

# Verificando quem é maior resulyado
maior = log_reg1
if clf1 > log_reg1 and clf1 > gbc1 or clf1 > lgbm1:
    maior = clf1
if gbc1 > log_reg1 and gbc1 > clf1 or gbc1 > lgbm1:
    maior = gbc1
if lgbm1 > log_reg1 and lgbm1 > clf1 or lgbm1 > gbc1:
    maior = lgbm1

print('O  melhor resultado do classificador é {}  '.format(maior))

if maior == log_reg1:
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    pickle.dump(log_reg, open('pre_log_reg.pkl', 'wb'))

if maior == clf1:
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pickle.dump(clf, open('pred_clf.pkl', 'wb'))

if maior == gbc1:
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    gbc.fit(X_train, y_train)
    pickle.dump(gbc, open('pred_gbc.pkl', 'wb'))

else:
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)
    # pickle.dump(lgbm, open('pred_lgbm.pkl', 'wb'))
    pickle_out = open("pre_lgbm.pkl", mode="wb")
    pickle.dump(lgbm, pickle_out)
    pickle_out.close()


