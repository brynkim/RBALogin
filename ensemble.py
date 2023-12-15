# %% [markdown]
# PACKAGES

# %%
from deepod.models.tabular import DevNet, PReNet, DeepSAD, FeaWAD, RoSAS
from deepod.metrics import tabular_metrics
# from autoencodernn import *
# from tapnet import *

# %%
import pickle
import os
from datetime import datetime

# %%
import numpy as np
import pandas as pd

# %%
# from sklearn.model_selection import train_test_split

# %% [markdown]
# DATASET

# %%
# dfs = []
# for filename in tqdm(os.listdir('./data/')):
#     if 'preprocessed' in filename:
#         dfs.append(pd.read_csv(f'./data/{filename}', index_col = 0))
# df = pd.concat(dfs).reset_index(drop = True)
# del(dfs)

# %%
# train_valid_df, test_df = train_test_split(df, test_size = 0.2, stratify = df['label'])
# train_df, valid_df = train_test_split(train_valid_df, test_size = 0.125, stratify = train_valid_df['label'])

# %%
# X_train = train_df.drop(['label'], axis = 1)
# y_train = train_df['label']

# X_valid = valid_df.drop(['label'], axis = 1)
# y_valid = valid_df['label']

# X_test = test_df.drop(['label'], axis = 1)
# y_test = test_df['label']

# %%
# X_train.to_csv('./checkpoint/x_train.csv')
# y_train.to_csv('./checkpoint/y_train.csv')

# X_valid.to_csv('./checkpoint/x_valid.csv')
# y_valid.to_csv('./checkpoint/y_valid.csv')

# X_test.to_csv('./checkpoint/x_test.csv')
# y_test.to_csv('./checkpoint/y_test.csv')

# %%
# with open('./checkpoint/anomaly_models.pkl', 'rb') as f:
#     models = pickle.load(f)

# %%
train_df = pd.read_csv('./checkpoint/train_df.csv', index_col = 0)
valid_df = pd.read_csv('./checkpoint/valid_df.csv', index_col = 0)
test_df = pd.read_csv('./checkpoint/test_df.csv', index_col = 0)

# %%
X_train = pd.read_csv('./checkpoint/x_train.csv')
y_train = pd.read_csv('./checkpoint/y_train.csv')

X_valid = pd.read_csv('./checkpoint/x_valid.csv')
y_valid = pd.read_csv('./checkpoint/y_valid.csv')

X_test = pd.read_csv('./checkpoint/x_test.csv')
y_test = pd.read_csv('./checkpoint/y_test.csv')

# %% [markdown]
# MODEL

# %%
models = []

# %%
def anomaly_preprocessing(df1, df2, df3):
    df1['type'] = 'train'
    df2['type'] = 'valid'
    df3['type'] = 'test'

    df = pd.concat([df1, df2, df3]).reset_index(drop = True)

    country_onehot = pd.get_dummies(df['country']).astype(int)
    risk_grades = df[['region_risk_grade', 'city_risk_grade', 'name_risk_grade']]
    browser_onehot = pd.get_dummies(df['browser_name']).astype(int)
    os_onehot = pd.get_dummies(df['os_name']).astype(int)
    legacys = df[['browser_is_legacy', 'os_is_legacy']]
    device_types = pd.get_dummies(df['device_type']).astype(int)
    rtts = df['rtt']
    type = df['type']
    label = df['label']
    df = pd.concat([country_onehot, risk_grades, browser_onehot, os_onehot, legacys, device_types, rtts, type, label], axis = 1)

    df1 = df[df['type'] == 'train'].drop('type', axis = 1)
    df2 = df[df['type'] == 'valid'].drop('type', axis = 1)
    df3 = df[df['type'] == 'test'].drop('type', axis = 1)

    return df1, df2, df3

# %%
# Anomaly Detection
print('Anomaly Detection Model')
ad_model_names = ['DevNet', 'PReNet', 'DeepSAD', 'FeaWAD', 'RoSAS']
ad_models = [
    # DevNet(),           # 86.5s
    # PReNet(),           # Too long. (2.4 hours)
    DeepSAD(epochs = 50),          # 75.4s
    # FeaWAD(epochs = 10000, lr = 0.01),           # Very fast but poor.
    # RoSAS(),            # Too long.
]

train_df_ad, valid_df_ad, test_df_ad = anomaly_preprocessing(train_df, valid_df, test_df)

X_train_ad = train_df_ad.drop(['label'], axis = 1)
y_train_ad = train_df_ad['label']

X_valid_ad = valid_df_ad.drop(['label'], axis = 1)
y_valid_ad = valid_df_ad['label']

X_test_ad = test_df_ad.drop(['label'], axis = 1)
y_test_ad = test_df_ad['label']

for model_name, model in zip(ad_model_names, ad_models):
    print('start -', datetime.now())
    
    model.fit(X_train_ad.to_numpy(), y_train_ad.to_numpy())
    print('Train Finish')
    pred_train = (model.decision_function(X_train_ad.to_numpy()) > 0.5).astype(int)
    auc_train, ap_train, f1_train = tabular_metrics(y_train_ad, pred_train)
    
    pred_valid = (model.decision_function(X_valid_ad.to_numpy()) > 0.5).astype(int)
    auc_valid, ap_valid, f1_valid = tabular_metrics(y_valid_ad, pred_valid)
    
    print(f'Trained with {model}')
    print(f'Train - AUC: {auc_train}, AP: {ap_train}, F1: {f1_train}')
    print(f'Valid - AUC: {auc_valid}, AP: {ap_valid}, F1: {f1_valid}')

    models.append(model)
    print('end -', datetime.now(), '\n')

# %%
# # AutoEncoder + NN
# print('AutoEncoder + NN Model')
# autoencoder_nn_model = model_v2(
#     train_data = train_df,          # Train set
#     valid_data = valid_df,          # Validation set
#     test_data = test_df,            # Test set
#     criteria = 0.5,                 # Classification threshold
#     split_ratio = [7, 1, 2],        # split ratio (format: [train,validation,test])
#     autoencoder_epochs = 50,        # epochs of autoencoder
#     classifier_epochs = 200,        # epochs of classifier
#     weight_for_attack = 15,         # weight for attack
# )

# models.append(autoencoder_nn_model)

# %%
# # TabNet
# print('TabNet Model')
# selected_columns = ['country_code', 'region', 'city_risk_grade', 'name_risk_grade', 'login_success', 'browser_is_legacy', 'os_is_legacy', 'rtt', 'device_type', 'label']
# categorical_columns = ['country_code', 'device_type', 'region']

# tabnet_model = TabNetModel(train_df, valid_df, test_df, selected_columns, categorical_columns, 'label')         #, pre_train_epochs = 5, epochs = 5)
# models.append(tabnet_model)

# %%
with open('./checkpoint/deepsad.pkl', 'wb') as f:
    pickle.dump(models, f)

# %%



