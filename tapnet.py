# %%
import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm, trange
import glob
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import pytorch_tabnet
import torch.nn.functional as F

# %%
# warnings.filterwarnings('ignore')
# tqdm.pandas()

# %%
# data = pd.read_csv('/kaggle/input/preprocessed/preprocessed.csv')

# %%
class TabNetModel:
    def __init__(self, train_set, valid_set, test_set, selected_columns, categorical_columns, target, pre_train_epochs = 100, epochs = 50):
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.selected_columns = selected_columns
        self.categorical_columns = categorical_columns
        self.target = target
        self.pre_train_epochs = pre_train_epochs
        self.epochs = epochs
        self.prepare_data()
        self.pretrain_model()
        self.train_model()

    def prepare_data(self):
        train_df = self.train_set[self.selected_columns]
        valid_df = self.valid_set[self.selected_columns]
        test_df = self.test_set[self.selected_columns]

        # 범주형 변수 인코딩
        for col in self.categorical_columns:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col])
            valid_df[col] = le.fit_transform(valid_df[col])
            test_df[col] = le.fit_transform(test_df[col])

        # X = df.drop(self.target, axis=1)
        # y = df[self.target]

        # 데이터 분할
        # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        # self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        # self.X_train, self.y_train = X_train, y_train

        self.X_train = train_df.drop(['label'], axis = 1)
        self.y_train = train_df['label']

        self.X_valid = valid_df.drop(['label'], axis = 1)
        self.y_valid = valid_df['label']

        self.X_test = test_df.drop(['label'], axis = 1)
        self.y_test = test_df['label']

        # 범주형 컬럼 인덱스
        self.cat_idxs = [i for i, col in enumerate(self.X_train.columns) if col in self.categorical_columns]
    
    # Pretrain 
    def pretrain_model(self):
        self.pretrainer = TabNetPretrainer(optimizer_fn=torch.optim.Adam,
                                           optimizer_params=dict(lr=2e-2),
                                           mask_type='entmax')

        self.pretrainer.fit(X_train=self.X_train.values,
                            eval_set=[self.X_test.values],
                            max_epochs=self.pre_train_epochs,
                            patience=10,
                            batch_size=512,
                            virtual_batch_size=128)
    
    # Custom Loss Function, weight false negative 조정
    def weighted_cross_entropy_with_logits(self, y_pred, y_true, weight_false_negative=2.0):
        loss = F.cross_entropy(y_pred, y_true, reduction='none')
        predicted_classes = torch.argmax(torch.nn.Softmax(dim=-1)(y_pred), dim=1)
        false_negatives = (y_true == 1) & (predicted_classes == 0)
        weighted_loss = torch.where(false_negatives, loss * weight_false_negative, loss)
        return weighted_loss.mean()

    def train_model(self):
        self.clf = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                                    optimizer_params=dict(lr=2e-2),
                                    scheduler_params={"step_size":50, "gamma":0.9},
                                    scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                    mask_type='entmax')

        self.clf.fit(X_train=self.X_train.values, y_train=self.y_train.values,
                     eval_set=[(self.X_valid.values, self.y_valid.values)],
                     eval_name=['val'],
                     eval_metric=['balanced_accuracy'],
                     max_epochs=self.epochs,
                     patience=10,
                     batch_size=512,
                     virtual_batch_size=128,
                     num_workers=1,
                     drop_last=False,
                     from_unsupervised=self.pretrainer,
                     loss_fn=self.weighted_cross_entropy_with_logits)

    def evaluate(self):
        preds = self.clf.predict(self.X_test.values)
        conf_matrix = confusion_matrix(self.y_test, preds)
        accuracy = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds)

        # Confusion Matrix 시각화
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

        print("Accuracy:", accuracy)
        print("F1 Score:", f1)


# # %%
# # 선택할 컬럼들
# selected_columns = ['country_code', 'region', 'city_risk_grade', 'name_risk_grade',
#                     'login_success', 'browser_is_legacy', 'os_is_legacy', 'rtt', 'device_type', 'label']

# # 범주형 컬럼들
# categorical_columns = ['country_code', 'device_type', 'region']

# # TabNetModel 클래스 인스턴스화
# tabnet_model = TabNetModel(data, selected_columns, categorical_columns, 'label')

# # 모델 평가
# tabnet_model.evaluate()



