import pandas as pd
import numpy as np
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class model_v2:
    def __init__(
        self,
        train_data,             # Train set
        valid_data,             # Validation set
        test_data,              # Test set
        criteria,               # Classification threshold
        split_ratio,            # split ratio (format: [train:validation:test])
        autoencoder_epochs,     # epochs of autoencoder
        classifier_epochs,      # epochs of classifier
        weight_for_attack,      # weight for attack
    ):
        # self.dfs = train_data
        self.train_set = train_data
        self.valid_set = valid_data
        self.test_set = test_data
        self.criteria = criteria
        self.split_ratio = split_ratio

        # 1. Encoding Categorical Features and Set Column Name List
        for target_col in ['region', 'city', 'name', 'os_name', 'device_type', 'browser_name', 'country_code']:
            prob_df = self.train_set.groupby(target_col)['label'].mean().reset_index(name=target_col + '_prob')
            self.train_set = self.train_set.merge(prob_df, on=target_col)
        
        for target_col in ['region', 'city', 'name', 'os_name', 'device_type', 'browser_name', 'country_code']:
            prob_df = self.valid_set.groupby(target_col)['label'].mean().reset_index(name=target_col + '_prob')
            self.valid_set = self.valid_set.merge(prob_df, on=target_col)

        for target_col in ['region', 'city', 'name', 'os_name', 'device_type', 'browser_name', 'country_code']:
            prob_df = self.test_set.groupby(target_col)['label'].mean().reset_index(name=target_col + '_prob')
            self.test_set = self.test_set.merge(prob_df, on=target_col)

        cat_col = [
            'country_code',  # 'region_risk_grade', 'city_risk_grade', 'name_risk_grade',
            'login_success', 'os_is_legacy', 'browser_is_legacy', # 'device_type',
        ]
        num_col = ['rtt', 'region_prob', 'city_prob', 'name_prob', 'os_name_prob', 'device_type_prob',
                   'browser_name_prob', ]
        target = 'label'

        label_encoders = {}
        for col in cat_col:
            le = LabelEncoder()
            self.train_set[col] = le.fit_transform(self.train_set[col])
            self.valid_set[col] = le.fit_transform(self.valid_set[col])
            self.test_set[col] = le.fit_transform(self.test_set[col])
            label_encoders[col] = le  # 나중에 역변환을 위해 저장

        # 2. Split Dataset
        # train_iter, val_iter, test_iter = self.stratified_split(
        #     target, self.split_ratio, cat_col, num_col
        # )
        train_iter = self.from_df_to_tensor(self.train_set, cat_col, num_col, target)
        val_iter = self.from_df_to_tensor(self.valid_set, cat_col, num_col, target)
        test_iter = self.from_df_to_tensor(self.test_set, cat_col, num_col, target)

        # 3. Build Model
        # 3.1 Pretrain with AutoEncoder
        cat_cols_info = [self.train_set[col].nunique() for col in cat_col]
        autoencoder = CategoricalAutoencoder(cat_cols_info, len(num_col), hidden_dim=50).to(device)
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.train_autoencoder(autoencoder, train_iter, criterion, optimizer, epochs=autoencoder_epochs)

        # 잠재 벡터 추출
        train_latent_vectors, train_labels = self.extract_latent_vectors(autoencoder, train_iter)
        val_latent_vectors, val_labels = self.extract_latent_vectors(autoencoder, val_iter)
        test_latent_vectors, test_labels = self.extract_latent_vectors(autoencoder, test_iter)

        # 잠재 벡터를 사용한 분류 모델 훈련을 위한 DataLoader 생성
        train_latent_dataset = TensorDataset(train_latent_vectors, train_labels)
        train_latent_data_loader = DataLoader(train_latent_dataset, batch_size=32, shuffle=True)

        # 검증 데이터셋을 위한 DataLoader 생성
        val_latent_dataset = TensorDataset(val_latent_vectors, val_labels)
        val_latent_data_loader = DataLoader(val_latent_dataset, batch_size=32, shuffle=True)

        # 3.2 Train with NN
        # 분류 모델 초기화
        classifier = Classifier(input_dim=train_latent_vectors.size(1), hidden_dim=100, output_dim=1).to(device)
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        # 가중치 설정: 레이블이 1인 샘플에 대해 더 높은 가중치 부여
        weights = torch.tensor([weight_for_attack])  # 레이블이 1인 샘플에 대한 가중치

        # 가중치를 BCELoss에 적용
        classifier_criterion = nn.BCELoss(weight=weights)

        # Early Stopping 인스턴스 생성
        early_stopping = EarlyStopping(patience=10, delta=0.001)

        # 훈련 함수 호출
        self.train_classifier(
            classifier,
            train_latent_data_loader,
            val_latent_data_loader,
            classifier_criterion,
            classifier_optimizer,
            classifier_epochs,
            early_stopping,
        )

        # 테스트 데이터셋을 위한 DataLoader 생성
        test_latent_dataset = TensorDataset(test_latent_vectors, test_labels)
        test_latent_data_loader = DataLoader(test_latent_dataset, batch_size=32, shuffle=True)

        # 테스트 함수 호출
        self.predicted_df = self.test_classifier(classifier, test_latent_data_loader, classifier_criterion)

    def from_df_to_tensor(self, df, cat_col, num_col, target):
        # 데이터를 NumPy 배열로 변환
        cat_data_np = df[cat_col].to_numpy(dtype=np.int64)  # dtype를 int64로 명시
        num_data_np = df[num_col].to_numpy(dtype=np.float32)
        labels_np = df[target].to_numpy(dtype=np.float32)

        # 데이터를 텐서로 변환
        cat_data_tensor = torch.tensor(cat_data_np, dtype=torch.long).to(device)
        num_data_tensor = torch.tensor(num_data_np, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels_np, dtype=torch.float32).to(device)

        # DataLoader 생성
        batch_size = 32
        dataset = TensorDataset(cat_data_tensor, num_data_tensor, labels_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader

    # def stratified_split(self, target, ratios, cat_col, num_col, random_state=42):
    #     """
    #     Splits a DataFrame into train, validation, and test sets according to the specified ratios.

    #     :param df: DataFrame to split.
    #     :param target: The target column for stratification.
    #     :param ratios: List or tuple with 3 numbers representing the train, validation, and test ratios.
    #     :param random_state: Random state for reproducibility.
    #     :return: Three DataFrames corresponding to the train, validation, and test sets.
    #     """

    #     def from_df_to_tensor(df, cat_col, num_col, target):
    #         # 데이터를 NumPy 배열로 변환
    #         cat_data_np = df[cat_col].to_numpy(dtype=np.int64)  # dtype를 int64로 명시
    #         num_data_np = df[num_col].to_numpy(dtype=np.float32)
    #         labels_np = df[target].to_numpy(dtype=np.float32)

    #         # 데이터를 텐서로 변환
    #         cat_data_tensor = torch.tensor(cat_data_np, dtype=torch.long)
    #         num_data_tensor = torch.tensor(num_data_np, dtype=torch.float32)
    #         labels_tensor = torch.tensor(labels_np, dtype=torch.float32)

    #         # DataLoader 생성
    #         batch_size = 32
    #         dataset = TensorDataset(cat_data_tensor, num_data_tensor, labels_tensor)
    #         data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #         return data_loader

    #     train_ratio, val_ratio, test_ratio = ratios
    #     total = sum(ratios)
    #     test_size = test_ratio / total
    #     val_size = val_ratio / (total - test_ratio)  # Adjusted for the remaining after test split

    #     # First split to separate out the test set
    #     df_train_val, df_test = train_test_split(
    #         self.dfs,
    #         test_size=test_size,
    #         stratify=self.dfs[target],
    #         random_state=random_state
    #     )

    #     # Second split to separate out the validation set
    #     df_train, df_val = train_test_split(
    #         df_train_val,
    #         test_size=val_size,
    #         stratify=df_train_val[target],
    #         random_state=random_state
    #     )
    #     train_iter = from_df_to_tensor(df_train, cat_col, num_col, target)
    #     val_iter = from_df_to_tensor(df_val, cat_col, num_col, target)
    #     test_iter = from_df_to_tensor(df_test, cat_col, num_col, target)

    #     return train_iter, val_iter, test_iter

    def train_autoencoder(self, model, data_loader, criterion, optimizer, epochs):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for cat_data, num_data, _ in data_loader:
                optimizer.zero_grad()
                reconstructed = model(cat_data, num_data)
                # 임베딩된 원본 범주형 데이터
                original_cat = torch.cat([embedding(cat_data[:, i]) for i, embedding in enumerate(model.embeddings)], 1)
                # 임베딩된 원본 데이터
                original_data = torch.cat([original_cat, num_data], 1)
                # Autoencoder의 출력과 임베딩된 원본 데이터를 비교
                loss = criterion(reconstructed, original_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}')

    def extract_latent_vectors(self, model, data_loader):
        model.eval()
        latent_vectors = []
        labels = []
        with torch.no_grad():
            for cat_data, num_data, label in data_loader:
                # 임베딩된 범주형 데이터
                embedded_cat = torch.cat([model.embeddings[i](cat_data[:, i]) for i in range(len(model.embeddings))], 1)
                # 임베딩된 범주형 데이터와 수치형 데이터를 결합
                combined_data = torch.cat([embedded_cat, num_data], 1)
                encoded = model.encoder(combined_data)
                latent_vectors.append(encoded)
                labels.append(label)
        return torch.cat(latent_vectors), torch.cat(labels)

    def train_classifier(self, model, train_loader, val_loader, criterion, optimizer, epochs, early_stopping):
        for epoch in range(epochs):
            # 훈련 과정
            model.train()
            total_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.view(-1, 1).to(device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # scheduler.step()
            # 검증 과정
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    labels = labels.view(-1, 1).to(device)
                    val_loss += criterion(outputs, labels).item()
            val_loss /= len(val_loader)

            print(
                f'Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}')

            # Early Stopping 확인
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def test_classifier(self, model, test_loader, criterion, criteria=0.5):
        model.eval()
        test_loss = 0
        correct = 0
        all_predicted = []
        all_labels = []
        results = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                probabilities = outputs  # Assuming binary classification
                labels = labels.view(-1, 1).to(device)
                test_loss += criterion(outputs, labels).item()

                # Use the criteria for determining the predicted labels
                predicted = (probabilities >= criteria).float()

                correct += (predicted == labels).sum().item()
                results.extend(zip(probabilities.cpu().numpy(), predicted.cpu().numpy(), labels.cpu().numpy()))
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader)
        accuracy = correct / len(test_loader.dataset)
        f1 = f1_score(all_labels, all_predicted, average='binary')
        print(f'Test Loss: {test_loss}, Test Accuracy: {accuracy}, F1 Score: {f1}')

        results_df = pd.DataFrame(results, columns=['Probability', 'Predicted Label', 'Actual Label'])
        results_df['Probability'] = results_df['Probability'].apply(lambda x: x[0])
        results_df['Predicted Label'] = results_df['Predicted Label'].apply(lambda x: x[0])
        results_df['Actual Label'] = results_df['Actual Label'].apply(lambda x: x[0])

        return results_df


class CategoricalAutoencoder(nn.Module):
    def __init__(self, cat_cols_info, num_numerical_cols, hidden_dim):
        super(CategoricalAutoencoder, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, min(50, (cardinality + 1) // 2))
            for cardinality in cat_cols_info
        ])
        total_embedding_dim = sum([min(50, (n + 1) // 2) for n in cat_cols_info])
        self.encoder = nn.Sequential(
            nn.Linear(total_embedding_dim + num_numerical_cols, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_embedding_dim + num_numerical_cols)
        )

    def forward(self, x_cat, x_num):
        x_cat = [embedding(x_cat[:, i]) for i, embedding in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, 1)
        x = torch.cat([x_cat, x_num], 1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# load dataset
# cur_dir = os.getcwd()
# data_dir = os.path.join(cur_dir, 'data')
# df_preprocessed = pd.read_csv(os.path.join(data_dir, 'preprocessed.csv'))

# Run Model
# result = model_v2(
#     data=df_preprocessed,       # dataset
#     criteria=0.5,   # Classification threshold
#     split_ratio=[7, 1, 2], # split ratio (format: [train,validation,test])
#     autoencoder_epochs=50,     # epochs of autoencoder
#     classifier_epochs=200,      # epochs of classifier
#     weight_for_attack=15,      # weight for attack
# )

# predicted_df = result.predicted_df

