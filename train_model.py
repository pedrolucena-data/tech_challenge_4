# ==========================================================
# IMPORTAÇÕES
# ==========================================================
import pandas as pd
import numpy as np
import pickle
import os

# Base para criação de transformadores customizados
from sklearn.base import BaseEstimator, TransformerMixin

# Pré-processamento
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

# Modelos
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Avaliação
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# ==========================================================
# CLASSES DE TRANSFORMAÇÃO
# ==========================================================

class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=None):
        if min_max_scaler is None:
            self.min_max_scaler = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        else:
            self.min_max_scaler = min_max_scaler
        self.scaler = MinMaxScaler()

    def fit(self, df, y=None):
        self.scaler.fit(df[self.min_max_scaler])
        return self

    def transform(self, df):
        df_copy = df.copy()
        df_copy[self.min_max_scaler] = self.scaler.transform(df_copy[self.min_max_scaler])
        return df_copy


class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_encoding=None):
        if one_hot_encoding is None:
            self.one_hot_encoding = ['CAEC', 'CALC', 'MTRANS']
        else:
            self.one_hot_encoding = one_hot_encoding
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, df, y=None):
        self.encoder.fit(df[self.one_hot_encoding])
        self.feature_names_ = self.encoder.get_feature_names_out(self.one_hot_encoding)
        return self

    def transform(self, df):
        df_enc = pd.DataFrame(
            self.encoder.transform(df[self.one_hot_encoding]),
            columns=self.feature_names_,
            index=df.index
        )
        outras_features = [col for col in df.columns if col not in self.one_hot_encoding]
        df_final = pd.concat([df[outras_features], df_enc], axis=1)

        for col in df_final.columns:
            if df_final[col].dtype == 'object':
                if set(df_final[col].unique()).issubset({'yes', 'no', 0, 1}):
                    df_final[col] = df_final[col].map({'no': 0, 'yes': 1})
                else:
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

        return df_final.astype(float)


class OrdinalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_features=None):
        if ordinal_features is None:
            self.ordinal_features = ['Obesity']
        else:
            self.ordinal_features = ordinal_features
        self.encoder = OrdinalEncoder()

    def fit(self, df, y=None):
        self.encoder.fit(df[self.ordinal_features])
        self.categories_ = self.encoder.categories_
        return self

    def transform(self, df):
        df_copy = df.copy()
        df_copy[self.ordinal_features] = self.encoder.transform(df_copy[self.ordinal_features])
        return df_copy


# ==========================================================
# FUNÇÃO PRINCIPAL DE TREINAMENTO
# ==========================================================

def train_and_save_model(
    data_path='data/Obesity.csv',
    save_path='model/obesity_model.pkl'
):
    # 1) LEITURA DOS DADOS
    df = pd.read_csv(data_path)
    df_proc = df.copy()

    # 2) AJUSTES INICIAIS
    df_proc['Gender'] = df_proc['Gender'].map({'Female': 0, 'Male': 1})
    for col in ['family_history', 'FAVC', 'SMOKE', 'SCC']:
        df_proc[col] = df_proc[col].map({'no': 0, 'yes': 1})

    cols_to_round = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    df_proc[cols_to_round] = df_proc[cols_to_round].round(0)

    # 3) PIPELINE DE TRANSFORMAÇÃO
    transform_pipeline = Pipeline([
        ('MinMax', MinMax()),
        ('OneHotEncodingNames', OneHotEncodingNames())
    ])

    # 4) PREPARAR TARGET
    target_encoder = OrdinalEncoder()
    y_encoded = target_encoder.fit_transform(df_proc[['Obesity']]).ravel()
    target_names = target_encoder.categories_[0].tolist()

    # 5) TRANSFORMAR FEATURES
    X_transformed = transform_pipeline.fit_transform(df_proc.drop('Obesity', axis=1))

    # 6) TREINO / TESTE
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 7) XGBOOST
    xgb = XGBClassifier(random_state=42, objective='multi:softprob', num_class=len(target_names))
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    xgb_cv = cross_val_score(xgb, X_transformed, y_encoded, cv=5).mean()

    # 8) RANDOM FOREST
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_cv = cross_val_score(rf, X_transformed, y_encoded, cv=5).mean()

    # 9) ESCOLHA DO MELHOR MODELO
    best_model = xgb if xgb_acc >= rf_acc else rf
    best_model_name = 'XGBoost' if xgb_acc >= rf_acc else 'Random Forest'

    # 10) PRINT DAS MÉTRICAS
    print(f"XGBoost       → Acurácia: {xgb_acc:.4f} | CV: {xgb_cv:.4f}")
    print(f"Random Forest → Acurácia: {rf_acc:.4f} | CV: {rf_cv:.4f}")
    print(f"\n✅ Melhor modelo: {best_model_name}")

    # 11) SALVAMENTO
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_data = {
        'model': best_model,
        'best_model_name': best_model_name,
        'xgb_model': xgb,
        'rf_model': rf,
        'transform_pipeline': transform_pipeline,
        'target_names': target_names,
        'features': X_transformed.columns.tolist(),
        'xgb_metrics': {'accuracy': float(xgb_acc), 'cv_mean': float(xgb_cv)},
        'rf_metrics': {'accuracy': float(rf_acc), 'cv_mean': float(rf_cv)},
        'xgb_importances': pd.Series(xgb.feature_importances_, index=X_transformed.columns)
            .sort_values(ascending=False).head(10).to_dict(),
        'rf_importances': pd.Series(rf.feature_importances_, index=X_transformed.columns)
            .sort_values(ascending=False).head(10).to_dict()
    }

    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModelo salvo em: {save_path}")
    print(f"Classes: {target_names}")
    print("Treinamento concluído com sucesso!")


if __name__ == "__main__":
    train_and_save_model()