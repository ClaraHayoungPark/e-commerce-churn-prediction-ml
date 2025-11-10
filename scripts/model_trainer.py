# 파일: scripts/model_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np


# -------------------------------------------------------------
# 1. 데이터 분할 함수
# -------------------------------------------------------------
def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.3, random_state: int = 42):
    """
    특성(X)과 타겟(y)을 분리하고, 학습/테스트 데이터셋으로 분할합니다.
    """

    # Target_Churn 및 'R_'로 시작하는 모든 특성은 제외
    feature_cols = [
        col for col in df.columns
        if col != target_col and not col.startswith('R_')
    ]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"데이터 분할 완료: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test


# -------------------------------------------------------------
# 2. 모델 학습 및 평가 함수
# -------------------------------------------------------------
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    XGBoost 모델을 학습하고 성능을 평가.
    """
    
    print("\n--- XGBoost 모델 학습 시작 ---")
    
    # 클래스 불균형 처리를 위한 가중치 계산 (neg_count / pos_count)
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight = neg_count / pos_count
    
    # XGBoost Classifier 설정
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- 모델 성능 평가 결과 ---")
    print(classification_report(y_test, y_pred, target_names=['Active (0)', 'Churned (1)']))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    return model, y_pred, roc_auc


# -------------------------------------------------------------
# 3. 단독 실행 방지
# -------------------------------------------------------------
if __name__ == '__main__':
    pass