# 파일: scripts/feature_engineer.py

import pandas as pd
import numpy as np
from datetime import datetime

# -------------------------------------------------------------
# 1. 고객 특성 생성 함수
# -------------------------------------------------------------

def create_customer_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    병합된 거래 데이터(df_merged)를 기반으로 RFM 및 추가적인 품질/결제 관련 특성을 생성.
    """
    
    # 분석 기준일: 가장 최근 주문일의 다음 날
    snapshot_date = df_merged['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    # 1. RFM 지표 계산 (Recency, Frequency, Monetary)
    df_rfm = df_merged.groupby('customer_unique_id', as_index=False).agg(
        R_Recency=('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
        F_Frequency=('order_id', 'nunique'),
        M_Monetary=('payment_value', 'sum')
    )
    
    # -------------------------------------------------------------
    # 2. 품질 및 결제 관련 특성 생성
    # -------------------------------------------------------------
    
    # 2-1. 배송 관련 특성
    df_delivery = df_merged.copy()
    df_delivery = df_delivery.dropna(subset=['order_delivered_customer_date'])

    # 배송 지연 여부 (예상일보다 늦게 도착하면 1)
    df_delivery['Delay_Days'] = (
        df_delivery['order_delivered_customer_date'] - df_delivery['order_estimated_delivery_date']
    ).dt.days
    df_delivery['Is_Delayed'] = (df_delivery['Delay_Days'] > 0).astype(int)

    # 배송 소요 일수
    df_delivery['Delivery_Time_Days'] = (
        df_delivery['order_delivered_customer_date'] - df_delivery['order_purchase_timestamp']
    ).dt.days

    # 고객 단위 평균 지연률 및 평균 배송 일수
    df_delivery_agg = df_delivery.groupby('customer_unique_id', as_index=False).agg(
        Avg_Is_Delayed=('Is_Delayed', 'mean'),
        Avg_Delivery_Time=('Delivery_Time_Days', 'mean')
    )

    # 2-2. 리뷰 평균 점수 및 고객 지역 정보
    df_reviews = df_merged.groupby('customer_unique_id', as_index=False).agg(
        Avg_Review_Score=('review_score', 'mean')
    )
    df_state = df_merged[['customer_unique_id', 'customer_state']].drop_duplicates(subset=['customer_unique_id'])
    
    # 2-3. 결제 유형 특성 (One-Hot Encoding)
    df_payment = pd.get_dummies(df_merged, columns=['payment_type'], drop_first=True, prefix='payment')
    payment_cols = [col for col in df_payment.columns if col.startswith('payment_')]
    df_payment_agg = df_payment.groupby('customer_unique_id', as_index=False)[payment_cols].sum()
    
    # -------------------------------------------------------------
    # 3. 모든 특성 병합
    # -------------------------------------------------------------
    df_features = df_rfm.copy()
    df_features = pd.merge(df_features, df_delivery_agg, on='customer_unique_id', how='left')
    df_features = pd.merge(df_features, df_reviews, on='customer_unique_id', how='left')
    df_features = pd.merge(df_features, df_state, on='customer_unique_id', how='left')
    df_features = pd.merge(df_features, df_payment_agg, on='customer_unique_id', how='left')
    
    # 결측치 처리 (정보가 없는 경우 0으로 대체)
    df_features[['Avg_Is_Delayed', 'Avg_Delivery_Time', 'Avg_Review_Score']] = \
        df_features[['Avg_Is_Delayed', 'Avg_Delivery_Time', 'Avg_Review_Score']].fillna(0)
    
    payment_agg_cols = [col for col in df_features.columns if col.startswith('payment_')]
    df_features[payment_agg_cols] = df_features[payment_agg_cols].fillna(0)
    
    print(f"--- 고객 특성 생성 완료: {df_features.shape} ---")
    return df_features


# -------------------------------------------------------------
# 2. 이탈 타겟 변수 정의 함수
# -------------------------------------------------------------

def define_churn_target(df_features: pd.DataFrame, churn_period_days: int = 365) -> pd.DataFrame:
    """
    Recency 기준으로 이탈 고객 여부(Target_Churn)를 정의.
    """
    df_features['Target_Churn'] = (df_features['R_Recency'] > churn_period_days).astype(int)
    churn_ratio = df_features['Target_Churn'].mean() * 100

    print(f"--- 이탈 타겟 변수 정의 완료 (기준: {churn_period_days}일, 비율: {churn_ratio:.2f}%) ---")
    return df_features


# -------------------------------------------------------------
# 3. 단독 실행 방지
# -------------------------------------------------------------
if __name__ == '__main__':
    pass