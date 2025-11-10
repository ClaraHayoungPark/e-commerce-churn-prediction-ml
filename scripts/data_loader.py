import pandas as pd
import os

# BASE_PATH를 상대 경로로 설정 (Jupyter Notebook에서 호출 시 경로 맞추기 용이)
BASE_PATH = '../data/raw'

CORE_FILES = {
    'customers': 'olist_customers_dataset.csv',
    'orders': 'olist_orders_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'payments': 'olist_order_payments_dataset.csv',
    'reviews': 'olist_order_reviews_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'translation': 'product_category_name_translation.csv'
}


def load_all_data(base_path: str = BASE_PATH) -> dict:
    """
    Olist의 주요 CSV 파일 7개를 로드해 DataFrame 딕셔너리로 반환.
    """
    print(f"--- 데이터 로딩 시작: {base_path} ---")
    dataframes = {}
    
    for name, filename in CORE_FILES.items():
        file_path = os.path.join(base_path, filename)
        try:
            dataframes[name] = pd.read_csv(file_path)
            print(f"  {name:<11}: {dataframes[name].shape[0]:,} rows")
        except FileNotFoundError:
            print(f"  [경고] 파일 {filename}을 찾을 수 없습니다. 경로를 확인하세요.")
            dataframes[name] = pd.DataFrame()  # 빈 DataFrame으로 대체
            
    print("--- 데이터 로딩 완료 ---\n")
    return dataframes


def clean_initial_data(df_dict: dict) -> dict:
    """
    날짜/시간 컬럼을 datetime 타입으로 변환하는 초기 정리 작업을 수행.
    """
    print("--- 초기 데이터 클리닝 및 타입 변환 ---")
    
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    
    if 'orders' in df_dict and not df_dict['orders'].empty:
        df_orders = df_dict['orders']
        for col in date_cols:
            if col in df_orders.columns:
                df_orders[col] = pd.to_datetime(df_orders[col], errors='coerce')
        print("  orders: 날짜 컬럼 변환 완료")
    else:
        print("  [주의] orders 데이터가 없어 변환을 건너뜁니다.")
        
    print("--- 초기 클리닝 완료 ---\n")
    return df_dict


def merge_core_tables(df_dict: dict) -> pd.DataFrame:
    """
    고객 이탈 예측에 필요한 핵심 테이블(고객, 주문, 결제, 리뷰)을 병합.
    """
    print("--- 핵심 테이블 병합 시작 ---")
    
    if any(df_dict.get(k) is None or df_dict.get(k).empty for k in ['orders', 'customers', 'payments', 'reviews']):
        print("  [오류] 필수 데이터(orders, customers, payments, reviews) 중 일부가 없습니다.")
        return pd.DataFrame()
        
    # 1. 주문 + 고객 (customer_id 기준)
    df_merged = pd.merge(
        df_dict['orders'],
        df_dict['customers'][['customer_id', 'customer_unique_id', 'customer_state']],
        on='customer_id',
        how='left'
    )
    print(f"  1단계 병합(orders + customers) 완료 → {df_merged.shape}")
    
    # 2. 결제 정보 추가 (order_id 기준)
    df_merged = pd.merge(
        df_merged,
        df_dict['payments'][['order_id', 'payment_value', 'payment_type']],
        on='order_id',
        how='left'
    )
    print(f"  2단계 병합(+ payments) 완료 → {df_merged.shape}")
    
    # 3. 리뷰 점수 추가 (order_id 기준)
    df_merged = pd.merge(
        df_merged,
        df_dict['reviews'][['order_id', 'review_score']],
        on='order_id',
        how='left'
    )
    print(f"  3단계 병합(+ reviews) 완료 → {df_merged.shape}")
    
    print(f"--- 최종 통합 데이터셋 크기: {df_merged.shape} ---\n")
    return df_merged


# scripts/data_loader.py
if __name__ == '__main__':
    # 스크립트를 단독 실행할 때 테스트용 코드
    # 일반적으로는 Jupyter Notebook에서 함수 호출로 사용
    # data_dict = load_all_data()
    # data_dict = clean_initial_data(data_dict)
    # df_final = merge_core_tables(data_dict)
    # print(df_final.head())
    pass