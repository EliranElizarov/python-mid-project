import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

class AmazonETL:
    def __init__(self, input_file):
        self.input_file = input_file
        self.dwh_path = "data/dwh"
        self.logs_path = "data/logs"
        self.logs = []
        self.start_time = None

    def log_step(self, step_name, status, count=0, error=None):
        self.logs.append({
            "step": step_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "row_count": count,
            "error": str(error) if error else "None"
        })

    def extract(self):
        self.start_time = time.perf_counter()
        try:
            df = pd.read_csv(self.input_file)
            self.log_step("Extract MRR", "Success", len(df))
            return df
        except Exception as e:
            self.log_step("Extract MRR", "Failed", error=e)
            raise

    def transform(self, df):
        try:
            def to_numeric(series, is_percent=False):
                clean = series.astype(str).str.replace('₹', '').str.replace(',', '').str.replace('%', '')
                num = pd.to_numeric(clean, errors='coerce')
                return num / 100 if is_percent else num

            def split_and_align(row, cols):
                splits = [str(row[c]).split(',') if pd.notnull(row[c]) else [] for c in cols]
                max_len = max(len(s) for s in splits)
                for s in splits: s.extend([None] * (max_len - len(s)))
                return list(zip(*splits))

            df['discounted_price'] = to_numeric(df['discounted_price'])
            df['actual_price'] = to_numeric(df['actual_price'])
            df['discount_percentage'] = to_numeric(df['discount_percentage'], is_percent=True)
            df['rating_count'] = to_numeric(df['rating_count']).fillna(0).astype(int)
            df['discount_price'] = df['actual_price'] - df['discounted_price']

            dim_product = df.sort_values('rating_count', ascending=False).drop_duplicates('product_id').copy()
            dim_product['category'] = dim_product['category'].str.split('|').str[0]
            dim_product = dim_product[['product_id', 'product_name', 'category', 'rating', 'rating_count']]
            self.log_step("Transform: dim_product", "Success", len(dim_product))

            user_cols = ['user_id', 'user_name']
            df_user_temp = df.copy()
            df_user_temp['zipped'] = df_user_temp.apply(lambda r: split_and_align(r, user_cols), axis=1)
            dim_user = df_user_temp.explode('zipped')
            dim_user[user_cols] = pd.DataFrame(dim_user['zipped'].tolist(), index=dim_user.index)
            dim_user['user_id'] = dim_user['user_id'].str.strip()
            dim_user = dim_user[user_cols].drop_duplicates('user_id').dropna(subset=['user_id'])
            self.log_step("Transform: dim_user", "Success", len(dim_user))

            fact_sales = df.rename(columns={'actual_price': 'full_price', 'discounted_price': 'price_after_discount'}).copy()
            fact_sales['user_id'] = fact_sales['user_id'].str.split(',')
            fact_sales = fact_sales.explode('user_id')
            fact_sales['user_id'] = fact_sales['user_id'].str.strip()
            sales_cols = ['product_id', 'user_id', 'full_price', 'discount_percentage', 'discount_price', 'price_after_discount']
            fact_sales = fact_sales[sales_cols].dropna(subset=['user_id'])
            self.log_step("Transform: fact_sales", "Success", len(fact_sales))

            review_cols = ['user_id', 'review_id', 'review_title', 'review_content']
            df_rev_temp = df.copy()
            df_rev_temp['zipped'] = df_rev_temp.apply(lambda r: split_and_align(r, review_cols), axis=1)
            fact_review = df_rev_temp.explode('zipped')
            fact_review[review_cols] = pd.DataFrame(fact_review['zipped'].tolist(), index=fact_review.index)
            fact_review['user_id'] = fact_review['user_id'].str.strip()
            fact_review = fact_review[['product_id'] + review_cols].dropna(subset=['review_id'])
            self.log_step("Transform: fact_review", "Success", len(fact_review))

            return {"dim_product": dim_product, "dim_user": dim_user, "fact_sales": fact_sales, "fact_review": fact_review}
        except Exception as e:
            self.log_step("Transformation", "Failed", error=e)
            raise

    def quality_check(self, tables):
        print("\n--- RUNNING DATA QUALITY CHECKS ---")
        try:
            orphans = tables['fact_sales'][~tables['fact_sales']['user_id'].isin(tables['dim_user']['user_id'])]
            missing_prods = tables['fact_sales'][~tables['fact_sales']['product_id'].isin(tables['dim_product']['product_id'])]
            null_prices = tables['fact_sales']['price_after_discount'].isnull().sum()
            
            print(f"Orphaned Users: {len(orphans)}")
            print(f"Missing Products in Fact: {len(missing_prods)}")
            print(f"Null Prices found: {null_prices}")
            
            status = "Success" if (len(orphans) + len(missing_prods) + null_prices) == 0 else "Warning"
            self.log_step("Quality Check", status)
        except Exception as e:
            self.log_step("Quality Check", "Failed", error=e)

    def load(self, tables_dict):
        if not os.path.exists(self.dwh_path): os.makedirs(self.dwh_path)
        for name, table_df in tables_dict.items():
            table_df.to_csv(os.path.join(self.dwh_path, f"{name}.csv"), index=False)
            self.log_step(f"Load {name}", "Success", len(table_df))

    def run_pipeline(self):
        print("Starting Amazon ETL Pipeline...")
        try:
            raw_df = self.extract()
            stg_tables = self.transform(raw_df)
            self.quality_check(stg_tables)
            self.load(stg_tables)
            print("\nPipeline execution finished successfully.")
            return stg_tables
        except Exception as e:
            print(f"Pipeline crashed: {e}")
        finally:
            if not os.path.exists(self.logs_path): os.makedirs(self.logs_path)
            log_df = pd.DataFrame(self.logs)
            log_df.to_csv(os.path.join(self.logs_path, "etl_log.csv"), index=False)
            print("\n--- FINAL LOG SUMMARY ---")
            print(log_df[['step', 'status', 'row_count']])

# --- EXECUTION (This runs inside the notebook AND is saved to the file) ---
etl = AmazonETL("data/mrr/amazon.csv")
tables = etl.run_pipeline()
