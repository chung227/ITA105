import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_validate, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import joblib

# =========================================================
# 0. ĐỌC DỮ LIỆU
# =========================================================
file_path = '/content/ITA105_Lab_8(3).csv'   # đổi lại nếu tên file khác trên Colab

df = pd.read_csv(file_path, engine='python', on_bad_lines='skip', encoding_errors='ignore')
print('Kích thước dữ liệu:', df.shape)
print(df.head())
print(df.info())

# =========================================================
# 1. CHIA NHÓM CỘT
# =========================================================
target = 'SalePrice'

numeric_cols = ['LotArea', 'Rooms', 'HasGarage', 'NoiseFeature']
cat_cols = ['Neighborhood', 'Condition']
text_col = 'Description'
date_col = 'SaleDate'
drop_cols = ['ImagePath']

X = df.drop(columns=[target])
y = df[target]

print('\nCác nhóm cột:')
print('Numeric   :', numeric_cols)
print('Category  :', cat_cols)
print('Text      :', text_col)
print('Date      :', date_col)
print('Bỏ qua    :', drop_cols)

# =========================================================
# 2. TẠO CÁC TRANSFORMER TỰ VIẾT
# =========================================================
class SafeToNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.lower_ = {}
        self.upper_ = {}
        for c in X.columns:
            q1 = X[c].quantile(0.25)
            q3 = X[c].quantile(0.75)
            iqr = q3 - q1
            self.lower_[c] = q1 - 1.5 * iqr
            self.upper_[c] = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in X.columns:
            X[c] = X[c].clip(self.lower_[c], self.upper_[c])
        return X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(pd.DataFrame(X).iloc[:, 0])
        dt = pd.to_datetime(s, errors='coerce')
        out = pd.DataFrame({
            'sale_year': dt.dt.year,
            'sale_month': dt.dt.month,
            'sale_quarter': dt.dt.quarter,
            'sale_day': dt.dt.day,
            'sale_dayofweek': dt.dt.dayofweek
        })
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(['sale_year', 'sale_month', 'sale_quarter', 'sale_day', 'sale_dayofweek'])


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(pd.DataFrame(X).iloc[:, 0]).fillna('').astype(str).str.lower()
        s = s.str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
        s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
        return s


# =========================================================
# 3. XÂY DỰNG PIPELINE TỔNG QUÁT
# =========================================================
numeric_pipe = Pipeline(steps=[
    ('to_numeric', SafeToNumeric()),
    ('imputer', SimpleImputer(strategy='median')),
    ('clip_outlier', IQRClipper()),
    ('power', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_pipe = Pipeline(steps=[
    ('cleaner', TextCleaner()),
    ('tfidf', TfidfVectorizer(max_features=30, stop_words='english'))
])

date_pipe = Pipeline(steps=[
    ('date_features', DateFeatureExtractor()),
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipe, numeric_cols),
    ('cat', cat_pipe, cat_cols),
    ('txt', text_pipe, text_col),
    ('date', date_pipe, [date_col])
], remainder='drop')

# =========================================================
# 4. BÀI 1 - CHẠY SMOKE TEST VÀ IN SCHEMA
# =========================================================
print('\n' + '='*60)
print('BÀI 1 - SMOKE TEST')
print('='*60)

X_demo = X.head(10)
X_demo_transformed = preprocessor.fit_transform(X_demo)
print('Smoke test chạy thành công trên 10 dòng')
print('Shape sau transform:', X_demo_transformed.shape)

feature_names = preprocessor.get_feature_names_out()
print('\nSchema cuối cùng (20 tên đầu tiên):')
for name in feature_names[:20]:
    print(name)
print('...')
print('Tổng số feature:', len(feature_names))

# =========================================================
# 5. BÀI 2 - KIỂM THỬ PIPELINE VỚI 5 BỘ DỮ LIỆU
# =========================================================
print('\n' + '='*60)
print('BÀI 2 - KIỂM THỬ PIPELINE')
print('='*60)

# Tạo các bộ test
full_data = X.copy()

missing_data = X.copy()
missing_data.loc[0:80, 'LotArea'] = np.nan
missing_data.loc[30:120, 'Neighborhood'] = np.nan
missing_data.loc[10:90, 'Description'] = np.nan
missing_data.loc[50:140, 'SaleDate'] = np.nan

skewed_data = X.copy()
skewed_data['LotArea'] = skewed_data['LotArea'] * 20
skewed_data.loc[0:5, 'LotArea'] = skewed_data['LotArea'].max() * 3

unseen_data = X.copy()
unseen_data.loc[0:10, 'Neighborhood'] = 'Z'
unseen_data.loc[0:10, 'Condition'] = 'UnknownCondition'

wrong_type_data = X.copy()
wrong_type_data.loc[0:10, 'LotArea'] = 'abc'
wrong_type_data.loc[5:15, 'Rooms'] = '???'
wrong_type_data.loc[0:10, 'SaleDate'] = 'sai_ngay'


test_sets = {
    'du_lieu_day_du': full_data,
    'missing_nhieu': missing_data,
    'lech_phan_phoi': skewed_data,
    'unseen_category': unseen_data,
    'sai_dinh_dang': wrong_type_data
}

preprocessor.fit(X)

report_rows = []
for name, X_test in test_sets.items():
    try:
        Xt = preprocessor.transform(X_test)
        is_numeric = np.issubdtype(Xt.dtype, np.number)
        report_rows.append({
            'Bo_test': name,
            'Co_loi': 'Không',
            'Shape_output': Xt.shape,
            'Numeric_matrix': is_numeric,
            'Ghi_chu': 'Pipeline chạy được'
        })
    except Exception as e:
        report_rows.append({
            'Bo_test': name,
            'Co_loi': 'Có',
            'Shape_output': 'Lỗi',
            'Numeric_matrix': False,
            'Ghi_chu': str(e)
        })

report_df = pd.DataFrame(report_rows)
print(report_df)

# So sánh trước / sau pipeline với cột LotArea
before_num = X[['LotArea']].copy()
after_num = numeric_pipe.fit_transform(X[['LotArea']])
after_num = pd.DataFrame(after_num, columns=['LotArea_after'])

print('\nMô tả thống kê trước pipeline:')
print(before_num.describe())
print('\nMô tả thống kê sau pipeline:')
print(after_num.describe())

plt.figure(figsize=(8, 4))
plt.hist(before_num['LotArea'], bins=30)
plt.title('Histogram LotArea trước pipeline')
plt.xlabel('LotArea')
plt.ylabel('Tần số')
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(after_num['LotArea_after'], bins=30)
plt.title('Histogram LotArea sau pipeline')
plt.xlabel('LotArea đã chuẩn hóa')
plt.ylabel('Tần số')
plt.show()

print('\nBáo cáo lỗi và cách sửa:')
print('- Missing nhiều: xử lý bằng SimpleImputer.')
print('- Lệch phân phối: giảm lệch bằng PowerTransformer.')
print('- Unseen category: xử lý bằng OneHotEncoder(handle_unknown="ignore").')
print('- Sai định dạng số: ép kiểu bằng SafeToNumeric, lỗi sẽ chuyển thành NaN rồi impute.')
print('- Outlier: không xóa dòng mà clip theo IQR để shape dữ liệu ổn định.')

# =========================================================
# 6. BÀI 3 - TÍCH HỢP PIPELINE VÀO MÔ HÌNH
# =========================================================
print('\n' + '='*60)
print('BÀI 3 - PIPELINE + MODEL')
print('='*60)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42)
}

if HAS_XGB:
    models['XGBoost'] = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for model_name, model in models.items():
    full_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    scores = cross_validate(
        full_pipe,
        X,
        y,
        cv=cv,
        scoring={
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        },
        return_train_score=False
    )

    results.append({
        'Model': model_name,
        'RMSE_mean': -scores['test_rmse'].mean(),
        'MAE_mean': -scores['test_mae'].mean(),
        'R2_mean': scores['test_r2'].mean(),
        'RMSE_std': scores['test_rmse'].std()
    })

results_df = pd.DataFrame(results).sort_values('RMSE_mean')
print('\nKết quả cross-validation với pipeline:')
print(results_df)

# So sánh với cách xử lý thủ công đơn giản
X_manual = X.copy()
X_manual['SaleDate'] = pd.to_datetime(X_manual['SaleDate'], errors='coerce')
X_manual['sale_year'] = X_manual['SaleDate'].dt.year
X_manual['sale_month'] = X_manual['SaleDate'].dt.month
X_manual['sale_quarter'] = X_manual['SaleDate'].dt.quarter
X_manual = X_manual.drop(columns=['SaleDate', 'ImagePath'])

for c in ['LotArea', 'Rooms', 'HasGarage', 'NoiseFeature', 'sale_year', 'sale_month', 'sale_quarter']:
    X_manual[c] = pd.to_numeric(X_manual[c], errors='coerce')
    X_manual[c] = X_manual[c].fillna(X_manual[c].median())

for c in ['Neighborhood', 'Condition', 'Description']:
    X_manual[c] = X_manual[c].fillna('missing').astype(str)

X_manual = pd.get_dummies(X_manual, columns=['Neighborhood', 'Condition'], drop_first=False)
X_manual['Description_len'] = X_manual['Description'].str.split().str.len()
X_manual = X_manual.drop(columns=['Description'])

manual_model = RandomForestRegressor(n_estimators=200, random_state=42)
manual_scores = cross_validate(
    manual_model,
    X_manual,
    y,
    cv=cv,
    scoring={
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
)

print('\nKết quả xử lý thủ công + RandomForest:')
print({
    'RMSE_mean': -manual_scores['test_rmse'].mean(),
    'MAE_mean': -manual_scores['test_mae'].mean(),
    'R2_mean': manual_scores['test_r2'].mean(),
    'RMSE_std': manual_scores['test_rmse'].std()
})

print('\nNhận xét Bài 3:')
print('- Pipeline giảm lỗi do quên xử lý missing, unseen category, lệch dữ liệu.')
print('- CV trong pipeline chuẩn hơn vì mỗi fold tự fit preprocessing trên train fold.')
print('- Làm vậy sẽ tránh data leakage từ tập validation/test vào bước xử lý.')

# Fit model tốt nhất để lấy feature importance
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
best_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
best_pipe.fit(X, y)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    names = best_pipe.named_steps['preprocessor'].get_feature_names_out()
    imp_df = pd.DataFrame({
        'Feature': names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    imp_df['Norm_Importance'] = imp_df['Importance'] / imp_df['Importance'].sum()
    print('\nTop 15 feature importances:')
    print(imp_df.head(15))

    text_imp = imp_df[imp_df['Feature'].str.contains('txt__')]
    print('\nTop text features ảnh hưởng nhiều nhất:')
    print(text_imp.head(10))

# =========================================================
# 7. BÀI 4 - TRIỂN KHAI PIPELINE THÀNH SẢN PHẨM
# =========================================================
print('\n' + '='*60)
print('BÀI 4 - INFERENCE PIPELINE')
print('='*60)

final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42))
])
final_model.fit(X, y)

joblib.dump(final_model, 'house_price_pipeline.pkl')
print('Đã lưu model: house_price_pipeline.pkl')


def predict_price(new_data):
    model = joblib.load('house_price_pipeline.pkl')
    preds = model.predict(new_data)
    return preds

# Test với dữ liệu mới chưa thấy trong train
new_data = pd.DataFrame([
    {
        'LotArea': 5000,
        'Rooms': 4,
        'HasGarage': 1,
        'NoiseFeature': 0.25,
        'Neighborhood': 'Z',
        'Condition': 'Good',
        'Description': 'modern sunny house with garage and garden',
        'SaleDate': '2011-05-12',
        'ImagePath': 'house_images/new_house.png'
    },
    {
        'LotArea': 'abc',
        'Rooms': 3,
        'HasGarage': 0,
        'NoiseFeature': -0.5,
        'Neighborhood': 'A',
        'Condition': 'UnknownCondition',
        'Description': 'small quiet cozy home near park',
        'SaleDate': 'sai_ngay',
        'ImagePath': 'house_images/new_house2.png'
    }
])

preds = predict_price(new_data)
print('\nDự đoán giá nhà cho dữ liệu mới:')
for i, p in enumerate(preds, 1):
    print(f'Nhà {i}: {p:,.2f}')

print('\nTài liệu mô tả ngắn:')
print('1. Pipeline gồm: xử lý số + categorical + text + date rồi đưa vào model.')
print('2. Đầu vào: DataFrame có cùng các cột như lúc train, trừ SalePrice.')
print('3. Đầu ra: giá nhà dự đoán.')
print('4. Rủi ro: unseen category, data drift, sai format, text mới khác nhiều so với train.')

# =========================================================
# 8. BÀI 5 - GỢI Ý MỞ RỘNG CHO GIẢNG VIÊN
# =========================================================
print('\n' + '='*60)
print('BÀI 5 - GỢI Ý MỞ RỘNG')
print('='*60)
print('- Có thể thêm xử lý ảnh từ ImagePath bằng CNN hoặc đặc trưng ảnh.')
print('- Có thể triển khai thành file app Streamlit.')
print('- Có thể theo dõi drift dữ liệu sau khi deploy.')
