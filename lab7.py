import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv("ITA105_Lab_7.csv")

print(df.head())
print(df.shape)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
print(num_cols)

skew_df = pd.DataFrame({
    "Column": num_cols,
    "Skewness": [df[col].skew() for col in num_cols]
})

skew_df["Abs_Skew"] = skew_df["Skewness"].abs()
top10_skew = skew_df.sort_values("Abs_Skew", ascending=False).head(10)
print(top10_skew)

top3_cols = top10_skew.head(3)["Column"].tolist()
print(top3_cols)

for col in top3_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Histogram + KDE - {col}")
    plt.xlabel(col)
    plt.ylabel("Tần số")
    plt.show()

for col in top3_cols:
    s = df[col].skew()
    if s > 1:
        print(f"{col}: lệch phải mạnh, có thể có outlier lớn.")
        print("Gợi ý biến đổi: log hoặc Box-Cox.")
    elif s > 0.5:
        print(f"{col}: lệch phải nhẹ/vừa.")
        print("Gợi ý biến đổi: log hoặc power transform.")
    elif s < -1:
        print(f"{col}: lệch trái mạnh.")
        print("Gợi ý biến đổi: PowerTransformer (Yeo-Johnson).")
    elif s < -0.5:
        print(f"{col}: lệch trái nhẹ/vừa.")
        print("Gợi ý biến đổi: power transform.")
    else:
        print(f"{col}: phân phối khá cân đối.")
        print("Không cần biến đổi nhiều.")

print("""
Tác động của skewness lên mô hình:
- Dữ liệu lệch làm Linear Regression dễ bị ảnh hưởng bởi outlier.
- Sai số có thể lớn hơn, mô hình kém ổn định hơn.
- Các giá trị quá lớn có thể kéo đường hồi quy lệch đi.
""")

positive_cols = [col for col in num_cols if (df[col] > 0).all()]
mixed_cols = [col for col in num_cols if (df[col] <= 0).any()]

print(positive_cols)
print(mixed_cols)

col_pos1 = positive_cols[0]
col_pos2 = positive_cols[1]
col_mix = mixed_cols[0]

print(col_pos1)
print(col_pos2)
print(col_mix)

log_pos1 = np.log(df[col_pos1])
log_pos2 = np.log(df[col_pos2])

boxcox_pos1, lambda1 = boxcox(df[col_pos1])
boxcox_pos2, lambda2 = boxcox(df[col_pos2])

pt = PowerTransformer(method="yeo-johnson")
power_mix = pt.fit_transform(df[[col_mix]]).flatten()

def plot_before_after(original, transformed, title1, title2):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(original, kde=True, bins=30)
    plt.title(title1)
    plt.subplot(1, 2, 2)
    sns.histplot(transformed, kde=True, bins=30)
    plt.title(title2)
    plt.tight_layout()
    plt.show()

plot_before_after(df[col_pos1], log_pos1, f"Trước - {col_pos1}", f"Sau Log - {col_pos1}")
plot_before_after(df[col_pos1], boxcox_pos1, f"Trước - {col_pos1}", f"Sau Box-Cox - {col_pos1}")
plot_before_after(df[col_pos2], log_pos2, f"Trước - {col_pos2}", f"Sau Log - {col_pos2}")
plot_before_after(df[col_pos2], boxcox_pos2, f"Trước - {col_pos2}", f"Sau Box-Cox - {col_pos2}")
plot_before_after(df[col_mix], power_mix, f"Trước - {col_mix}", f"Sau Power - {col_mix}")

compare = pd.DataFrame({
    "Column": [col_pos1, col_pos2, col_mix],
    "Skew trước": [
        df[col_pos1].skew(),
        df[col_pos2].skew(),
        df[col_mix].skew()
    ],
    "Skew sau Log": [
        pd.Series(log_pos1).skew(),
        pd.Series(log_pos2).skew(),
        np.nan
    ],
    "Skew sau Box-Cox": [
        pd.Series(boxcox_pos1).skew(),
        pd.Series(boxcox_pos2).skew(),
        np.nan
    ],
    "Skew sau Power": [
        np.nan,
        np.nan,
        pd.Series(power_mix).skew()
    ]
})

remarks = []

for i in range(len(compare)):
    row = compare.iloc[i]
    values = {
        "Log": abs(row["Skew sau Log"]) if pd.notnull(row["Skew sau Log"]) else 999,
        "Box-Cox": abs(row["Skew sau Box-Cox"]) if pd.notnull(row["Skew sau Box-Cox"]) else 999,
        "Power": abs(row["Skew sau Power"]) if pd.notnull(row["Skew sau Power"]) else 999
    }
    best = min(values, key=values.get)
    remarks.append(f"{best} tốt nhất")

compare["Nhận xét"] = remarks

print(compare)
print(lambda1)
print(lambda2)

print("""
Ý nghĩa lambda trong Box-Cox:
- Lambda là tham số cho biết mức biến đổi phù hợp nhất.
- Nếu lambda gần 0 thì Box-Cox gần giống log.
- Mục tiêu là làm phân phối bớt lệch hơn.
""")

print("""
Phân tích:
- Log phù hợp với dữ liệu dương, lệch phải.
- Box-Cox thường mạnh hơn log vì tự tìm lambda tối ưu.
- PowerTransformer (Yeo-Johnson) dùng được khi dữ liệu có âm hoặc 0.
""")

target = "SalePrice"
feature_cols = [col for col in num_cols if col != target]

X = df[feature_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_A = LinearRegression()
model_A.fit(X_train, y_train)
pred_A = model_A.predict(X_test)

rmse_A = np.sqrt(mean_squared_error(y_test, pred_A))
r2_A = r2_score(y_test, pred_A)

y_train_log = np.log(y_train)
model_B = LinearRegression()
model_B.fit(X_train, y_train_log)

pred_B_log = model_B.predict(X_test)
pred_B_real = np.exp(pred_B_log)

rmse_B = np.sqrt(mean_squared_error(y_test, pred_B_real))
r2_B = r2_score(y_test, pred_B_real)

X_train_C = X_train.copy()
X_test_C = X_test.copy()

skew_cols_for_model = [col for col in feature_cols if abs(df[col].skew()) > 0.5]

pt_model = PowerTransformer(method="yeo-johnson")

X_train_C[skew_cols_for_model] = pt_model.fit_transform(X_train_C[skew_cols_for_model])
X_test_C[skew_cols_for_model] = pt_model.transform(X_test_C[skew_cols_for_model])

model_C = LinearRegression()
model_C.fit(X_train_C, y_train)
pred_C = model_C.predict(X_test_C)

rmse_C = np.sqrt(mean_squared_error(y_test, pred_C))
r2_C = r2_score(y_test, pred_C)

result_model = pd.DataFrame({
    "Mô hình": ["Version A - Raw", "Version B - Log Target", "Version C - Power Features"],
    "RMSE (test)": [rmse_A, rmse_B, rmse_C],
    "R²": [r2_A, r2_B, r2_C]
})

print(result_model)

best_model = result_model.sort_values("RMSE (test)").iloc[0]
print(best_model)

print("""
Phân tích ảnh hưởng:
- Log-transform ở biến mục tiêu giúp giảm ảnh hưởng của các giá trị SalePrice quá lớn.
- Power transform giúp các cột đầu vào bớt lệch, giảm nhiễu và bớt ảnh hưởng outlier.
- Nếu RMSE giảm thì transform đã giúp mô hình tốt hơn.
- Mô hình có RMSE thấp nhất và R² cao hơn thường là mô hình tốt hơn.
""")

business_cols = top10_skew.head(2)["Column"].tolist()
print(business_cols)

for col in business_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Version A - Raw data: {col}")
    plt.show()

    if (df[col] > 0).all():
        transformed = np.log(df[col])
        method = "log"
    else:
        temp_pt = PowerTransformer(method="yeo-johnson")
        transformed = temp_pt.fit_transform(df[[col]]).flatten()
        method = "power"

    plt.figure(figsize=(8, 4))
    sns.histplot(transformed, kde=True, bins=30)
    plt.title(f"Version B - Transformed ({method}): {col}")
    plt.show()

df["log_price_index"] = np.log(df["SalePrice"])

print(df[["SalePrice", "log_price_index"]].head())

print("""
Insight cho người không chuyên:
1. Tại sao cần biến đổi?
- Vì dữ liệu gốc bị lệch mạnh nên khó nhìn ra xu hướng chung.
- Một vài giá trị quá lớn có thể làm biểu đồ và phân tích bị méo.

2. Biểu đồ transform giúp gì?
- Dữ liệu đều hơn, dễ nhìn hơn.
- Dễ phát hiện nhóm khách hàng hoặc phân khúc giá.
- Dễ nhìn thấy bất thường hơn.

3. Ảnh hưởng đến hiểu thị trường/khách hàng:
- Giúp so sánh giá hoặc diện tích công bằng hơn.
- Hỗ trợ nhận ra khu vực giá cao bất thường.
- Giúp đọc xu hướng thị trường rõ hơn.

4. Ứng dụng của log_price_index:
- Phân nhóm khách hàng theo mức giá.
- Hỗ trợ phát hiện khu vực có giá bất thường.
- Hỗ trợ mô hình dự báo giá tốt hơn.

5. Khuyến nghị kinh doanh:
- Nên transform các biến lệch mạnh trước khi phân tích.
- Khi báo cáo nên xem cả dữ liệu gốc và dữ liệu sau transform.
- Với bài toán dự báo giá, nên thử log target hoặc power transform để giảm sai số.
""")
