import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


data = pd.read_csv('cleaned.csv')


print('=====EXPLORATORY DATA ANALYSIS=====')
print('Shape:', data.shape)
print('Preview:\n', data.head())
print("Info:\n", data.info())
print("Describe:\n", data.describe())


print('Missing values check:')
print(data.isnull().sum())


scaler = StandardScaler()

scaled_cols = [
    "study_hours", "sleep_hours", "attendance",
    "screen_time", "activities", "stress", "prev_gpa"
]

data_scaled = data.copy()
data_scaled[scaled_cols] = scaler.fit_transform(data_scaled[scaled_cols])


#VISUALIZATION

# # CORRELATION HEATMAP
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()

# # DATA DISTRIBUTION 
# data.hist(figsize=(12, 8))
# plt.tight_layout()
# plt.show()

# sns.scatterplot(x="attendance", y="cgpa", data=data)
# plt.title("Attendance vs CGPA")
# plt.show()


# sns.scatterplot(x="study_hours", y="cgpa", data=data)
# plt.title("Study Hours vs CGPA")
# plt.show()


# sns.scatterplot(x="screen_time", y="cgpa", data=data)
# plt.title("Phone Usage vs CGPA")
# plt.show()

# MODEL TRAINING

X = data.drop(columns=['cgpa'])
y = data['cgpa']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_pred_linear_reg = linear_reg.predict(X_test)


mae_linear_reg = mean_absolute_error(y_test, y_pred_linear_reg)
mse_linear_reg = mean_squared_error(y_test, y_pred_linear_reg)
rmse_linear_reg = np.sqrt(mse_linear_reg)
r2_linear_reg = r2_score(y_test, y_pred_linear_reg)

# print("Linear Regression Metrics")
# print("MAE:", mae_linear_reg)
# print("MSE:", mse_linear_reg)
# print("RMSE:", rmse_linear_reg)
# print("R2:", r2_linear_reg)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2, random_state=42)

lr_poly = LinearRegression()
lr_poly.fit(X_train_p, y_train_p)


y_pred_poly = lr_poly.predict(X_test_p)

# print("Polynomial Regression Metrics")
# print("MAE:", mean_absolute_error(y_test_p, y_pred_poly))
# print("MSE:", mean_squared_error(y_test_p, y_pred_poly))
# print("RMSE:", np.sqrt(mean_squared_error(y_test_p, y_pred_poly)))
# print("R2:", r2_score(y_test_p, y_pred_poly))

coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_reg.coef_
})

print("MLE Coeffs:\n", coeff_df.sort_values(by="Coefficient", ascending=False))

def cgpa_category(cgpa):
    if cgpa < 7:
        return "Low"
    elif cgpa < 8.5:
        return "Medium"
    else:
        return "High"

data["cgpa_category"] = data["cgpa"].apply(cgpa_category)

print("CGPA Distribution:\n", data['cgpa_category'].value_counts())

le = LabelEncoder()
data["cgpa_cat_encoded"] = le.fit_transform(data["cgpa_category"])


X_cls = data_scaled[[
    "study_hours",
    "sleep_hours",
    "attendance",
    "screen_time",
    "activities",
    "stress",
    "prev_gpa"
]]

y_cls = data["cgpa_cat_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

def evaluate_model(model , X_test, y_test, y_prob=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy Score: {acc:.2f}")
    print(f"Precision Score: {prec:.2f}")
    print(f"Recall Score: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        print(f"ROC AUC Score: {auc:.2f}")

# log_reg = LogisticRegression()
# log_reg.fit(X_train, y_train)

# y_prob_lr = log_reg.predict_proba(X_test)

# print('=====LOGISTIC REGRESSION METRICS=====')
# evaluate_model(log_reg, X_test, y_test, y_prob_lr)


# dt = DecisionTreeClassifier(max_depth=5, random_state=42)
# dt.fit(X_train, y_train)

# print('=====DESCISION TREE CLASSIFIER METRICS=====')
# evaluate_model(dt, X_test, y_test)


# nb = GaussianNB()
# nb.fit(X_train, y_train)

# print('=====NAIVES BAYES METRICS=====')
# evaluate_model(nb, X_test, y_test)


# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)

# print('=====K NEAREST NEIGHBOURS CLASSIFIER METRICS=====')
# evaluate_model(knn, X_test, y_test)

svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train, y_train)

y_prob_svm = svm.predict_proba(X_test)
print('=====SUPPORT VECTOR CLASSIFIER METRICS=====')
evaluate_model(svm, X_test, y_test, y_prob_svm)