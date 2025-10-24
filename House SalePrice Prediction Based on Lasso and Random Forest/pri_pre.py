import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

df = pd.read_excel(r"C:\Users\Cynth\houpr_predi\price.xlsx")
numerical_features = ["MSSubClass","LotArea","OverallCond","YearRemodAdd","BsmtFinSF2","TotalBsmtSF"]
df_filtered = df[ numerical_features + ["SalePrice"]]
x = df_filtered[numerical_features]
y = df_filtered["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

encoder = ColumnTransformer(
    transformers=[
        ("numbers","passthrough",numerical_features),
    ],
    remainder="drop"
)
data = pd.concat([x, y], axis=1).dropna()
x = data[numerical_features]
y = data["SalePrice"]

lasso = Lasso(max_iter=100000)
params = {"alpha":[0.001,0.01,0.1,1]}
grid_search = GridSearchCV(lasso, params,scoring="r2",cv=5)
grid_search.fit(x_train,y_train)
lasso_best = grid_search.best_estimator_
lasso_train_pred = lasso_best.predict(x_train)
lasso_test_pred = lasso_best.predict(x_test)
print("Lasso:")
print(f"best alpha:{grid_search.best_params_['alpha']}")
print(f"train score:{r2_score(y_train,lasso_train_pred):.2f}")
print(f"test score:{r2_score(y_test,lasso_test_pred):.2f}")

rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(x_train,y_train)
rf_train_pred = rf_model.predict(x_train)
rf_test_pred = rf_model.predict(x_test)
print("\nRandom Forest:")
print(f"train score:{r2_score(y_train,rf_train_pred):.2f}")
print(f"test score:{r2_score(y_test,rf_test_pred):.2f}")

encoder.fit(x)
example_data_raw = pd.DataFrame(
    [[90,9000,7,2001,0,1179]],
    columns=["MSSubClass","LotArea","OverallCond","YearRemodAdd","BsmtFinSF2","TotalBsmtSF"]
)
example_data_encoded = encoder.transform(example_data_raw)

lasso_pred = lasso_best.predict(example_data_encoded)
print(f"\nLasso Prediction of House SalePrice: {lasso_pred[0]:.2f}")
rf_pred = rf_model.predict(example_data_encoded)
print(f"Random Forest Prediction of House SalePrice: {rf_pred[0]:.2f}")