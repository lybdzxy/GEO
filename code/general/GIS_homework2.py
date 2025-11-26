import pandas as pd
from sklearn.linear_model import LinearRegression

#数据读取
data_path=f'E:/gis/Week05-OLS/data/housing.csv'
data=pd.read_csv(data_path)

#看看数据是啥样的
pd.set_option('display.max_columns', None)
print(data)

#对房价（median_house_value）与收入（median_income）做一元线性回归
x=data['median_income']
y=data['median_house_value']
reg = LinearRegression()
reg.fit(x.values.reshape([-1, 1]), y)
print("beta0 =", reg.intercept_)
print("beta1 =", reg.coef_[0])
#二者呈现正相关性，收入越高房价越高

#ocean_proximity有`1H OCEAN`、`INLAND`、`ISLAND`、`NEAR BAY`、`NEAR OCEAN`五种，根据我的个人感觉，将它们由好到坏依次赋值，值越大越好
ocean_proximity_mapping = {
    "ISLAND": 5,
    "NEAR BAY": 4,
    "NEAR OCEAN": 3,
    "1H OCEAN": 2,
    "INLAND": 1
}
data["ocean_proximity_encoded"] = data["ocean_proximity"].map(ocean_proximity_mapping)
print(data[["ocean_proximity", "ocean_proximity_encoded"]])


#对房价（median_house_value）和收入（median_income）、房屋年龄（housing_median_age）、房间数（total_rooms）、人口（population）、户数（households）、临海性（ocean_proximity）做多元线性回归，并对结果进行分析和解读。<br>
x_columns = [
    'median_income',
    'housing_median_age',
    'total_rooms',
    'population',
    'households',
    'ocean_proximity_encoded'
]
data.dropna(subset=x_columns, inplace=True)#删有缺失值的行
X = data[x_columns]
y = data['median_house_value']

reg = LinearRegression().fit(X, y)

print("intercept =", reg.intercept_)
print("     coef =", reg.coef_)
#房价与收入、房屋年龄、户数和临海性呈正相关，与房间数、人口呈负相关，收入、房屋年龄、户数和临海性越高，房间数、人口越低，房价越高