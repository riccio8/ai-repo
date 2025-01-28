from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [2, 5, 10, 17]
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
print(model.predict(poly.transform([[5]])))
