from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [2.5, 5, 7.5, 10]
model = LinearRegression()
model.fit(X, y)

print(model.predict([[5]]))  
