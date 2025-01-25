def J(theta_0, theta_1, x, y):
    m = len(y)  # number of samples
    cost = 0
    for i in range(m):
        cost += ((theta_0 + theta_1 * x[i]) - y[i]) ** 2
        print(x[i], y[i])
        print(cost)
    return (1 / (2 * m)) * cost


# Input example
theta_0 = 1 # change that with 0 and u'll see that the cost function is perfect
theta_1 = 2
x = [1, 2, 3]  # x value
y = [2, 4, 6]  # y value

# compute the final cost
result = J(theta_0, theta_1, x, y)
print("Costo:", result)

