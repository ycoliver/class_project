from sklearn.linear_model import LinearRegression
import numpy as np
x = np.array([10,8,13,15,9])
y = np.array([60,55,75,80,64])

### fitting the model

model = LinearRegression()  
model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

print('coef:',model.coef_[0])
print('b:',model.intercept_)


### predict
y_pred = model.coef_[0] * x + model.intercept_
print('fitting model predict:', y_pred)




