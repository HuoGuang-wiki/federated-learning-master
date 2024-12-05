import numpy as np
import matplotlib.pyplot as plt
# Size of the points dataset.
m = 20

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))

# Points y-coordinate
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
'''
plt.plot(X1,y)
plt.show()
plt.scatter(X1,y)
plt.show()
'''
# The Learning Rate alpha.
alpha = 0.01

def error_function(theta, X, y):
    '''Error function J definition.(误差函数J)'''
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    '''Gradient of the function J definition..(误差函数J梯度定义，对theta求导结果)'''
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y) #梯度下降最快的方向
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta

optimal = gradient_descent(X, y, alpha)
print('optimal:', optimal)
print('error function:', error_function(optimal, X, y)[0,0])
#绘制拟合后的图形和散点图
plt.rcParams['font.sans-serif'] = 'SimHei' #用于正常显示中文
plt.title('数据散点图及其梯度下降法拟合出的直线') ## 添加标题
plt.xlabel('x')## 添加x轴的名称
plt.ylabel('y')## 添加y轴的名称
plt.xlim([0,20])## 确定x轴范围
plt.ylim([0,20])## 确定y轴范围
plt.xticks([0,5,10,15,20])## 规定x轴刻度
plt.yticks([0,5,10,15,20])## 确定y轴刻度
plt.plot(X,optimal[0]+optimal[1]*X)
plt.scatter(X1,y)
plt.show()
