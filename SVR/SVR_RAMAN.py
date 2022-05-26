# the SVR model for Raman spectrum
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

pX = r"../data/X_var.csv"
py = r"../data/y_var.csv"

with open(pX, encoding='utf-8') as f:
    X = np.loadtxt(f, delimiter=",")
with open(py, encoding='utf-8') as f:
    y = np.loadtxt(f, delimiter=",")

svr_rbf = SVR(kernel="rbf", C=1, gamma="auto", epsilon=0.1)
svr_lin = SVR(kernel="linear", C=1, gamma="auto")
svr_poly = SVR(kernel="poly", C=1, gamma="auto", degree=4, epsilon=0.1, coef0=1)

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ["RBF", "Linear", "Polynomial"]
model_color = ["m", "c", "g"]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 6))
for ix, svr in enumerate(svrs):
    for iy in range(3):
        axes[ix, iy].scatter(
            y[70:, iy],
            svr.fit(X[:70], y[:70, iy]).predict(X[70:]),
            marker='+',
            color=model_color[ix]
        )
        axes[ix, iy].scatter(
            y[:70, iy],
            svr.fit(X[:70], y[:70, iy]).predict(X[:70]),
            marker='.',
            color=model_color[ix]
        )
        axes[ix, iy].plot(
            [min(y[:, iy]), max(y[:, iy])],
            [min(y[:, iy]), max(y[:, iy])]
        )
        axes[ix, iy].set_xlabel("实际值",fontsize=8)
        axes[ix, iy].set_ylabel("预测值",fontsize=8)

fig.text(0.25, 0.04, r"研究法辛烷值", ha="center", va="center",fontsize=8, color='b')
fig.text(0.5, 0.04, r"密度（20℃）", ha="center", va="center",fontsize=8, color='b')
fig.text(0.75, 0.04, r"初馏点", ha="center", va="center",fontsize=8, color='b')
fig.text(0.06, 0.75, r"径向基函数核", ha="center", va="center",fontsize=8, color='b', rotation="vertical")
fig.text(0.06, 0.5, r"线性核", ha="center", va="center",fontsize=8, color='b', rotation="vertical")
fig.text(0.06, 0.25, r"多项式核", ha="center", va="center",fontsize=8, color='b', rotation="vertical")
fig.suptitle("SVR", fontsize=10)
plt.show()
