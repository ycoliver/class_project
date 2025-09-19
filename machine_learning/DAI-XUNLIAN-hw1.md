# Q1
## Q1a

### A1a
监督学习与非监督学习的主要区别在于损失函数的优化目标是否与给定的真实标签有关系，监督学习需要每条样本对应一个真实的标签（优化方向），比如SFT、支持向量机、逻辑回归等等，而非监督学习往往通过样本间的内部关系确定优化目标，比如对比学习等等


## Q1b
### A1b

1) False 2) False 3) False 4) True

---

## Q1c
### A1c
如果X是列满秩矩阵$n * d$,那么对于任意非零向量$x$，$Xx = b$有唯一解，且$b$一定是非零向量

考虑$(X x)^T (X x) = b^T b > 0$，
令$X^T X = Y$，那么$x^T X^T X x = x^T Y x > 0$对任意非零向量$x$都成立，即$X^T X = Y$是正定矩阵

# Q2
## Q2a

### A2a

X可以被SVD分解为
$$
X = V * [\Sigma, 0] * [U_1^T; U_2^T] = V \Sigma U_1^T
$$
令$V \Sigma = A$, $A$满秩
求解X的最小二乘解等价于求解 $A$关于$U_1^T \theta = z$的最小二乘解，此解可以被表示为$(A^T A)^{-1} A^T y = (\Sigma^TV^TV \Sigma)^{-1} \Sigma^T V^T y$

由于$V$是正交矩阵，上式等价于$z^* = (\Sigma^T \Sigma)^{-1}\Sigma^T  V^T y = \Sigma^{-1}V^T y = U_1^T * \theta$

那么对于$U_1^T*\theta = z^*$，代入$\theta_p = U_1 z^*$，满足$U_1^T U_1 z^* = z^*$，即$U_1 z^*$为一个特解

而对于$U_1^T \theta = 0$，$U_2$可以满足齐次解，所以$\theta$的通解为$U_1 \Sigma^{-1}V^T y + U_2 w$，其中$w$是任意向量



## Q2b

### A2b
直接对最优化方程求导，可以得到导数为0时，满足
$(X^T X + \lambda I)w = X^T y$

那么最优解为 $w^*=(X^TX+\lambda I)^{-1}X^T y$

# Q3
## Q3a

### A3a:
似然函数最大化：$L(\theta) = P(y|\theta) = P(\epsilon) = \prod(\epsilon_i) = \prod(e^{-|\epsilon_i|/b}/{2b})$

等价于对数似然函数最大化：$l(\theta) = \sum(log(1/{2b}) - |\epsilon_i|/b) = n*log(1/{2b} - \sum(|\epsilon_i|/b))$

$\epsilon_i = y_i - (X*\theta)_i$

所以：$l(\theta) = n*log(1/(2b)) - \sum(|y_i - (X*\theta)_i|)/b$

要使得$L(\theta)$最大化等价于$\sum(|y_i - (X*\theta)_i|)$最小化，即$argmin_{\theta}||y - X*\theta||_1$

## Q3b

### A3b
$h_\mu(z_j) = \begin{cases} 
z_j^2/{2\mu} & |z_j| < \mu \\
|z_j| - \mu/2 & |z_j| >= \mu
\end{cases}$

$h_j'(z_j) = \begin{cases}
z_j/\mu & |z_j| < \mu \\
1 & z_j > \mu \\
-1 & z_j < -\mu
\end{cases}$

所以 $L(\theta) = H_\mu(X\theta - y ) = H_\mu(z)$，令$z = X\theta - y$

$\nabla L(\theta) = H_\mu'(z) = X^T\nabla H'(X\theta-y)$

## Q3c

### A3c
see in the python file "..code_source/p3/p3.py"

# Q4
## Q4a

### A4a

对于完全分类的解$\theta^*$：对$\forall i \in N$, 有$yi(\theta^*)x_i > 0$ (两项同号),所以 $\rho = \min y_i (\theta^* x_i) > 0$恒成立

## Q4b

### Q4a

证明:
已知
$$
\begin{cases}
y_{k-1} (\theta_{k-1}^T x_{k-1}) < 0 \\
\theta_k = \theta_{k-1} + y_j x_j^T

\end{cases}
$$

**分情况讨论**

假设 $\rho = min_{1<=i<=n} y_i(\theta^{*T})x_i = y_j (\theta^{*T}x_i)$ 表示在最小的索引

①k-1 = j, 此时$\theta = \theta_{k-1} + y_j x_j^T = y_jx_j^T$

即$\theta_K^T\theta^* = \theta_{k-1}^T \theta^* + y_jx_j^T \theta^* = \theta_{k-1}^T\theta^* + y_j\theta^{*T}x_j = \theta_{k-1}^T\theta^* + \rho$

②$k-1 \neq b$, 此时$\theta_k = \theta_{k-1} + y_jx_j^T$

即$\theta_k^T\theta^* = \theta_{k-1}^T \theta^* + y_{k-1}x_{k-1}^T \theta^* = \theta_{k-1}^T\theta^* + y_{k-1}\theta^{*T}x_{k-1} \geq \theta_{k-1}^T\theta^* + yj\theta^{*T}x_j = \theta_{k-1}^T\theta^* + \rho$

综上所述$\theta_k^T\theta^* \geq \theta_{k-1}^T\theta^* + \rho$ , 同时由数学归纳法,$\theta_k^T\theta^* \geq k\rho$

## Q4c

### A4c

已知
$$
\begin{cases}
y_{k-1} (\theta_{k-1}^T x_{k-1}) < 0 \\
\theta_k = \theta_{k-1} + y_j x_j^T

\end{cases}
$$

$\theta_k \theta_{k-1}=\theta_{k-1}\theta{k-1} + y_{k-1}x_{k-1}^T\theta_{k-1}$,由于$y_{k-1}x_{k-1}^T\theta_{k-1}<0$,$\theta_k\theta_{k-1}<\theta_{k-1}\theta_{k-1}=||\theta_{k-1}||^2$

而$||\theta_k||^2 = ||(\theta{k-1}+y_{k-1}x_{k-1}^T)||^2 = ||\theta_{k-1}||^2 + 2||\theta_{k-1}y_{k-1}x_{k-1}^T|| + ||y_{k-1}x_{k-1}^T||^2 + m$, 
其中 $y \in {\{-1,1\}}$,常数$m<0$

所以$||\theta_k||^2 \leq ||\theta_{k-1}||^2 + ||x_{k-1}||^2$

## Q4d

### A4d

由于
$||\theta_k||^2 \leq ||\theta_{k-1}|| + ||x_{k-1}||^2 \leq ||\theta_{k-1}||^2 + max_{1 \leq i \leq n}||x_i||^2$

所以
$||\theta_k||^2 \leq k*\max_{1 \leq i \leq n}||x_i||^2 = k R^2$



## Q4e

### A4e

已知

$$
\begin{cases}
\theta_k^T\theta^* \geq \theta_{k-1}^T \theta^* + \rho \\ 
||\theta_k||^2  \leq k * R^2 \\ 
R = max_{1 \leq i \leq n} ||x_i||
\end{cases}
$$
那么

$$\frac{\theta_k^*\theta^*}{||\theta_k||} \geq \frac{k\rho}{||\theta_k||} \geq \frac{k\rho}{\sqrt{k}R^2} = \sqrt{k} \frac{\rho}{R}$$

又由于
$$
\frac{\theta_k\theta^*}{||\theta_k||||\theta^*||} \leq 1
$$
且对于$\overline{k}$, $\theta_{\overline{k}}=\theta^*$
于是
$$
\overline{k} \leq \frac{R^2||\theta^*||^2}{\rho^2}
$$
