[TOC]

# Notes on Expectation Propagation

## Basics

### Moment matching

Suppose we want to approximate an unknown target distribution (a posterior for example) $\hat{p}(\boldsymbol{\theta})$ by a spherical Gaussian
$$
q(\boldsymbol{\theta})=\mathcal{N}\left(\boldsymbol{\theta}\vert \mathbf{m}, v\mathbf{I}\right)=\frac{1}{(2\pi v)^{d/2}}\exp\left[-\frac{1}{2v}(\boldsymbol{\theta}-\mathbf{m})^\top(\boldsymbol{\theta}-\mathbf{m})\right]. \label{eq:spherical-gaussian}
$$
We can minimize the reverse KL divergence
$$
\begin{align}
    D_{\text{KL}} \left[q(\boldsymbol{\theta}) \parallel \hat{p}(\boldsymbol{\theta}) \right] &= \int q(\boldsymbol{\theta}) \ln q(\boldsymbol{\theta}) \text{d}\theta - \int q(\boldsymbol{\theta}) \ln \hat{p}(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta},
\end{align}
$$
which is widely used in Variational Inference. However, we can also choose to minimize the forward KL divergence:
$$
\begin{align}
    \label{eq:KL}
    D_{\text{KL}} \left[\hat{p}(\boldsymbol{\theta}) \parallel q(\boldsymbol{\theta}) \right] &= \int \hat{p}(\boldsymbol{\theta}) \ln \hat{p}(\boldsymbol{\theta}) \text{d}\theta - \int \hat{p}(\boldsymbol{\theta}) \ln q(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta}\nonumber \\
    &= - \int \hat{p}(\boldsymbol{\theta}) \ln q(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta}+ \text{const}.
\end{align}
$$
Taking derivatives with respect to $\mathbf{m}$ and $v$ gives: 
$$
\begin{align}
    \nabla_{\mathbf{m}} D_{\text{KL}} \left[\hat{p}(\boldsymbol{\theta}) \parallel q(\boldsymbol{\theta}) \right] &= - \int \hat{p}(\boldsymbol{\theta}) \frac{\partial \ln q(\boldsymbol{\theta})}{\partial \mathbf{m}} \text{d}\boldsymbol{\theta}\nonumber \\
    &= - \int \hat{p}(\boldsymbol{\theta}) \left(\frac{\boldsymbol{\theta}- \mathbf{m}}{v}\right) \text{d}\boldsymbol{\theta}\\
    \nabla_{v} D_{\text{KL}} \left[\hat{p}(\boldsymbol{\theta}) \parallel q(\boldsymbol{\theta}) \right] &= - \int \hat{p}(\boldsymbol{\theta}) \frac{\partial \ln q(\boldsymbol{\theta})}{\partial v} \text{d}\boldsymbol{\theta}\nonumber \\
    &= - \int \hat{p}(\boldsymbol{\theta}) \left[-\frac{d}{2v} + \frac{1}{2v^2} \left( \boldsymbol{\theta}^\top \boldsymbol{\theta}-2 \boldsymbol{\theta}^\top \mathbf{m} + \mathbf{m}^\top\mathbf{m}\right) \right] \text{d}\boldsymbol{\theta}
\end{align}
$$
By setting derivatives to zero and rearranging equations, we arrive at
$$
\begin{align}
    \label{eq:moment-matching-1}
    \mathbf{m} &= \int \hat{p}(\boldsymbol{\theta}) \boldsymbol{\theta}\text{d}\boldsymbol{\theta}= \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right] \\
    \label{eq:moment-matching-2}
    v d + \mathbf{m}^\top \mathbf{m}&= \int \hat{p}(\boldsymbol{\theta}) \boldsymbol{\theta}^\top \boldsymbol{\theta}\text{d}\boldsymbol{\theta}= \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}^\top \boldsymbol{\theta}\right]
\end{align}
$$
Eq. ($\ref{eq:moment-matching-1},\ref{eq:moment-matching-2}$) says that minimizing the forward KL divergence is equivalent to setting the mean $\mathbf{m}$ equal to the mean of $\hat{p}(\boldsymbol{\theta})$ and setting the covariance $v\mathbf{I}$ equal to the covariance of $\hat{p}(\boldsymbol{\theta})$. It is called **moment matching**, which will be very useful later.

### Gaussian integral

We will use the following Gaussian integral result:
$$
\begin{align}
\label{eq:gaussian-int}
\int\exp\left[-\frac{1}{2}\mathbf{x}^\top\mathbf{A}\mathbf{x}+\mathbf{c}^\top\mathbf{x}\right]\text{d}\mathbf{x}=\sqrt{\det(2\pi\mathbf{A}^{-1})}\exp\left[\frac{1}{2}\mathbf{c}^\top\mathbf{A}^{-\top}\mathbf{c}\right]
\end{align}
$$

## Assumed-density filtering

### Background: the clutter problem

Before introducing the Expectation Propagation (EP) formally, we would like to talk about the assumed-density filtering (ADF), which is a special case of EP and forms the basis for EP. As a concrete example, consider the clutter problem discussed in Minka's thesis. The datapoints are sampled from a mixture of Gaussians: 
$$
\begin{align}
    p(\mathbf{x}\vert \boldsymbol{\theta})&=(1-w)\mathcal{N}\left(\mathbf{x} \vert \boldsymbol{\theta}, \mathbf{I}\right)+w\mathcal{N}\left(\mathbf{x} \vert \mathbf{0}, 10\mathbf{I}\right), \\
    p(\boldsymbol{\theta})&\sim\mathcal{N}(\mathbf{0}, 100\mathbf{I}),
\end{align}
$$
where $\boldsymbol{\theta}$ is a $d$-dimensional unknown parameters we want to infer, $p(\mathbf{x}\vert \boldsymbol{\theta})$ is the likelihood and $p(\boldsymbol{\theta})$ is the prior. Given i.i.d. datapoints $\mathcal{D}=\{\mathbf{x}_1, \mathbf{x}_2, \cdots, \mathbf{x}_N\}$, the joint distribution of data $\mathcal{D}$ and parameter $\boldsymbol{\theta}$ is
$$
p(\mathcal{D},\boldsymbol{\theta})=p(\boldsymbol{\theta})p(\mathcal{D}\vert \boldsymbol{\theta})=
p(\boldsymbol{\theta}) \prod_{i=1}^{N}p(\mathbf{x}_i\vert \boldsymbol{\theta}),
$$
and we are interested in estimating the posterior distribution $p(\boldsymbol{\theta}\vert \mathcal{D})$ (inferring the unknown parameters $\boldsymbol{\theta}$): 
$$
p(\boldsymbol{\theta}\vert \mathcal{D})=\frac{p(\boldsymbol{\theta})p(\mathcal{D}\vert \boldsymbol{\theta})}{p(\mathcal{D})} = \frac{p(\boldsymbol{\theta})p(\mathcal{D}\vert \boldsymbol{\theta})}{\int p(\boldsymbol{\theta})p(\mathcal{D}\vert \boldsymbol{\theta})\text{d}\boldsymbol{\theta}}.
$$

### ADF in a Nutshell

The main steps of ADF is summarized as follows.

#### Factorize the joint distribution

First, the joint distribution $p(\mathcal{D},\boldsymbol{\theta})$ is factorized as
$$
p(\mathcal{D},\boldsymbol{\theta})=p(\boldsymbol{\theta}) \prod_{i=1}^{N}t_i(\boldsymbol{\theta}),
$$
where $t_i(\boldsymbol{\theta})=p(\mathbf{x}_i\vert \boldsymbol{\theta})$. From my point of view, it's just renaming the likelihood.

#### Approximate the posterior by a tractable distribution

We would like to approximate the posterior by a tractable distribution $q(\boldsymbol{\theta})$. In this clutter problem, we choose spherical Gaussian in Eq. ($\ref{eq:spherical-gaussian}$): 
$$
q(\boldsymbol{\theta})=\mathcal{N}\left(\boldsymbol{\theta} \vert \mathbf{m}_{\theta}, v_{\theta}\mathbf{I}\right)
\label{eq:parameterized}
$$
and the learnable parameters are $\{\mathbf{m}_\theta, v_\theta\}$.

#### Calculate the approximate posterior

Suppose that given datapoints $\mathcal{D}_{i-1}=\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_{i-1}\}$, our current estimate of the posterior is $q(\boldsymbol{\theta}\vert \mathcal{D}_{i-1})$. Then, after observing a new datapoint $\mathbf{x}_{i}$, we can update the posterior using Bayes' rule: 
$$
\underbrace{p(\boldsymbol{\theta}\vert \mathcal{D}_{i})}_{\text{new posterior}} \propto \underbrace{p(\mathbf{x}_{i} \vert \boldsymbol{\theta})}_{\text{likelihood}} \quad\underbrace{q(\boldsymbol{\theta}\vert \mathcal{D}_{i-1})}_{\text{prior}},
$$
where $p(\boldsymbol{\theta}\vert \mathcal{D}_{i})$ is our new estimate of the posterior, $p(\mathbf{x}_{i} \vert \boldsymbol{\theta})$ is the likelihood and the "old" posterior $q(\boldsymbol{\theta}\vert \mathcal{D}_{i-1})$ can be treated as the prior. In the clutter problem, the above equation becomes 
$$
\begin{align}
\label{eq:update-new-posterior}
    \hat{p}(\boldsymbol{\theta})=\frac{t_i(\boldsymbol{\theta})q(\boldsymbol{\theta})}{\int t_i(\boldsymbol{\theta})q(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta}}=\frac{t_i(\boldsymbol{\theta})q(\boldsymbol{\theta})}{Z_i},
\end{align}
$$
where $\hat{p}(\boldsymbol{\theta})$ is the new posterior, $q(\boldsymbol{\theta})$ is the "old" posterior and $Z_i$ is the normalizing factor.

#### Update parameters

After obtaining the new posterior in Eq. ($\ref{eq:update-new-posterior}$), the final step is to update our parameters $\{\mathbf{m}_\theta, v_\theta\}$ so that the new $q(\boldsymbol{\theta})$ is as close as possible to $\hat{p}(\boldsymbol{\theta})$. (Note that $\hat{p}(\boldsymbol{\theta})$ is not what we want. What we want is a spherical Gaussian parameterized by $\{\mathbf{m}_\theta, v_\theta\}$ so we must “project” $\hat{p}(\boldsymbol{\theta})$ back into $q(\boldsymbol{\theta})$.) This is done by minimizing the forward KL divergence between $\hat{p}(\boldsymbol{\theta})$ and $q(\boldsymbol{\theta})$ and **moment matching** tells us that 
$$
\begin{align}
		\label{eq:adf-1}
    \mathbf{m}_\theta^\text{new} &= \int \hat{p}(\boldsymbol{\theta}) \boldsymbol{\theta}\text{d}\boldsymbol{\theta}= \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right] \\
    \label{eq:adf-2}
    v_\theta^\text{new} d + \mathbf{m}_\theta^{\text{new} \top} \mathbf{m}_\theta^\text{new}&= \int \hat{p}(\boldsymbol{\theta}) \boldsymbol{\theta}^\top \boldsymbol{\theta}\text{d}\boldsymbol{\theta}= \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}^\top \boldsymbol{\theta}\right]
\end{align}
$$
Next time, a new datapoint $\mathbf{x}_{i+1}$ comes in and we can update parameters $\{\mathbf{m}_\theta, v_\theta\}$ sequentially in this way.

### More details in ADF

#### How to compute the expectations

The next question is: how to compute the expectations on the r.h.s of Eq. ($\ref{eq:adf-1}, \ref{eq:adf-2}$)? We resort to the following property: for arbitrary function $t(\boldsymbol{\theta})$ and spherical Gaussian $q(\boldsymbol{\theta})=\mathcal{N}(\mathbf{m}_{\theta},v_{\theta}\mathbf{I})$, we have 
$$
\begin{align}
    \hat{p}(\boldsymbol{\theta}) &= \frac{t(\boldsymbol{\theta})q(\boldsymbol{\theta})}{\int t(\boldsymbol{\theta})q(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta}}=\frac{t(\boldsymbol{\theta})q(\boldsymbol{\theta})}{Z} \\
    \frac{\partial \ln Z}{\partial \mathbf{m}_{\theta}} &= \frac{1}{Z}\int t(\boldsymbol{\theta}) \frac{\partial q(\boldsymbol{\theta})}{\partial \mathbf{m}_\theta} \text{d}\boldsymbol{\theta}\nonumber \\
    &= \frac{1}{Z} \int t(\boldsymbol{\theta}) \frac{q(\boldsymbol{\theta})(\boldsymbol{\theta}- \mathbf{m}_\theta)}{v_\theta} \text{d}\boldsymbol{\theta}\nonumber \\
    &= \frac{1}{v_{\theta}} \left[\mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right] - \mathbf{m}_{\theta}\right], \label{eq:lnZ-to-m}\\
    \frac{\partial \ln Z}{\partial v_{\theta}} &= \frac{1}{Z}\int t(\boldsymbol{\theta}) \frac{\partial q(\boldsymbol{\theta})}{\partial v_\theta} \text{d}\boldsymbol{\theta}\nonumber \\
    &= \frac{1}{Z} \int t(\boldsymbol{\theta}) q(\boldsymbol{\theta}) \left[-\frac{d}{2v_\theta} + \frac{1}{2v_\theta^2} \left( \boldsymbol{\theta}^\top \boldsymbol{\theta}-2 \boldsymbol{\theta}^\top \mathbf{m}_\theta + \left\lVert\mathbf{m}_\theta\right\rVert^2\right) \right] \text{d}\boldsymbol{\theta}\nonumber \\
    &= \frac{1}{2v_{\theta}^2} \left\{\mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}^\top \boldsymbol{\theta}\right] - 2\mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right]\mathbf{m}_{\theta} + \left\lVert\mathbf{m}_{\theta}\right\rVert^2\right\} -\frac{d}{2v_{\theta}}. \label{eq:lnZ-to-v}
\end{align}
$$
Rearranging the above equations we obtain 
$$
\begin{align}
    \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right] &= v_\theta \frac{\partial \ln Z}{\partial \mathbf{m}_{\theta}} + \mathbf{m}_\theta \\
    \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}^\top \boldsymbol{\theta}\right] &= 2 v_\theta^2 \frac{\partial \ln Z}{\partial v_{\theta}} + v_\theta d + 2 v_\theta \frac{\partial \ln Z}{\partial \mathbf{m}_{\theta}} \mathbf{m}_\theta + \left\lVert\mathbf{m}_\theta\right\rVert^2 \\
    \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}^\top \boldsymbol{\theta}\right] - \left(\mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right]\right)^\top \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right] &= v_\theta d - v_\theta^2 \left[ \left(\frac{\partial \ln Z}{\partial \mathbf{m}_\theta}\right)^\top  \frac{\partial \ln Z}{\partial \mathbf{m}_\theta} - 2 \frac{\partial \ln Z}{\partial v_\theta} \right]
\end{align}
$$
As long as we can compute $\frac{\partial \ln Z}{\partial \mathbf{m}_\theta}$ and $\frac{\partial \ln Z}{\partial v_\theta}$, we are able to calculate those expectations.

#### Equations for the clutter problem

For the clutter problem, there exits analytical solution. To recap, our current estimate (after observing $\{\mathbf{x}_1, \cdots, \mathbf{x}_{i-1}\}$) of the posterior is given by Eq. ($\ref{eq:parameterized}$). Then after observing a new datapoint $\mathbf{x}_i$, we calculate the expectations of $\hat{p}(\boldsymbol{\theta})$ and then set $\mathbf{m}^{\text{new}}_{\boldsymbol{\theta}}$ equal to $\mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right]$ and $d v^{\text{new}}_{\theta}$ equal to $\mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}^\top \boldsymbol{\theta}\right] - \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right]^\top \mathbb{E}_{\hat{p}}\left[\boldsymbol{\theta}\right]$. First, we calculate the normalizing factor $Z_i$: 
$$
\begin{align}
    Z_i &= \int t_i(\boldsymbol{\theta})q(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta}\nonumber \\
    &= \int \left[(1-w)\mathcal{N}\left(\mathbf{x}_i \vert \boldsymbol{\theta}, \mathbf{I}\right)+w\mathcal{N}\left(\mathbf{x}_i \vert \mathbf{0}, 10\mathbf{I}\right)\right]q(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta}\nonumber \\
    &= \int \left[(1-w)\mathcal{N}\left(\mathbf{x}_i \vert \boldsymbol{\theta}, \mathbf{I}\right)+w\mathcal{N}\left(\mathbf{x}_i \vert \mathbf{0}, 10\mathbf{I}\right)\right]\mathcal{N}\left(\boldsymbol{\theta} \vert \mathbf{m}_{\theta}, v_{\theta}\mathbf{I}\right) \text{d}\boldsymbol{\theta}\nonumber \\
    &= (1-w)\mathcal{N}\left(\mathbf{x}_i \vert \mathbf{m}_{\theta}, (v_{\theta}+1)\mathbf{I}\right) + w\mathcal{N}\left(\mathbf{x}_i \vert \mathbf{0}, 10\mathbf{I}\right)
\end{align}
$$
In deriving the above equation, we use the Gaussian integral in Eq. ($\ref{eq:gaussian-int}$) so that $\int \mathcal{N}\left(\mathbf{x}_i \vert \boldsymbol{\theta}, \mathbf{I}\right) \mathcal{N}\left(\boldsymbol{\theta} \vert \mathbf{m}_{\theta}, v_{\theta}\mathbf{I}\right)\text{d}\boldsymbol{\theta}=\mathcal{N}\left(\mathbf{x}_i \vert \mathbf{m}_{\theta}, (v_{\theta}+1)\mathbf{I}\right)$.

By defining $r_i = \frac{1}{Z_i} (1-w)\mathcal{N}\left(\mathbf{x}_i \vert \mathbf{m}_{\theta}, (v_{\theta}+1)\mathbf{I}\right)$, the derivatives are given by
$$
\begin{align}
    \frac{\partial \ln Z_i}{\partial \mathbf{m}_{\theta}} &= r_i \frac{\mathbf{x}_i - m_{\boldsymbol{\theta}}}{v_{\theta}+1} \\
    \frac{\partial \ln Z_i}{\partial v_{\theta}} &= -\frac{d r_i}{2(v_{\theta}+1)} + \frac{r_i}{2(v_{\theta}+1)^2}(\mathbf{x}_i-\mathbf{m}_{\theta})^\top(\mathbf{x}_i-\mathbf{m}_{\theta})
\end{align}
$$
and the final update equations are 
$$
\begin{align}
    \mathbf{m}^{\text{new}}_{\theta} &= \mathbf{m}_{\theta} + v_{\theta} r_i  \frac{\mathbf{x}_i - \mathbf{m}_{\theta}}{v_{\theta}+1} \\
    v^{\text{new}}_{\theta} &= v_{\theta} - r_i \frac{v_{\theta}^2}{v_{\theta}+1} + \frac{r_i (1 - r_i) v_{\theta}^2}{(v_{\theta}+1)^2 d}(\mathbf{x}_i-\mathbf{m}_{\theta})^\top(\mathbf{x}_i-\mathbf{m}_{\theta})
\end{align}
$$

## Expectation Propagation

### EP in a Nutshell

The core idea of EP is summarized as follows.

#### Factorize the joint distribution

Same as in ADF.

#### Approximate the posterior by a tractable distribution

In EP, rather than using a single Gaussian, we choose to approximate the posterior by the product of Gaussians:
$$
\begin{align}
\label{eq:product-of-factors}
q(\boldsymbol{\theta})=\frac{\prod_{i=1}^N \tilde{t}_i(\boldsymbol{\theta})}{\int \prod_{i=1}^N \tilde{t}_i(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta}},
\end{align}
$$
where each $\tilde{t}_i(\boldsymbol{\theta})$ takes the form of Eq. ($\ref{eq:parameterized}$) with parameters $\{\mathbf{m}_\theta^i, v_\theta^i\}$. A nice property of Gaussian is that the product of Gaussian PDF is also a Gaussian PDF. In this way,  $q(\boldsymbol{\theta})$ is also a Gaussian with mean $\mathbf{m}_{\theta}$ and variance $v_{\theta}\mathbf{I}$. If we remove a Gaussian factor, the remaining product of Gaussians is also a Gaussian with mean $\mathbf{m}_{\theta}^{\setminus i}$ and variance $v^{\setminus i}_{\theta}\mathbf{I}$. 

Although $q(\boldsymbol{\theta})$ is still a spherical Gaussian (same as ADF, so if the posterior is a mixture of gaussian, it can only capture a single component), we hope that by introducing these product factors $\tilde{t}_i(\boldsymbol{\theta})$, it will result in a more accurate estimate of the posterior. 

#### Calculate the approximate posterior

Suppose that given all datapoints $\mathcal{D}_N=\{\mathbf{x}_1,\mathbf{x}_2,\cdots,\mathbf{x}_N\}$, our estimate of the posterior is $q(\boldsymbol{\theta})$ given by Eq. ($\ref{eq:product-of-factors}$). Every time we randomly remove $\mathbf{x}_i$. After $\mathbf{x}_i$ being removed, our current estimate of the *leave-one-out* posterior $q^{\setminus i}(\boldsymbol{\theta})$ is given by
$$
\begin{align}
    q^{\setminus i}(\boldsymbol{\theta}) = \frac{q(\boldsymbol{\theta})}{\tilde{t}_i(\boldsymbol{\theta})}.
\end{align}
$$
Then suppose we "re-observing" $\mathbf{x}_i$. We can calculate the new posterior $q^{\text{new}}(\boldsymbol{\theta})$, as we did in ADF using Bayes' rule: 
$$
\begin{align}
\hat{p}(\boldsymbol{\theta})=\frac{t_i(\boldsymbol{\theta})q^{\setminus i}(\boldsymbol{\theta})}{\int t_i(\boldsymbol{\theta})q^{\setminus i}(\boldsymbol{\theta}) \text{d}\boldsymbol{\theta}}=\frac{t_i(\boldsymbol{\theta})q^{\setminus i}(\boldsymbol{\theta})}{Z_i}.
\end{align}
$$

#### Update parameters

Same as in ADF, we then update parameters $\{\mathbf{m}_\theta, v_\theta\}$ so that the new $q^{\text{new}}(\boldsymbol{\theta}^{\text{}})$ is as close as possible to $\hat{p}(\boldsymbol{\theta})$ by moment matching. (Note that in fact, we don't need to conpute $q(\boldsymbol{\theta})$ by the product of Gaussian using Eq. ($\ref{eq:product-of-factors}$), it is given by moment matching!) Finally, since we have updated our new posterior, we can refine $\tilde{t}_i(\boldsymbol{\theta})$ (update parameters $\{\mathbf{m}_\theta^i, v_\theta^i\}$) according to 
$$
\tilde{t}_i(\boldsymbol{\theta})=Z_i \frac{q^{\text{new}}(\boldsymbol{\theta})}{q^{\setminus i}(\boldsymbol{\theta})}.
$$

### More details in EP

#### How to compute the leave-one-out distribution

Most of equations are the same as in ADF. For the update equations, see Minka's thesis or PRML. Here we only introduce how to calculate the leave-one-out distribution: the division of two spherical Gaussian 
$$
\begin{align}
    q^{\text{new}}(\boldsymbol{\theta})=\frac{\mathcal{N}\left(\boldsymbol{\theta} \vert \mathbf{m}_1, v_1 \mathbf{I}\right)}{\mathcal{N}\left(\boldsymbol{\theta} \vert \mathbf{m}_2, v_2 \mathbf{I}\right)}
\end{align}
$$
is also a Gaussian random variable with mean and variance 
$$
\begin{align}
    (v^{\text{new}})^{-1} &= (v_1)^{-1} - (v_2)^{-1} \\
    \mathbf{m}^{\text{new}} &= \mathbf{m}_2 + (v_2 + v^{\text{new}})v_2^{-1}(\mathbf{m}_1 - \mathbf{m}_2)
\end{align}
$$

