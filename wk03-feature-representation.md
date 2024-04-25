# Week 3: Features
## Feature representation - transforming through-origin to not-through-origin

A **linear separator not through the origin can be written as a linear separator through the origin** by transforming the data and $\theta$. Initially,

$$h\left(x; \theta, \theta_0 \right) = \text{sign}\left(\theta^Tx + \theta_0 \right)$$

Consider transforming each data point such that

$$\phi\left(\left[x_1, \dots, x_d \right] \right) = \left[x_1, \dots, x_d, 1 \right]$$

and transforming $\theta$ such that

$$\theta_{\text{new}} = \left[\theta_1, \dots, \theta_d, \theta_0 \right]$$

Now, 

$$\begin{align*}
h\left(x_{\text{new}}; \theta_{\text{new}} \right) &= x_1\theta_1 + \dots + x_d\theta_d+ \theta_0 \\
&= \theta^T x + \theta_0
\end{align*}$$

## Feature representation - polynomial basis

Consider this dataset in 1-D:

![Non linearly separable points in 1-D: + followed by -, then -, then +](https://i.ibb.co/7kgZdDs/Screenshot-2024-04-08-201000.png)

These points are not linearly separable, but after the transformation $\phi(x) = \left[x, x^2 \right]$, they are separable. One of the possible separators is:

![Linearly separable data set in 2-D after applying the transformation](https://i.ibb.co/LJqH1bp/Screenshot-2024-04-08-201243.png)

Assume that the separator shown above is $x^2 - 1 > 0$. This linear separator looks like a nonlinear separator in the original, 1-D space: 

![A nonlinear separator in 1-D](https://i.ibb.co/nDPwR14/Screenshot-2024-04-08-201511.png)

(Why does it look this way? We find which values of $x$ satisfy $x^2 - 1 = 0$. The answers are $1$ and $-1$, so these points form our separator. The regions satisfying $x^2-1>0$ are $x>1$ and $x<-1$, so these are labeled as positive.)

This solution was very dataset-specific. A more systematic approach is to construct a new **feature space** using a **polynomial basis**. The idea is to choose a positive integral $k$, then include a feature for every possible product of $k$ different dimensions in the original input.

| Order ($k$) | $d=1$ | in general | 
| :---: | :---: | :---: |
| 0 | $\left[1 \right]$ | $[1]$ |
| 1 | $\left[1, x \right]$ | $\left[1, x_1, \dots, x_d \right]$ |
| 2 | $\left[1, x, x^2 \right]$ | $\left[1, x_1, \dots, x_d, x_1^2, x_1x_2, \dots \right]$ |
| 3 | $\left[1, x, x^2, x^3 \right]$ | $\left[1, x_1, \dots, x_d, x_1^2, x_1x_2, \dots, x_1^3, x_1^2x_2, x_1x_2x_3, \dots \right]$ |
| $\vdots$ | $\vdots$ | $\vdots$ |

Consider the XOR dataset:

![XOR dataset](https://i.ibb.co/Bsyqncm/Screenshot-2024-04-08-200820.png)

$k=1$ doesn't work. But if we try $k=2$, the feature transformation is:

$$\phi\left(\left(x_1, x_2 \right) \right) = \left(1,x_1, x_2, x_1^2, x_1x_2, x_2^2 \right)$$

After 4 iterations, the perceptron finds the separator $\theta = \left(0, 0, 0, 0, 4, 0 \right), \theta_0 = 0$, i.e.

$$0 + 0x_1 + 0x_2 + 0x_1^2 + 4x_1x_2 + 0x_2^2+ 0 = 0$$

In 2-D, this looks like a nonlinear separator:

![Separating into quadrants where the data points lie](https://i.ibb.co/mDMJ7ND/Screenshot-2024-04-08-204329.png)

Some data sets require high $k$ to find separators. For example, after $200$ iterations for bases of orders $2$, $3$, $4$ and $5$, the separators still don't work for second- and third-order basis representations:

![](https://i.ibb.co/9pk1QVS/Screenshot-2024-04-08-204708.png)

Fourth- and fifth-order basis representations *work* in that they find correct classifiers, but they may be overfitting.

## Feature representation strategies for dealing with varied data

Say you have discrete values to be classified into categories, e.g., mobile phone companies (Apple, Samsung, etc). What are some ways to represent this data?

- **Numeric coding**: assign Apple to be $1$, Samsung to be $2$, Nokia to be $3$, and so on. This isn't ideal, because Samsung isn't "greater" than Apple, and the "difference" between Samsung and Apple and Nokia and Samsung isn't "equal" in the sense the numbers assigned to them imply. However, the machine "learns" the arbitrary ordering as significant. This is illustrated by problem 2 in the week 3 homework.
- **One-hot coding**: Say your discrete feature has $m$ possible values. The feature can be encoded as a vector of $m$ booleans. So, Apple could be $\left[ 1, 0, \dots, 0\right]$, Samsung could be $\left[0, 1, \dots, 0 \right]$, etc. (Could also use $-1$ for "false" instead of $0$ --- you'd have to play with both and see which works out better.)
- **Binary code**: Could represent $k$ values using a vector of length just $\log k$ instead of length $k$ required for one-hot coding. For a concrete example, consider this encoding: Apple $\rightarrow 00$, Samsung $\rightarrow 01$, Nokia $\rightarrow 10$, Motorola $\rightarrow 11$. With this encoding, we only need $\log 4 = 2$ bits instead of the $4$ bits that would be required by one-hot coding. But binary encoding is a bad idea, because it forces the algorithm to have to figure out how to decode binary. 
In general, compression makes representation smaller but harder to work with. We want "more roomy, easy to interpret" encodings in machine learning. 

Also consider **factoring** values into separate features:

- The "car" feature can be decomposed into "make" and "model" features.
- The blood group feature $\{\text{A+}, \text{A-}, \text{B+}, \text{B-}, \text{AB+}, \text{AB-}, \text{O+}, \text{O-}\}$ can be decomposed into three separate features: $\{\text{A}, \text{not A}\}$, $\{\text{B}, \text{not B}\}$, $\{+, -\}$. (Note that $\text{O}$ can be treated as having neither feature $\text{A}$ nor feature $\text{B}$.) Two plausible encodings:
	-  6-D vector with two dimensions to encode each of the factors using a one-hot encoding. So, $\text{A+}$ is $\left(1, 0, 0, 1, 1, 0\right)$, i.e, $\left(\text{A}, \text{not B}, + \right)$ and $\text{AB+}$ is $\left(1, 0, 1, 0, 1, 0 \right)$
	- 3-D vector with one dimension for each factor, encoding its presence as $1.0$ and absence as $-1.0$ (which sometimes works better than $0.0$). So, $\text{A+}$ is $\left(1.0, -1.0, 1.0 \right)$ and $\text{AB+}$ is $\left(1.0, 1.0, 1.0 \right)$.

If you know something about how your inputs are structured, you can make make a big difference to the success of your learning algorithm with your choice of how to represent the features of the problem.

If a feature has much larger values than another, it will take the algorithm a lot of time to find parameters to put them on an equal basis. **Standardize** your feature values to keep $R$ small:

$\phi \left(x\right) = \frac{x-\overline{x}}{\sigma}$ 

$\overline{x}$ is the mean of $x^{(i)}$ and $\sigma$ is the standard deviation of $x^{(i)}$.
