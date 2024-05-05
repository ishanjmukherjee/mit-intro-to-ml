# Week 4: Margin Maximization

## Machine learning as optimization - framework

The idea is to turn the problem of training a machine learning model into an *optimization* problem. Maybe then we'll be able to apply the vast literature on optimization to ML problems.

We write an **objective function** $J(\Theta)$ which captures how we feel about the model $\Theta$ (e.g., how well it does on training data, how generalizable it is to test data). We generally try to minimize $J(\Theta)$, i.e., find a model $\Theta^{*}$ such that 

$$\Theta^{\*} = \arg\min_\Theta J(\Theta)$$

(Note that $\Theta$ denotes *all* the parameters that define a model. For a linear classifier, that's just $\theta$ and $\theta_0$, so $\Theta = {\theta, \theta_0}$.)

A common objective function is

$$J(\Theta) = \left( \frac{1}{n} \sum_{i=1}^{n} \underbrace{\mathcal{L}(h(x^{(i)}; \Theta), y^{(i)})}\_{\text{loss}} \right) +\underbrace{\lambda}\_{\text{hyperparameter}} \underbrace{R(\Theta)}_{\text{regularizer}}$$

The **loss** captures how unhappy we are about the prediction $h(x^{(i)}; \Theta)$ that the model $\Theta$ makes on a particular data point $x^{(i)}$ when the actual label is $y^{(i)}$.

On the other hand, the **regularizer** taxes complexity. If a model is too complex, it's a sign that it has overfit to training data and consequently might perform poorly on test data.

For example, below, $h_1$ has zero training loss, but it makes sense to prefer $h_2$ in spite of its two misclassifications. This is because we expect that it will perform better on test data drawn from the same distribution as this training data.

![](https://i.ibb.co/rM0vLKT/images-logistic-regression-regularization-tikzpicture-1-crop.png)

A common regularizer is

$$R(\Theta) = \| \Theta - \Theta_{prior} \|^2$$

when we have some idea in advance that $\Theta$ ought to be near some value $\Theta_{prior}$. In the absence of such knowledge, $\Theta_{prior} = 0$, i.e.,

$$R(\Theta) = \| \Theta \|^2$$   

$\lambda$ is a hyperparameter. It tells us how much we are willing to trade off loss on the training data versus preference over hypotheses (e.g., a preference for simpler models).

## Logistic regression - setting and sigmoid function

An example loss function: the 0-1 loss function used in the perceptron gives a $0$ for a *correct* prediction and $1$ for an *incorrect* prediction:

$$L_{01} (g, a) = 
\begin{cases}
0 & \text{if } a = g \\
1 & \text{otherwise}
\end{cases}$$

($g$ is "guess" and $a$ is "actual".)

This isn't ideal, since we want to be able to express 

1. *how right/wrong* our guess is, and
2. non-categorical predictions, i.e., the prediction should carry with it a degree of certainty about its guess.

A new hypothesis class, **linear logistic classifiers**, solves these issues. These hypotheses are still parametrized by a $d$-dimensional vector $\theta$ and a scalar $\theta_0$, but instead of making predictions in $\{+1, -1\}$, they make real-valued predictions in $(0, 1)$. 

A linear logistic classifier is of the form

$$h(x; \theta, \theta_0) = \sigma(\theta^Tx + \theta_0)$$

where $\sigma(z)$ is the **sigmoid** or **logistic function**

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

The function's plot is:

![](https://i.ibb.co/N3m7ZyZ/images-logistic-regression-a-new-hypothesis-class-linear-logistic-classifiers-tikzpicture-1-crop.png)

The range of $\sigma$ is $(0, 1)$. Importantly, it is never exactly $0$ or $1$. Because of this, the output of $\sigma$ can be interpreted as a probability.

## Linear logistic classifier - hypothesis class

The sigmoid function can predict a range of real values. In order to be a classifier, it must be modified to output a value from a finite set, like $\{+1, -1\}$.

We can set **thresholds** on $\sigma$ outputs to make it a classifier. 

For example, the classifier predicts $+1$ if $\sigma(\theta^Tx + \theta_0) > 0.5$, and $-1$ otherwise. (We can modify the threshold. For example, we can set the threshold less than $0.5$ if a false positive is less costly than a false negative, e.g., when deciding whether to send emergency assistance to an area.) 

Interestingly, $\sigma(\theta^Tx + \theta_0) > 0.5$ when $\theta^T x + \theta_0 > 0$, so we're really using linear classifiers under the hood! Here's a short proof:

$$\begin{align\*}
\sigma(\theta^Tx + \theta_0) &> 0.5 \\
\frac{1}{1+e^{-(\theta^Tx+\theta_0)}} &> 0.5 \\
0.5 + 0.5e^{-(\theta^Tx+\theta_0)}  &< 1 \\
0.5e^{-(\theta^Tx+\theta_0)} &< 0.5\\
e^{-(\theta^Tx+\theta_0)}  &< 1 \\
e^{-(\theta^Tx+\theta_0)} &< e^0 \\
-(\theta^Tx+\theta_0) &< 0 \\
\theta^Tx+\theta_0 &> 0
\end{align\*}$$

But setting this up as a linear logistic classifier rather than a linear classifier is useful because:

1. it makes optimization easier, and
2. a continuous value can convey more information than a simple yes/no!

This plots $\sigma$ against one-dimensional data for three different $(\theta, \theta_0)$:

![](https://i.ibb.co/HNWf1zY/images-logistic-regression-a-new-hypothesis-class-linear-logistic-classifiers-tikzpicture-2-crop.png)

Marking the $x$ where $\sigma$ reaches the threshold gives us the classifier. 

When our inputs $x^{(i)}$ are in two-dimensional space with features $x_1$ and $x_2$, the output of the linear logistic classifier is a surface, as shown below (picture from [here](https://www.codeproject.com/Articles/1207728/Build-Simple-AI-NET-Library-Part-ML-Algorithms)):

![](https://i.ibb.co/1ZSGJTJ/45.png)
