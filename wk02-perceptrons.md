# Week 2: Perceptrons
## The perceptron algorithm
In pseudocode (from the [notes](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week2/perceptron/5)):

![Pseudocode for the perceptron algorithm](https://i.ibb.co/y84ZYsQ/images-perceptron-algorithm-codebox-1-crop.png)

The input parameters are the dataset $\mathcal{D}_n$ and the hyperparameter $\tau$.

The outer loop repeats $\tau$ (which is up to us) times. 

The inner loop goes through all all the data points (repeats $n$ times). For each iteration, it updates $\theta$ and $\theta_0$ only if the current $\theta$ and $\theta_0$ give an *incorrect* prediction. 

(Note that $y^{(i)}$ is the correct prediction and $\text{sgn}(\theta^Tx^{(i)} + \theta_0)$ is the guess. For $y^{(i)}(\theta^Tx^{(i)} + \theta_0)) > 0$, either both $y^{(i)}, (\theta^Tx^{(i)} + \theta_0)) > 0$ or both $y^{(i)}, (\theta^Tx^{(i)} + \theta_0)) < 0$ (i.e., the guess should be *correct*). If, instead, $y^{(i)}(\theta^Tx^{(i)} + \theta_0)) < 0$, $\text{sgn}(\theta^Tx^{(i)} + \theta_0)$ generated an incorrect prediction and $\theta$ and $\theta_0$ must be updated.)

If $\theta$ and $\theta_0$ already generate a perfect classifier for the dataset (i.e., one that gives the correct prediction for *each* data point), the `if` condition is never satisfied and $\theta$ and $\theta_0$ are never updated. 

An optimization: if the algorithm goes through the inner loop once (i.e., tests every data point for a current $\theta$ and $\theta_0$) without making an update, it will never make any updates for further iterations of the outer loop. So, a flag inside the inner loop should be set to `True` once an update is made. If the program exits out of the inner loop with a `False` flag, the outer loop should just terminate. 

Why is $\theta = \theta + y^{(i)}x^{(i)}$ and $\theta_0 = \theta_0 + y^{(i)}$ the correct update? Kaelbling says this will be proved using optimization methods in a later lecture.
## The perceptron algorithm in action - an example
The first point/example in the training data is always a mistake. Since $\theta = \vec{0}$ and $\theta_0 = 0$ initially, $\theta^T x + \theta_0 = 0 \implies y^{(i)} (\theta^T x + \theta_0) = 0$, triggering the `if` condition.

Kaelbling presents a [~3-minute demonstration](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week2/perceptron/4) of the perceptron in action.
## Evaluating learning algorithms - validation
Suppose you run your model learning algorithm to find that $\theta$ and $\theta_0$ produce the lowest error on the training set. Call this error $\mathcal{E}_n(\theta, \theta_0)$. $\mathcal{E}_n(\theta, \theta_0)$ is *too low* of an estimate for the error on the test data set --- it would have worked to just memorize the training data and get zero error; finding a model that generalizes to test data is the hard part.

To get a more accurate sense for the error in our model's predictions, we portion off some training data as *validation data*, i.e., data never shown to the model while training and used exclusively to find its error.

The plot thickens when we have to choose between multiple learning algorithms. Already we have seen two: the random linear classifier and the perceptron. In theory, we can generate models by using these algorithms upon training data, then use the validation data to find the models' errors, then pick the model with the lower error. But here's the catch: we selected for performance against the validation data, which *we're not supposed to do*! The validation data was exclusively for predicting the model's error on test data, and by selecting the model with the best performance on validation data, we used validation data a little like training data. We might have overfitted!

Later in the course, we'll study how to use training data cleverly and find accurate error estimates for test data.
## Perceptron - overview of plan
Usually, learning algorithms are discovered when someone formulates an objective, then derives an algorithm which would do well with respect to that objective. The perceptron evolved in the opposite fashion --- [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) came up with the algorithm first, then mathematicians proved why it worked (well).

In the next few videos, we will prove how the perceptron satisfies certain kinds of correctness.
## Perceptron through origin algorithm
![Pseudocode for the perceptron through origin algorithm](https://i.ibb.co/8bHFJcH/images-perceptron-offset-codebox-1-crop.png)

In this version of the perceptron, the separator always passes through the origin; there's no offset $\theta_0$ (or rather, it is implicitly zero).
## Theory of perceptron - Linear separability
A training data set $\mathcal{D}$ is **linearly separable** if there exist $\theta, \theta_0$ such that for all $i$:

$$y^{(i)}(\theta^Tx^{(i)} + \theta_0) > 0$$

Another way to say this is that some linear separator model exists which can predict correctly for each point in the data set:

$$h(x^{(i)}; \theta, \theta_0) = y^{(i)}$$

Another way to say this is that the training error is zero:

$$\mathcal{E}_n(h) = 0$$ 
## Theory of perceptron - margin of a dataset
The **margin of a labeled data point** $(x,y)$ with respect to a separator $\theta, \theta_0$ is:

$$y \cdot \frac{\theta^Tx+\theta_0}{\lVert\theta\rVert}$$

$\frac{\theta^Tx+\theta_0}{\lVert\theta\rVert}$ is the signed distance of the point from the hyperplane. Its product with $y$ will be positive only if $h(x) = \text{sgn}(\theta^Tx+\theta_0) = y$. So, it will be negative if the point is incorrectly classified.

The margin *measures* how good the prediction was. If the point is positive and distant from the hyperplane, the prediction was very good. If it is closer to the plane, its signed distance $\frac{\theta^Tx+\theta_0}{\lVert\theta\rVert}$ will be smaller, and thus the margin will be smaller. 

The **margin of a data set** is the lowest margin of any data point within it:

$$\min_i \left( y^{(i)}\cdot \frac{\theta^Tx^{(i)} + \theta_0}{\lVert \theta \rVert} \right)$$
## Perceptron convergence theorem
For simplicity, consider only linear separators through the origin. 

If there exists some $\theta^{\*}$ such that:
 
- $y^{(i)} \frac{\theta^{\*} x^{(i)}}{\lVert \theta^{\*} \rVert} \ge \gamma$ ($\gamma > 0$) ($\gamma$ for "gap") for all $i$ (i.e., the margin of $\mathcal{D}$ with respect to $\theta^{\*}$ is a positive $\gamma$; in other words, the data set $\mathcal{D}$ is linearly separable), and
- $\lVert x^{(i)} \rVert \le R$ (i.e., some circle of radius $R$ contains all the data points),

the perceptron will make at most $\left(\frac{R}{\gamma}\right)^2$ mistakes. 

Further, after the algorithm has finished running, the hypothesis generated will be a *perfect* linear separator of the data.
## Proof sketch of the perceptron convergence theorem
Initialize $\theta^{(0)} = \vec{0}$. Let $\theta^{(k)}$ be the hyperplane after the perceptron algorithm has made $k$ mistakes. Let $\theta^{\*}$ be the good separator. Note that both $\theta^{(k)}$ and $\theta^{\*}$ go through the origin. We will show that the angle between them decreases usefully on each post-mistake update.

The cosine of the angle between the separators is:

$$\begin{align\*}
\cos(\theta^{(k)}, \theta^{\*}) &= \frac{\theta^{(k)} \cdot \theta^{\*}}{\lVert \theta^{(k)} \rVert \lVert \theta^{\*} \rVert} \\
&= \left(\frac{\theta^{(k)} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} \right) \left(\frac{1}{\lVert \theta^{(k)} \rVert} \right)
\end{align\*}$$

We will consider the upper bounds of these factors in turn. 

Bounding the first factor:

$$\begin{align\*}
\frac{\theta^{(k)} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} &= \frac{\left( \theta^{(k-1)} + y^{(i)} x^{(i)} \right) \theta^{\*}}{\lVert \theta^{\*} \rVert} \\
&= \frac{ \theta^{(k-1)} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} + \frac{\left( y^{(i)} x^{(i)} \right) \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} \\
&\ge  \frac{ \theta^{(k-1)} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} + \gamma
\end{align\*}$$

To complete this proof by induction, note:

$$\begin{align\*}
\frac{\theta^{(1)} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} &= \frac{\theta^{(0)} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} + \frac{\left( y^{(i)} x^{(i)} \right) \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} \\
&= \underbrace{\frac{\vec{0} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert}}_{0} + \frac{\left( y^{(i)} x^{(i)} \right) \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} \\
&\ge \gamma
\end{align\*}$$

So:

$$
\frac{\theta^{(k)} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} \ge k\gamma
$$

Before looking at the second factor, note that since $(x^{(i)}, y^{(i)})$ is classified incorrectly after the $(k-1)$ th mistake, $y^{(i)} \left(\theta^{(k-1)} \cdot x^{(i)} \right) \le 0$ (i.e., the prediction $\text{sgn}\left(\theta^{(k-1)} \cdot x^{(i)} \right)$ differs from $y^{(i)}$). Also, whatever $y^{(i)}$ is ($+1$, $0$ or $-1$), $\left(y^{(i)}\right)^2$ is $0$ or $1$. 

Bounding the second factor:

$$\begin{align\*}
\lVert \theta^{(k)} \rVert ^{2} &= \lVert \theta^{(k-1)} + y^{(i)} x^{(i)} \rVert ^{2} \\  
&= \lVert \theta^{(k-1)} \rVert ^{2} +  \underbrace{2y^{(i)}\left(\theta^{(k-1)} \cdot x^{(i)} \right)}\_{\le 0} + \underbrace{\lVert x^{(i)} \rVert ^{2}}\_{\le R^{2}} \underbrace{(y^{(i)})^{2}}\_{1 \text{ or } 0} \\
&\le \lVert \theta^{(k-1)} \rVert ^{2} + R^{2} 
\end{align\*}$$

To complete this proof by induction, note:

$$\begin{align\*}
\lVert \theta^{(1)} \rVert ^{2} &= \lVert \underbrace{\theta^{(0)}}\_{\vec{0}} + y^{(i)} x^{(i)} \rVert ^{2} \\
&= \lVert y^{(i)} x^{(i)} \rVert ^{2} \\
&\le R^{2}
\end{align\*}$$

So:

$$
\lVert \theta^{(k)} \rVert ^{2} \le kR^2
$$

Substituting these bounds in the dot product formulation:

$$ \begin{align\*}
\cos\left(\theta^{(k)}, \theta^{\*} \right) &= \left(\frac{\theta^{(k)} \cdot \theta^{\*}}{\lVert \theta^{\*} \rVert} \right) \left(\frac{1}{\lVert \theta^{(k)} \rVert} \right) \\
&\ge \left(k\gamma \right) \left(\frac{1}{\sqrt{k} R} \right) \\
&\ge \sqrt{k}\left(\frac{\gamma}{R} \right)
\end{align\*} $$

Since $\cos\left(\theta^{(k)}, \theta^{\*} \right) \le 1$ for any $\theta^{(k)}, \theta^{\*}$,

$$\begin{align\*}
1 &\ge \sqrt{k}\left(\frac{\gamma}{R} \right) \\
k &\le \left(\frac{R}{\gamma} \right)^{2}
\end{align\*}$$

Thus, the number of mistakes $k$ will be at most $\left(\frac{R}{\gamma} \right)^{2}$.

Note that while the perceptron is guaranteed to converge on *some* hypothesis, it may not necessarily be the *best possible* hypothesis.

This [Cornell CS lecture note](https://www.cs.cornell.edu/courses/cs4780/2022fa/lectures/lecturenote03.html) illustrates a perceptron updating:

![Perceptron update diagram](https://i.ibb.co/64m1Vpr/Perceptron-Update.png)
