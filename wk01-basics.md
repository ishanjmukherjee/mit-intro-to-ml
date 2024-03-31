# Week 1: Basics
## Perspective and history
Fundamentally, machine learning is about **getting data in some form and aggregating in a way that lets us make predictions**, e.g., predictions about the stock market or the weather, or the direction a robot must turn in to help achieve some goal.

A motivating use case for machine learning is **computer vision**:

- Today, face-detection software in smartphone cameras is ubiquitous. 
- Some decades ago, researchers were trying to come up with algorithms to detect faces. These mostly didn't work so well. 
- What worked was writing programs which analyzed data (classified into "face" and "not-face") and spit out predictions. 
- So, humans still have to write programs. It's just that these programs are one level of abstraction further up. We are not writing the image-classifying algorithm  itself, we're writing a program that can analyze data to decide how to recognize a face.

## Estimation and generalization

The challenges of machine learning, step-by-step:

- Get data.
- Define the space of possible solutions.
- Characterize the objective function.
- Find an algorithm which could find a good solution.
- Run said algorithm.
- Validate the results.

The *only* part of the problem that the computer does for us is running the algorithm. Machine learning is not magic!

A deep philosophical challenge in machine learning is using data the model was *trained on* make predictions about *other data*. Kaelbling calls this almost an act of "hubris". For example: getting data on temperature in a location and using it to make predictions about temperatures there in the future. 

## Supervised learning: setting

The best-understood setting of machine learning is **supervised learning**. In supervised learning, you get a dataset:

$$\mathcal{D}_{n} = \lbrace(x^{(1)}, y^{(1)}), \dots, (x^{(n)}, y^{(n)})\rbrace$$ 

You can think of each $x^{(i)}$ as *input* and each $y^{(i)}$ as *output*. The goal is to learn a mapping (i.e., a relationship) such that given any $x^{(i)}$, we can predict a $y^{(i)}$.

In this course, $x^{(i)}$ will usually be $d$-dimensional vectors:

$$x^{(i)} \in \mathbb{R}^d$$

In the first few weeks of this course, we will work on *binary classification*:

$$y^{(i)} \in \lbrace+1, -1\rbrace$$

An example of a binary search problem: given an image, determine whether there is a face in it (say, $+1$) or not ($-1$).

**Feature representation**:

- The data we care about (songs, people, cars, etc) cannot be abstractly fed into a model. 
- A feature representation $\psi(x) \in \mathbb{R}^n$ has to be devised by humans first. 
- That said, throughout the course, we will abstract this away. We are truly mapping $\psi{(x^{(i)})}$ to $y^{(i)}$, but saying that we are mapping $x^{(i)}$ to $y^{(i)}$ will suffice.

## Supervised learning: hypotheses

An individual hypothesis is some rule mapping inputs to outputs:

$$h(x^{(i)}; \theta) = y^{(i)}$$

$\theta$ is a parameter.

Each hypothesis belongs to a space of all possible hypotheses, called a **hypothesis class**:

$$h \in \mathcal{H}$$ 

One of our tasks in machine learning is to find this hypothesis class.

## Evaluating predictions: loss functions

The **loss function** captures how "sad" we are that our model predicted some $g \in \lbrace+1,-1\rbrace$ (for "guess") when some $a \in \lbrace+1,-1\rbrace$ (for "actual") was the answer:

$$\mathcal{L}(g,a)$$

## Evaluating predictions: training set error

Minimizing training data loss may not minimize test data loss. For example: a student preparing for an exam can memorize all the homework problems and have near-zero loss on those. However, the exam will have questions they've never seen before; so, they need to learn a hypothesis that *generalizes* to have a small loss on *new data*.

For now, though, we will aim for a small loss on training data as a *proxy* for small loss on new data. 

Training set has $n$ elements: $\lbrace(x^{(1)},y^{(1)}), \dots, (x^{(n)},y^{(n)})\rbrace$. Training set error is:

$$\epsilon_n(h) = \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(h(x^{(i)}),y^{(i)})$$

Training set has $n'$ elements: $\lbrace(x^{(n+1)},y^{(n+1)}), \dots, (x^{(n+n')},y^{(n+n')})\rbrace$. Test set error is:

$$\epsilon_{n'}(h) = \frac{1}{n'} \sum_{i=n+1}^{n+n'} \mathcal{L}(h(x^{(i)}),y^{(i)})$$

## Learning algorithms

We have to write a **learning algorithm** parametrized on the hypothesis class $\mathcal{H}$ which takes in a training data set $\mathcal{D}$ and outputs a hypothesis $h$. Note that we don't individually write the hypotheses themselves; instead, we write an algorithm that looks at the data and finds a good hypothesis for us.  

$$\begin{CD} 
@>\mathcal{D}>> \boxed{\text{Learning Algorithm}(\mathcal{H})}  @>h>> \end{CD}$$

You can sometimes intuitively guess a good learning algorithm, but in the vast majority of cases, we use optimization methods.

## Linear classifiers

**Linear classifiers** are a hypothesis class:

$$h(x;\theta,\theta_0) = \text{sgn}(\theta^Tx + \theta_0) = 
\begin{cases}
+1 &\text{if } \theta^Tx + \theta_0 > 0\\
-1 &\text{otherwise} 
\end{cases}$$

The parameters $\theta \in \mathbb{R}^{d}, \theta_{0} \in \mathbb{R}$ describe a particular hypothesis.

A note on dimensions:

$$\begin{align*}
x^{(i)} &: d \times 1 \text{ (column) vector} \\
\theta &: d \times 1 \text{ (column) vector} \\
\theta^{T} &: 1 \times d \text{ (row) vector}
\end{align*}$$

So, $\theta^{T}x + \theta_0 \in \mathbb{R}$. In other words, $\theta^{T}x + \theta_0$ is a scalar. 

## The random linear classifier algorithm

This is the **random linear classifier in pseudocode** (from the [notes](https://openlearninglibrary.mit.edu/courses/course-v1:MITx+6.036+1T2019/courseware/Week1/linear_classifiers/7)):

![](https://i.ibb.co/nwzfDsN/images-linear-classifiers-learning-linear-classifiers-codebox-1-crop.png)

The function randomly generates $(\theta, \theta_{0})$ pairs $k$ times, and chooses the pair that gives the least error.

The arguments of the function are the training data set $\cal D$, the **hyperparameter** $k$, and the degree of $x$ --- $d$. A hyperparameter is a parameter of the machine learning algorithm (as opposed to being a parameter of the hypothesis).

For this algorithm, as $k$ gets larger, the error $\epsilon$ decreases, because repeating the algorithm more times makes it more likely you stumble across a good hypothesis. However, you also start hitting diminishing returns. The relationship might look something like this:

![Plot of hyperparameter vs k](https://i.ibb.co/mJcywgb/k-v-error.png)



