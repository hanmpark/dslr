# MATH.md — Mathematical Foundations of DSLR

## 1. Descriptive Statistics

### Count
The number of non-missing values in a column.

### Mean (Arithmetic Average)
The sum of all values divided by the count:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

### Variance and Standard Deviation
Variance measures the average squared deviation from the mean. We use the **sample variance** (dividing by `n - 1`, known as Bessel's correction) to get an unbiased estimate:

$$s^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

Standard deviation is the square root of variance — it brings the unit back to the original scale:

$$s = \sqrt{s^2}$$

### Percentiles (Quantiles)
The **p-th percentile** is the value below which p% of the data falls. To compute it:

1. Sort the values in ascending order.
2. Compute the rank index: `i = p / 100 * (n - 1)` (zero-based, continuous).
3. If `i` is not an integer, **linearly interpolate** between the two adjacent values:

$$Q_p = x_{\lfloor i \rfloor} + (i - \lfloor i \rfloor) \cdot (x_{\lceil i \rceil} - x_{\lfloor i \rfloor})$$

This matches pandas' default `linear` interpolation method. The three key percentiles used are:
- **Q1 (25%)** — lower quartile
- **Q2 (50%)** — median
- **Q3 (75%)** — upper quartile

---

## 2. Pearson Correlation Coefficient

Used in `scatter_plot.py` to find the two most similar features. It measures the **linear relationship** between two variables, ranging from -1 (perfect negative) to +1 (perfect positive):

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

We look for the pair with `|r|` closest to 1, meaning their distributions are the most alike.

---

## 3. Feature Standardization (Z-score Normalization)

Before training, each feature is rescaled so that it has **mean 0** and **standard deviation 1**:

$$x'_i = \frac{x_i - \mu}{\sigma}$$

Where:
- $\mu$ is the **training set mean** for that feature
- $\sigma$ is the **training set standard deviation** for that feature

**Why it matters:** Gradient descent converges much faster when all features are on the same scale. Without it, features with large ranges dominate the gradient updates.

**Important:** The same $\mu$ and $\sigma$ computed on the training set must be saved and reused to normalize the test set. Never recompute normalization statistics from the test data.

---

## 4. Logistic Regression

### 4.1 The Problem: Binary Classification

Logistic regression predicts the probability that a sample belongs to class 1 (vs. class 0). Unlike linear regression, the output is bounded between 0 and 1.

### 4.2 The Sigmoid Function

The **sigmoid** (logistic) function maps any real number to `(0, 1)`:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Key properties:
- $\sigma(0) = 0.5$
- As $z \to +\infty$, $\sigma(z) \to 1$
- As $z \to -\infty$, $\sigma(z) \to 0$

The model computes $z = \theta^T x$ (a linear combination of features and weights), then squashes it through the sigmoid to get a probability:

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

**Numerical stability note:** Clip $z$ to $[-500, 500]$ before computing `exp(-z)` to avoid floating-point overflow.

### 4.3 The Hypothesis

Given a feature vector $x \in \mathbb{R}^{d+1}$ (with a bias term $x_0 = 1$) and a weight vector $\theta \in \mathbb{R}^{d+1}$:

$$h_\theta(x) = \sigma\left(\sum_{j=0}^{d} \theta_j x_j\right)$$

The output is interpreted as: **"the probability that sample $x$ belongs to class 1."**

### 4.4 The Cost Function (Log-Loss / Binary Cross-Entropy)

We can't use mean squared error for classification — the resulting cost surface has many local minima. Instead, we use the **log-loss**:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

Where:
- $m$ = number of training samples
- $y^{(i)} \in \{0, 1\}$ = true label for sample $i$
- $h_\theta(x^{(i)})$ = predicted probability for sample $i$

**Intuition:**
- When $y = 1$: the cost is $-\log(h)$. If $h \to 1$ (correct), cost $\to 0$. If $h \to 0$ (wrong), cost $\to \infty$.
- When $y = 0$: the cost is $-\log(1 - h)$. If $h \to 0$ (correct), cost $\to 0$. If $h \to 1$ (wrong), cost $\to \infty$.

### 4.5 Gradient of the Cost Function

The partial derivative of $J(\theta)$ with respect to each weight $\theta_j$ is:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$$

In matrix form (more efficient with numpy):

$$\nabla_\theta J = \frac{1}{m} X^T (h - y)$$

Where:
- $X \in \mathbb{R}^{m \times (d+1)}$ = feature matrix (with bias column)
- $h \in \mathbb{R}^m$ = vector of predicted probabilities
- $y \in \mathbb{R}^m$ = vector of true labels

### 4.6 Gradient Descent

Gradient descent iteratively updates the weights in the direction that decreases the cost:

$$\theta := \theta - \alpha \cdot \nabla_\theta J(\theta)$$

Where $\alpha$ is the **learning rate** — a hyperparameter that controls the step size.

- Too large: overshoots the minimum, diverges.
- Too small: converges too slowly.
- Typical range with standardized features: `0.01 – 1.0`.

The loop runs for a fixed number of iterations (e.g., 500–1000).

---

## 5. One-vs-All (OvR) Strategy

Since there are **4 Hogwarts houses**, we train **4 separate binary classifiers**, one per house:

| Classifier | Positive class | Negative class |
|------------|---------------|----------------|
| Classifier 1 | Gryffindor (1) | All others (0) |
| Classifier 2 | Hufflepuff (1) | All others (0) |
| Classifier 3 | Ravenclaw (1)  | All others (0) |
| Classifier 4 | Slytherin (1)  | All others (0) |

Each classifier $k$ learns its own weight vector $\theta^{(k)}$.

### Prediction

At prediction time, run all 4 classifiers and assign the house with the **highest probability**:

$$\hat{y} = \arg\max_{k} \; h_{\theta^{(k)}}(x) = \arg\max_{k} \; \sigma\left(\theta^{(k)T} x\right)$$

---

## 6. Summary of the Full Pipeline

```
Raw CSV
  ↓
Drop non-numerical columns (Name, Birthday, Best Hand)
  ↓
Fill NaN with column mean (computed on training set only)
  ↓
Z-score normalize each feature  →  save (μ, σ) per feature
  ↓
Add bias column (x₀ = 1)
  ↓
For each house k:
    set y = 1 where house == k, else 0
    initialize θ⁽ᵏ⁾ = 0
    run gradient descent for N iterations
    save θ⁽ᵏ⁾
  ↓
Save { θ⁽ᵏ⁾, μ, σ } → weights.json
  ↓
At prediction:
    normalize test data with saved μ, σ
    compute h⁽ᵏ⁾(x) for all 4 classifiers
    assign house = argmax over k
  ↓
Output houses.csv
```
