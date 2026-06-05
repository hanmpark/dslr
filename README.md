# DSLR — Data Science × Logistic Regression

A Hogwarts Sorting Hat classifier built from scratch using logistic regression (one-vs-all).  
Given a student's course grades, the model predicts which Hogwarts house they belong to.

---

## Dataset

| File | Rows | Purpose |
|---|---|---|
| `dataset_train.csv` | 1600 | Training data — includes the `Hogwarts House` column |
| `dataset_test.csv` | 400 | Test data — `Hogwarts House` column is empty (to predict) |

**4 houses:** Gryffindor, Hufflepuff, Ravenclaw, Slytherin  
**13 numerical features:** course grades (Arithmancy, Astronomy, Herbology, etc.)  
**Ignored columns:** Index, First Name, Last Name, Birthday, Best Hand  
**Note:** Some values are missing (NaN) — every program handles this.

---

## Programs

### 1. `describe.py`

**Purpose:** Display statistics for every numerical feature, similar to pandas `describe()`.

**Usage:**
```
python describe.py dataset_train.csv
```

**What it does, step by step:**

1. **`read_csv`** — Opens the CSV with Python's built-in `csv` module. For each column, it keeps only the rows that contain a valid number (skips empty cells and non-numeric values). Returns a dictionary `{ feature_name: [list of floats] }`.

2. **`compute_stats`** — For each feature, computes 8 statistics **entirely manually** (no numpy, no pandas):
   - **Count** — iterates through the list and increments a counter.
   - **Mean** — sums all values, divides by count.
   - **Std** — computes `sqrt( Σ(x - mean)² / n )` using a loop.
   - **Min / Max** — obtained from `sorted()` (first and last element).
   - **25%, 50%, 75%** — sorts the values and picks the element at index `int(0.25 * n)`, `int(0.5 * n)`, `int(0.75 * n)`.

3. **`print_stats`** — Formats the results as an aligned table: feature names as columns, stat names as rows.

**Key constraint:** `count()`, `mean()`, `std()`, `min()`, `max()`, `percentile()` and similar built-in functions are forbidden. Everything is implemented with loops and basic arithmetic.

---

### 2. `histogram.py`

**Purpose:** Visualize score distributions per house for every course. Answers: *"Which course has the most homogeneous distribution across all four houses?"*

**Usage:**
```
python histogram.py dataset_train.csv
```

**What it does, step by step:**

1. **Data grouping** — Reads all rows and organizes values into a nested dictionary: `data_by_house[house][feature] = [list of floats]`.

2. **Plotting** — Creates a grid of subplots (3 columns, as many rows as needed). For each course, draws 4 overlapping histograms (one per house, each with `alpha=0.5` transparency so they are all visible at the same time). Each house has a fixed color (red, yellow, blue, green).

3. **`find_most_homogeneous`** — After showing the plot, this function computes a score for each course:
   - Calculates the mean and standard deviation for each house on that course.
   - Computes `score = std(house_means) / mean(house_stds)`.
   - A **low score** means the four house averages are close together relative to their internal spread → distributions are similar → the course is homogeneous.
   - Returns the feature with the lowest score and prints it to the terminal.

---

### 3. `scatter_plot.py`

**Purpose:** Find the two most correlated features and plot them. Answers: *"What are the two most similar features?"*

**Usage:**
```
python scatter_plot.py dataset_train.csv
```

**What it does, step by step:**

1. **`ft_correlation`** — Computes the **Pearson correlation coefficient** manually for a pair of features:
   ```
   r = Σ(x - mean_x)(y - mean_y) / sqrt( Σ(x-mean_x)² × Σ(y-mean_y)² )
   ```
   `r` ranges from -1 to +1. A value close to ±1 means the two features evolve together.

2. **Pair ranking** — Iterates over every possible pair of features (78 pairs for 13 features), computes `|r|` for each, sorts them in descending order, and prints the full ranking to the terminal.

3. **Plotting** — Takes the top pair and draws a scatter plot where each dot is a student, colored by house.

---

### 4. `pair_plot.py`

**Purpose:** Display a full N×N grid of charts for all features. Helps visually identify which features separate the houses well — useful before choosing features for logistic regression.

**Usage:**
```
python pair_plot.py dataset_train.csv
```

**What it does, step by step:**

1. **Grid construction** — Creates an N×N matplotlib grid (N = 13 features). Each cell `(i, j)` shows the relationship between `feature[i]` (Y-axis) and `feature[j]` (X-axis).

2. **Diagonal cells (`i == j`)** — Shows a **histogram** for that feature (`make_histogram`). Values are manually bucketed into 20 bins; 4 overlapping bars are drawn per house with transparency.

3. **Off-diagonal cells (`i ≠ j`)** — Shows a **scatter plot** (`make_scatter`). Each student is a tiny dot (size=2) colored by house. Where houses form distinct clusters, the feature pair is useful for classification.

4. **`print_feature_analysis`** — After showing the plot, prints a feature selection analysis to the terminal:

   - **F-ratio per feature:**  
     Measures how well a course separates the 4 houses.  
     ```
     F = between-group variance / within-group variance
     ```
     - Between-group: how far apart are the house averages from the global average?  
     - Within-group: how spread out are scores inside each house?  
     - High F → houses score very differently → feature is useful.

   - **Redundant pairs (Pearson r ≥ 0.85):**  
     If two features are highly correlated, one is redundant. The one with the lower F-ratio is marked for removal.

   - **Final recommendation:** Prints a "Keep / Drop" list for logistic regression.

---

### 5. `logreg_train.py`

**Purpose:** Train a logistic regression model on `dataset_train.csv` and save the weights to `weights.json`.

**Usage:**
```
python logreg_train.py dataset_train.csv
```

**What it does, step by step:**

#### Preprocessing (`load_and_preprocess`)

1. **Missing values** — For each feature, computes the mean of all non-empty rows. Any missing value for a student is replaced by that mean.

2. **Z-score normalization** — Scales every feature to have mean ≈ 0 and std ≈ 1:
   ```
   x_normalized = (x - mean) / std
   ```
   This is critical: without it, features with large numerical ranges would dominate the gradient updates. The mean and std computed here are saved and reused during prediction.

#### One-vs-all strategy

Instead of solving 4 classes at once, the model trains **4 independent binary classifiers**:
- Gryffindor vs everyone else
- Hufflepuff vs everyone else
- Ravenclaw vs everyone else
- Slytherin vs everyone else

For each classifier, the target label is `1` for students of that house, `0` for all others.

#### Binary training (`train_binary`)

Each classifier learns a weight vector `w` of size 13 (one per feature) plus a bias term `w[0]`.

**Sigmoid function:**
```
sigmoid(z) = 1 / (1 + e^(-z))
```
Converts any real number into a probability between 0 and 1. Implemented with two branches to avoid numerical overflow.

**Gradient descent loop (1000 iterations):**

For each iteration:
1. For each student, compute the prediction:
   ```
   z = w[0] + w[1]*x1 + w[2]*x2 + ... + w[12]*x12
   h = sigmoid(z)
   ```
2. Compute the error: `error = h - true_label`
3. Accumulate the gradient:
   ```
   grad[0] += error            (bias term)
   grad[j] += error * x[j]    (for each feature j)
   ```
4. Update weights:
   ```
   w[j] = w[j] - learning_rate * grad[j] / m
   ```
   (learning rate = 0.5, m = number of students)

#### Accuracy check

After training all 4 classifiers, the program runs them on the training set. For each student, it picks the house with the highest sigmoid output and checks if it matches the real label. Prints the training accuracy.

#### Saving the model (`weights.json`)

Saves everything needed to make predictions later:
```json
{
  "features": [...],
  "houses":   [...],
  "means":    { "Astronomy": 39.8, ... },
  "stds":     { "Astronomy": 520.1, ... },
  "weights":  {
    "Gryffindor": [bias, w1, w2, ...],
    ...
  }
}
```

---

### 6. `logreg_predict.py`

**Purpose:** Use the trained model from `weights.json` to predict the house of every student in `dataset_test.csv`. Outputs `houses.csv`.

**Usage:**
```
python logreg_predict.py dataset_test.csv weights.json
```

**What it does, step by step:**

1. **Load inputs** — Reads `dataset_test.csv` (400 students) and `weights.json` (the trained model).

2. **Normalize** — Applies **the exact same normalization** as during training, using the means and stds saved in `weights.json`:
   ```
   x_normalized = (x - training_mean) / training_std
   ```
   This is essential: the weights were learned on normalized data, so the input at prediction time must use the same scale. Missing values are replaced by the training mean.

3. **Predict** — For each student, runs all 4 classifiers:
   ```
   z = w[0] + w[1]*x1 + ... + w[12]*x12
   p = sigmoid(z)
   ```
   The house with the **highest probability** wins.

4. **Write output** — Saves `houses.csv`:
   ```
   Index,Hogwarts House
   0,Ravenclaw
   1,Gryffindor
   ...
   ```

---

## How the programs connect

```
dataset_train.csv
      │
      ├──► describe.py        → prints statistics to terminal
      ├──► histogram.py       → shows score distributions per house
      ├──► scatter_plot.py    → shows the two most correlated features
      ├──► pair_plot.py       → shows full feature grid + keep/drop advice
      └──► logreg_train.py    → trains the model → weights.json
                                                         │
dataset_test.csv ──────────────────────────────────────►│
                                                    logreg_predict.py
                                                         │
                                                         └──► houses.csv
```

---

## Project structure

```
.
├── dataset_train.csv
├── dataset_test.csv
├── describe.py
├── histogram.py
├── scatter_plot.py
├── pair_plot.py
├── logreg_train.py
├── logreg_predict.py
├── weights.json          ← generated by logreg_train.py
└── houses.csv            ← generated by logreg_predict.py
```
