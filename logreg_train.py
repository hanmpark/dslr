#!/usr/bin/env python3
"""
logreg_train.py - Train a one-vs-all logistic regression classifier.
Usage: python3 logreg_train.py dataset_train.csv
Outputs: weights.json (model weights + normalization params)
"""

import sys
import csv
import math
import json


FEATURES = [
    'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
    'Divination', 'Muggle Studies', 'Ancient Runes',
    'History of Magic', 'Transfiguration', 'Potions',
    'Care of Magical Creatures', 'Charms', 'Flying'
]
HOUSES = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
LOG_INTERVAL = 100
EPSILON = 1e-15


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def load_and_preprocess(filepath):
    """Load CSV, extract features, compute normalization stats, return X, y, means, stds."""
    with open(filepath) as f:
        rows = list(csv.DictReader(f))

    # Compute means (for NaN filling)
    raw = {feat: [] for feat in FEATURES}
    missing = {feat: 0 for feat in FEATURES}
    invalid = {feat: 0 for feat in FEATURES}
    for row in rows:
        for feat in FEATURES:
            v = row.get(feat, '')
            if v == '':
                missing[feat] += 1
                continue
            try:
                raw[feat].append(float(v))
            except ValueError:
                invalid[feat] += 1

    means = {}
    stds = {}
    for feat in FEATURES:
        vals = raw[feat]
        m = sum(vals) / len(vals) if vals else 0.0
        means[feat] = m
        if len(vals) > 1:
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            stds[feat] = math.sqrt(var) if var > 0 else 1.0
        else:
            stds[feat] = 1.0

    # Build X (z-score normalized) and y
    X = []
    y = []
    for row in rows:
        sample = []
        for feat in FEATURES:
            v = row.get(feat, '')
            if v == '':
                val = means[feat]
            else:
                try:
                    val = float(v)
                except ValueError:
                    val = means[feat]
            val = (val - means[feat]) / stds[feat]
            sample.append(val)
        X.append(sample)
        y.append(row.get('Hogwarts House', ''))

    return X, y, means, stds, missing, invalid


def print_preprocess_summary(means, stds, missing, invalid):
    print("\nPreprocessing summary")
    print(f"{'Feature':<35} {'mean':>12} {'std':>12} {'missing':>9} {'invalid':>9}")
    print("-" * 82)
    for feat in FEATURES:
        print(f"{feat:<35} {means[feat]:>12.4f} {stds[feat]:>12.4f}"
              f" {missing[feat]:>9} {invalid[feat]:>9}")
    print("\nMissing or invalid values are replaced with the training mean,")
    print("then each feature is normalized with z-score: (value - mean) / std.\n")


def train_binary(X, binary_y, lr=0.5, epochs=1000):
    """Train one binary logistic regression classifier. Returns [bias, w1, w2, ...]."""
    m = len(X)
    n = len(X[0])
    w = [0.0] * (n + 1)  # w[0] = bias

    for epoch in range(epochs):
        grad = [0.0] * (n + 1)
        total_loss = 0.0
        correct = 0
        for i in range(m):
            z = w[0] + sum(w[j + 1] * X[i][j] for j in range(n))
            h = sigmoid(z)
            err = h - binary_y[i]
            safe_h = min(max(h, EPSILON), 1.0 - EPSILON)
            total_loss += -(binary_y[i] * math.log(safe_h)
                            + (1.0 - binary_y[i]) * math.log(1.0 - safe_h))
            if (h >= 0.5) == (binary_y[i] == 1.0):
                correct += 1
            grad[0] += err
            for j in range(n):
                grad[j + 1] += err * X[i][j]
        for j in range(n + 1):
            w[j] -= lr * grad[j] / m

        if epoch == 0 or (epoch + 1) % LOG_INTERVAL == 0 or epoch == epochs - 1:
            avg_loss = total_loss / m
            accuracy = 100.0 * correct / m
            print(f"    epoch {epoch + 1:4d}/{epochs}"
                  f" - loss: {avg_loss:.6f}"
                  f" - binary acc: {accuracy:6.2f}%")

    print(f"    learned bias: {w[0]:.6f}")
    return w


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 logreg_train.py <dataset_train.csv>")
        sys.exit(1)

    X, y, means, stds, missing, invalid = load_and_preprocess(sys.argv[1])
    print(f"Loaded {len(X)} samples, {len(FEATURES)} features.")
    print_preprocess_summary(means, stds, missing, invalid)

    weights = {}
    for house in HOUSES:
        binary_y = [1.0 if yi == house else 0.0 for yi in y]
        pos = int(sum(binary_y))
        neg = len(binary_y) - pos
        print(f"Training {house}: {pos} positive / {neg} negative")
        w = train_binary(X, binary_y, lr=0.5, epochs=1000)
        weights[house] = w
        print(f"Finished {house}.\n")

    # Training accuracy
    correct = 0
    for i in range(len(X)):
        best_h, best_p = None, -1.0
        for house in HOUSES:
            w = weights[house]
            z = w[0] + sum(w[j + 1] * X[i][j] for j in range(len(FEATURES)))
            p = sigmoid(z)
            if p > best_p:
                best_p = p
                best_h = house
        if best_h == y[i]:
            correct += 1
    print(f"Training accuracy: {correct}/{len(X)} ({100 * correct / len(X):.2f}%)")

    # Save model
    model = {
        'features': FEATURES,
        'houses': HOUSES,
        'means': means,
        'stds': stds,
        'weights': weights
    }
    with open('weights.json', 'w') as f:
        json.dump(model, f, indent=2)
    print("Model saved to weights.json")


if __name__ == '__main__':
    main()
