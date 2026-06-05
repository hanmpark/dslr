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
    for row in rows:
        for feat in FEATURES:
            v = row.get(feat, '')
            if v != '':
                try:
                    raw[feat].append(float(v))
                except ValueError:
                    pass

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

    return X, y, means, stds


def train_binary(X, binary_y, lr=0.5, epochs=1000):
    """Train one binary logistic regression classifier. Returns [bias, w1, w2, ...]."""
    m = len(X)
    n = len(X[0])
    w = [0.0] * (n + 1)  # w[0] = bias

    for _ in range(epochs):
        grad = [0.0] * (n + 1)
        for i in range(m):
            z = w[0] + sum(w[j + 1] * X[i][j] for j in range(n))
            h = sigmoid(z)
            err = h - binary_y[i]
            grad[0] += err
            for j in range(n):
                grad[j + 1] += err * X[i][j]
        for j in range(n + 1):
            w[j] -= lr * grad[j] / m

    return w


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 logreg_train.py <dataset_train.csv>")
        sys.exit(1)

    X, y, means, stds = load_and_preprocess(sys.argv[1])
    print(f"Loaded {len(X)} samples, {len(FEATURES)} features.")

    weights = {}
    for house in HOUSES:
        binary_y = [1.0 if yi == house else 0.0 for yi in y]
        pos = int(sum(binary_y))
        print(f"  Training {house} ({pos} positive)...", end=" ", flush=True)
        w = train_binary(X, binary_y, lr=0.5, epochs=1000)
        weights[house] = w
        print("done.")

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
