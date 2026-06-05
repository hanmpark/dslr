#!/usr/bin/env python3
"""
logreg_predict.py - Predict Hogwarts houses using trained logistic regression.
Usage: python3 logreg_predict.py dataset_test.csv weights.json
Outputs: houses.csv
"""

import sys
import csv
import math
import json


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 logreg_predict.py <dataset_test.csv> <weights.json>")
        sys.exit(1)

    # Load test data
    with open(sys.argv[1]) as f:
        rows = list(csv.DictReader(f))

    # Load model
    with open(sys.argv[2]) as f:
        model = json.load(f)

    features = model['features']
    houses = model['houses']
    means = model['means']
    stds = model['stds']
    weights = model['weights']

    # Predict
    predictions = []
    for row in rows:
        # Normalize features
        sample = []
        for feat in features:
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

        # Pick house with highest sigmoid output
        best_h, best_p = None, -1.0
        for house in houses:
            w = weights[house]
            z = w[0] + sum(w[j + 1] * sample[j] for j in range(len(features)))
            p = sigmoid(z)
            if p > best_p:
                best_p = p
                best_h = house
        predictions.append(best_h)

    # Write output
    with open('houses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Hogwarts House'])
        for i, house in enumerate(predictions):
            writer.writerow([i, house])

    print(f"Predictions saved to houses.csv ({len(predictions)} rows)")


if __name__ == '__main__':
    main()
