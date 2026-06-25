#!/usr/bin/env python3
import csv
import json
import math
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
TRAIN_PATH = os.path.join(ROOT, "dataset_train.csv")
TEST_PATH = os.path.join(ROOT, "dataset_test.csv")
WEIGHTS_PATH = os.path.join(ROOT, "weights.json")
HOUSES_PATH = os.path.join(ROOT, "houses.csv")

NON_FEATURES = [
    "Index", "Hogwarts House", "First Name", "Last Name", "Birthday",
    "Best Hand"
]
FEATURES = [
    "Astronomy", "Herbology", "Defense Against the Dark Arts",
    "Divination", "Muggle Studies", "Ancient Runes",
    "History of Magic", "Transfiguration", "Potions",
    "Care of Magical Creatures", "Charms", "Flying"
]
HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = {
    "Gryffindor": "#e74c3c",
    "Hufflepuff": "#f1c40f",
    "Ravenclaw": "#3498db",
    "Slytherin": "#2ecc71",
}
EPSILON = 1e-15


def is_float(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def ft_count(values):
    count = 0
    for _ in values:
        count += 1
    return count


def ft_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def ft_mean(values):
    return ft_sum(values) / ft_count(values) if values else 0.0


def ft_std(values):
    if not values:
        return 0.0
    mean = ft_mean(values)
    return (ft_sum((value - mean) ** 2 for value in values)
            / ft_count(values)) ** 0.5


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def numeric_features(headers):
    return [header for header in headers if header not in NON_FEATURES]


def describe_dataset():
    headers, rows = read_csv_rows(TRAIN_PATH)
    features = numeric_features(headers)
    stats = {}
    for feature in features:
        values = []
        for row in rows:
            value = row.get(feature, "")
            if value != "" and is_float(value):
                values.append(float(value))
        if not values:
            continue
        sorted_values = sorted(values)
        count = ft_count(values)
        stats[feature] = {
            "Count": count,
            "Mean": ft_mean(values),
            "Std": ft_std(values),
            "Min": sorted_values[0],
            "25%": sorted_values[int(0.25 * count)],
            "50%": sorted_values[int(0.5 * count)],
            "75%": sorted_values[int(0.75 * count)],
            "Max": sorted_values[-1],
        }
    return {"features": list(stats.keys()), "stats": stats}


def values_by_house(rows, feature):
    data = {house: [] for house in HOUSES}
    for row in rows:
        house = row.get("Hogwarts House", "")
        value = row.get(feature, "")
        if house in data and value != "" and is_float(value):
            data[house].append(float(value))
    return data


def histogram_data(feature, bins=24):
    _, rows = read_csv_rows(TRAIN_PATH)
    grouped = values_by_house(rows, feature)
    all_values = []
    for values in grouped.values():
        all_values.extend(values)
    if not all_values:
        return {"feature": feature, "bins": [], "houses": HOUSES, "colors": COLORS}

    low = min(all_values)
    high = max(all_values)
    width = (high - low) / bins if high != low else 1.0
    bin_edges = [low + i * width for i in range(bins)]
    result = {house: [0] * bins for house in HOUSES}

    for house, values in grouped.items():
        for value in values:
            idx = int((value - low) / width)
            if idx >= bins:
                idx = bins - 1
            result[house][idx] += 1

    return {
        "feature": feature,
        "low": low,
        "high": high,
        "width": width,
        "bins": bin_edges,
        "counts": result,
        "houses": HOUSES,
        "colors": COLORS,
    }


def paired_values(rows, x_feature, y_feature):
    x_vals = []
    y_vals = []
    for row in rows:
        x = row.get(x_feature, "")
        y = row.get(y_feature, "")
        if x != "" and y != "" and is_float(x) and is_float(y):
            x_vals.append(float(x))
            y_vals.append(float(y))
    return x_vals, y_vals


def pearson(x_vals, y_vals):
    n = ft_count(x_vals)
    if n == 0:
        return 0.0
    mean_x = ft_sum(x_vals) / n
    mean_y = ft_sum(y_vals) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for i in range(n):
        dx = x_vals[i] - mean_x
        dy = y_vals[i] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / ((den_x ** 0.5) * (den_y ** 0.5))


def best_correlated_pair(rows, features):
    best = None
    best_corr = 0.0
    best_abs = -1.0
    for i in range(ft_count(features)):
        for j in range(i + 1, ft_count(features)):
            x_vals, y_vals = paired_values(rows, features[i], features[j])
            corr = pearson(x_vals, y_vals)
            if abs(corr) > best_abs:
                best_abs = abs(corr)
                best_corr = corr
                best = (features[i], features[j])
    return best[0], best[1], best_corr


def scatter_data(x_feature=None, y_feature=None):
    headers, rows = read_csv_rows(TRAIN_PATH)
    features = numeric_features(headers)
    auto = not x_feature or not y_feature
    if auto:
        x_feature, y_feature, corr = best_correlated_pair(rows, features)
    else:
        x_vals, y_vals = paired_values(rows, x_feature, y_feature)
        corr = pearson(x_vals, y_vals)

    points = []
    for row in rows:
        x = row.get(x_feature, "")
        y = row.get(y_feature, "")
        house = row.get("Hogwarts House", "")
        if house in HOUSES and x != "" and y != "" and is_float(x) and is_float(y):
            points.append({
                "x": float(x),
                "y": float(y),
                "house": house,
            })

    return {
        "xFeature": x_feature,
        "yFeature": y_feature,
        "correlation": corr,
        "auto": auto,
        "points": points,
        "houses": HOUSES,
        "colors": COLORS,
    }


def pair_data():
    headers, rows = read_csv_rows(TRAIN_PATH)
    features = numeric_features(headers)
    samples = []
    for row in rows:
        sample = {"house": row.get("Hogwarts House", "")}
        for feature in features:
            value = row.get(feature, "")
            sample[feature] = float(value) if value != "" and is_float(value) else None
        samples.append(sample)
    return {
        "features": features,
        "samples": samples,
        "houses": HOUSES,
        "colors": COLORS,
    }


def correlation_matrix_data():
    headers, rows = read_csv_rows(TRAIN_PATH)
    features = numeric_features(headers)
    matrix = []
    pairs = []
    for row_feature in features:
        row = []
        for col_feature in features:
            x_vals, y_vals = paired_values(rows, row_feature, col_feature)
            corr = pearson(x_vals, y_vals)
            row.append(corr)
        matrix.append(row)

    for i in range(ft_count(features)):
        for j in range(i + 1, ft_count(features)):
            x_vals, y_vals = paired_values(rows, features[i], features[j])
            corr = pearson(x_vals, y_vals)
            pairs.append({
                "x": features[i],
                "y": features[j],
                "correlation": corr,
                "absCorrelation": abs(corr),
            })
    pairs.sort(key=lambda pair: pair["absCorrelation"], reverse=True)
    return {
        "features": features,
        "matrix": matrix,
        "topPairs": pairs[:8],
    }


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def load_and_preprocess(path):
    _, rows = read_csv_rows(path)
    raw = {feature: [] for feature in FEATURES}
    missing = {feature: 0 for feature in FEATURES}
    invalid = {feature: 0 for feature in FEATURES}

    for row in rows:
        for feature in FEATURES:
            value = row.get(feature, "")
            if value == "":
                missing[feature] += 1
                continue
            if is_float(value):
                raw[feature].append(float(value))
            else:
                invalid[feature] += 1

    means = {}
    stds = {}
    for feature in FEATURES:
        values = raw[feature]
        mean = ft_mean(values) if values else 0.0
        std = ft_std(values)
        means[feature] = mean
        stds[feature] = std if std > 0 else 1.0

    x_rows = []
    y_rows = []
    for row in rows:
        sample = []
        for feature in FEATURES:
            value = row.get(feature, "")
            if value == "" or not is_float(value):
                numeric = means[feature]
            else:
                numeric = float(value)
            sample.append((numeric - means[feature]) / stds[feature])
        x_rows.append(sample)
        y_rows.append(row.get("Hogwarts House", ""))

    return x_rows, y_rows, means, stds, missing, invalid


def binary_metrics(x_rows, binary_y, weights):
    loss = 0.0
    correct = 0
    for i, sample in enumerate(x_rows):
        z = weights[0]
        for j, value in enumerate(sample):
            z += weights[j + 1] * value
        h = sigmoid(z)
        safe_h = min(max(h, EPSILON), 1.0 - EPSILON)
        loss += -(binary_y[i] * math.log(safe_h)
                  + (1.0 - binary_y[i]) * math.log(1.0 - safe_h))
        if (h >= 0.5) == (binary_y[i] == 1.0):
            correct += 1
    return loss / len(x_rows), 100.0 * correct / len(x_rows)


def multi_accuracy(x_rows, y_rows, all_weights):
    correct = 0
    for i, sample in enumerate(x_rows):
        best_house = None
        best_prob = -1.0
        for house in HOUSES:
            weights = all_weights[house]
            z = weights[0]
            for j, value in enumerate(sample):
                z += weights[j + 1] * value
            prob = sigmoid(z)
            if prob > best_prob:
                best_prob = prob
                best_house = house
        if best_house == y_rows[i]:
            correct += 1
    return correct, 100.0 * correct / len(x_rows)


def predict_probs(sample, model):
    best_house = None
    best_prob = -1.0
    probs = {}
    for house in model["houses"]:
        weights = model["weights"][house]
        z = weights[0]
        for j, value in enumerate(sample):
            z += weights[j + 1] * value
        prob = sigmoid(z)
        probs[house] = prob
        if prob > best_prob:
            best_prob = prob
            best_house = house
    return best_house, best_prob, probs


def sample_from_row(row, model):
    sample = []
    used_values = {}
    for feature in model["features"]:
        value = row.get(feature, "")
        if value == "" or not is_float(value):
            numeric = model["means"][feature]
        else:
            numeric = float(value)
        used_values[feature] = numeric
        sample.append((numeric - model["means"][feature]) / model["stds"][feature])
    return sample, used_values


def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        return None
    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model():
    model = load_model()
    if model is None:
        return {"error": "weights.json is missing. Train the model first."}

    _, rows = read_csv_rows(TRAIN_PATH)
    matrix = {actual: {pred: 0 for pred in HOUSES} for actual in HOUSES}
    mistakes = []
    correct = 0
    total = 0
    for row in rows:
        actual = row.get("Hogwarts House", "")
        if actual not in HOUSES:
            continue
        sample, _ = sample_from_row(row, model)
        predicted, probability, probs = predict_probs(sample, model)
        matrix[actual][predicted] += 1
        total += 1
        if predicted == actual:
            correct += 1
        elif ft_count(mistakes) < 12:
            mistakes.append({
                "index": row.get("Index", ""),
                "actual": actual,
                "predicted": predicted,
                "probability": probability,
                "probabilities": probs,
            })
    accuracy = 100.0 * correct / total if total else 0.0
    return {
        "houses": HOUSES,
        "matrix": matrix,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "mistakes": mistakes,
    }


def train_events():
    x_rows, y_rows, means, stds, missing, invalid = load_and_preprocess(TRAIN_PATH)
    epochs = 1000
    learning_rate = 0.5
    log_interval = 25
    yield {
        "type": "preprocess",
        "samples": len(x_rows),
        "features": FEATURES,
        "missing": missing,
        "invalid": invalid,
        "means": means,
        "stds": stds,
        "epochs": epochs,
        "learningRate": learning_rate,
        "logInterval": log_interval,
    }

    all_weights = {}

    for house in HOUSES:
        binary_y = [1.0 if y == house else 0.0 for y in y_rows]
        positives = int(ft_sum(binary_y))
        weights = [0.0] * (len(FEATURES) + 1)
        yield {
            "type": "house-start",
            "house": house,
            "positive": positives,
            "negative": len(binary_y) - positives,
        }

        for epoch in range(epochs):
            grad = [0.0] * (len(FEATURES) + 1)
            for i, sample in enumerate(x_rows):
                z = weights[0]
                for j, value in enumerate(sample):
                    z += weights[j + 1] * value
                h = sigmoid(z)
                err = h - binary_y[i]
                grad[0] += err
                for j, value in enumerate(sample):
                    grad[j + 1] += err * value

            for j in range(len(weights)):
                weights[j] -= learning_rate * grad[j] / len(x_rows)

            if epoch == 0 or (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
                loss, acc = binary_metrics(x_rows, binary_y, weights)
                yield {
                    "type": "epoch",
                    "house": house,
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": loss,
                    "accuracy": acc,
                }

        all_weights[house] = weights
        yield {
            "type": "house-end",
            "house": house,
            "bias": weights[0],
        }

    correct, accuracy = multi_accuracy(x_rows, y_rows, all_weights)
    model = {
        "features": FEATURES,
        "houses": HOUSES,
        "means": means,
        "stds": stds,
        "weights": all_weights,
    }
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    yield {
        "type": "done",
        "correct": correct,
        "total": len(x_rows),
        "accuracy": accuracy,
        "weightsPath": "weights.json",
    }


def predict():
    model = load_model()
    if model is None:
        return {"error": "weights.json is missing. Train the model first."}

    _, rows = read_csv_rows(TEST_PATH)

    predictions = []
    counts = {house: 0 for house in HOUSES}
    preview = []
    for index, row in enumerate(rows):
        sample, _ = sample_from_row(row, model)
        best_house, best_prob, probs = predict_probs(sample, model)

        counts[best_house] += 1
        predictions.append((index, best_house))
        if len(preview) < 12:
            preview.append({
                "index": index,
                "house": best_house,
                "probability": best_prob,
                "probabilities": probs,
            })

    with open(HOUSES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Hogwarts House"])
        for index, house in predictions:
            writer.writerow([index, house])

    return {
        "rows": len(predictions),
        "output": "houses.csv",
        "counts": counts,
        "preview": preview,
    }


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        if not os.path.exists(path):
            self.send_error(404)
            return
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path

        try:
            if path == "/":
                return self.send_file(os.path.join(WEB_DIR, "index.html"))
            if path in ("/app.js", "/styles.css"):
                return self.send_file(os.path.join(WEB_DIR, path.lstrip("/")))
            if path == "/api/summary":
                headers, rows = read_csv_rows(TRAIN_PATH)
                features = numeric_features(headers)
                house_counts = {house: 0 for house in HOUSES}
                for row in rows:
                    house = row.get("Hogwarts House", "")
                    if house in house_counts:
                        house_counts[house] += 1
                return self.send_json({
                    "trainRows": len(rows),
                    "testRows": len(read_csv_rows(TEST_PATH)[1]),
                    "features": features,
                    "featuresForModel": FEATURES,
                    "houses": HOUSES,
                    "colors": COLORS,
                    "houseCounts": house_counts,
                    "hasWeights": os.path.exists(WEIGHTS_PATH),
                    "hasPredictions": os.path.exists(HOUSES_PATH),
                })
            if path == "/api/describe":
                return self.send_json(describe_dataset())
            if path == "/api/histogram":
                feature = query.get("feature", [FEATURES[0]])[0]
                return self.send_json(histogram_data(feature))
            if path == "/api/scatter":
                x_feature = query.get("x", [None])[0]
                y_feature = query.get("y", [None])[0]
                return self.send_json(scatter_data(x_feature, y_feature))
            if path == "/api/pair":
                return self.send_json(pair_data())
            if path == "/api/correlation":
                return self.send_json(correlation_matrix_data())
            if path == "/api/evaluation":
                return self.send_json(evaluate_model())
            if path == "/api/predict":
                return self.send_json(predict())
            if path == "/api/train-stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                for event in train_events():
                    chunk = "data: " + json.dumps(event) + "\n\n"
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
                return
            self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)


def main():
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"DSLR dashboard running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
