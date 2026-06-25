import csv
import sys
import matplotlib.pyplot as plt


def read_csv(filename):
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = []
        for row in reader:
            rows.append(row)
    return headers, rows


def get_numeric_features(headers):
    non_numeric = ["Index", "Hogwarts House", "First Name", "Last Name",
                   "Birthday", "Best Hand"]
    return [h for h in headers if h not in non_numeric]


def ft_count(values):
    count = 0
    for _ in values:
        count += 1
    return count


def ft_sum(values):
    total = 0
    for value in values:
        total += value
    return total


def ft_correlation(x_vals, y_vals):
    n = ft_count(x_vals)
    if n == 0:
        return 0

    mean_x = ft_sum(x_vals) / n
    mean_y = ft_sum(y_vals) / n

    num = 0
    den_x = 0
    den_y = 0
    for i in range(n):
        dx = x_vals[i] - mean_x
        dy = y_vals[i] - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy

    if den_x == 0 or den_y == 0:
        return 0
    return num / (den_x ** 0.5 * den_y ** 0.5)


def get_paired_values(rows, f1, f2):
    x_vals = []
    y_vals = []
    for row in rows:
        v1 = row[f1]
        v2 = row[f2]
        if v1 != "" and v2 != "":
            try:
                x_vals.append(float(v1))
                y_vals.append(float(v2))
            except ValueError:
                pass
    return x_vals, y_vals


def find_most_correlated_pair(rows, features):
    best_pair = None
    best_corr = 0
    best_abs_corr = -1

    for i in range(ft_count(features)):
        for j in range(i + 1, ft_count(features)):
            x_vals, y_vals = get_paired_values(rows, features[i], features[j])
            corr = ft_correlation(x_vals, y_vals)
            abs_corr = abs(corr)
            if abs_corr > best_abs_corr:
                best_abs_corr = abs_corr
                best_corr = corr
                best_pair = (features[i], features[j])

    return best_pair[0], best_pair[1], best_corr


def main():
    if len(sys.argv) not in (2, 4):
        print("Usage: python scatter_plot.py dataset_train.csv [feature_x feature_y]")
        sys.exit(1)

    headers, rows = read_csv(sys.argv[1])
    features = get_numeric_features(headers)
    if len(sys.argv) == 4:
        feature_x = sys.argv[2]
        feature_y = sys.argv[3]
        if feature_x not in features or feature_y not in features:
            print("Error: selected features must be numeric dataset features")
            sys.exit(1)
        x_vals, y_vals = get_paired_values(rows, feature_x, feature_y)
        corr = ft_correlation(x_vals, y_vals)
    else:
        feature_x, feature_y, corr = find_most_correlated_pair(rows, features)

    print(f"Displaying: {feature_x} vs {feature_y} (r = {corr:.4f})")

    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    colors = {
        "Gryffindor": "#e74c3c",
        "Hufflepuff": "#f1c40f",
        "Ravenclaw": "#3498db",
        "Slytherin": "#2ecc71"
    }

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    for house in houses:
        x_vals = []
        y_vals = []
        for row in rows:
            if row["Hogwarts House"] != house:
                continue
            v1 = row[feature_x]
            v2 = row[feature_y]
            if v1 != "" and v2 != "":
                try:
                    x_vals.append(float(v1))
                    y_vals.append(float(v2))
                except ValueError:
                    pass
        ax.scatter(x_vals, y_vals, alpha=0.6, label=house,
                   color=colors[house], edgecolors="black", linewidth=0.3, s=30)

    ax.set_xlabel(feature_x, fontsize=12)
    ax.set_ylabel(feature_y, fontsize=12)
    ax.set_title(f"Scatter Plot: {feature_x} vs {feature_y}\n(r = {corr:.4f})",
                 fontsize=14)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
