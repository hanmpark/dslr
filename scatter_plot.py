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
    """Compute Pearson correlation coefficient manually."""
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
    """Get values for two features, only keeping rows where both are present."""
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset_train.csv")
        sys.exit(1)

    headers, rows = read_csv(sys.argv[1])
    features = get_numeric_features(headers)
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    colors = {
        "Gryffindor": "#e74c3c",
        "Hufflepuff": "#f1c40f",
        "Ravenclaw": "#3498db",
        "Slytherin": "#2ecc71"
    }

    # Compute correlation for all feature pairs
    all_pairs = []
    for i in range(ft_count(features)):
        for j in range(i + 1, ft_count(features)):
            f1 = features[i]
            f2 = features[j]
            x_vals, y_vals = get_paired_values(rows, f1, f2)
            corr = ft_correlation(x_vals, y_vals)
            all_pairs.append((abs(corr), corr, f1, f2))

    # Sort by absolute correlation descending
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    # Print full ranking
    print("Correlation ranking of all feature pairs:")
    print(f"{'Rank':<6} {'|r|':<10} {'r':<10} {'Feature 1':<30} {'Feature 2'}")
    print("-" * 90)
    for rank, (abs_corr, corr, f1, f2) in enumerate(all_pairs, 1):
        print(f"{rank:<6} {abs_corr:<10.4f} {corr:<10.4f} {f1:<30} {f2}")

    # Best pair
    _, _, best_f1, best_f2 = all_pairs[0]
    best_corr = all_pairs[0][1]
    print(f"\nThe two most similar features are:")
    print(f"  '{best_f1}' and '{best_f2}' (r = {best_corr:.4f})")

    # Plot scatter plot for the most similar pair
    fig, ax = plt.subplots(figsize=(10, 8))

    for house in houses:
        x_vals = []
        y_vals = []
        for row in rows:
            if row["Hogwarts House"] != house:
                continue
            v1 = row[best_f1]
            v2 = row[best_f2]
            if v1 != "" and v2 != "":
                try:
                    x_vals.append(float(v1))
                    y_vals.append(float(v2))
                except ValueError:
                    pass
        ax.scatter(x_vals, y_vals, alpha=0.6, label=house,
                   color=colors[house], edgecolors="black", linewidth=0.3, s=30)

    ax.set_xlabel(best_f1, fontsize=12)
    ax.set_ylabel(best_f2, fontsize=12)
    ax.set_title(f"Scatter Plot: {best_f1} vs {best_f2}\n(r = {best_corr:.4f})",
                 fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
