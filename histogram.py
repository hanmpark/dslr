import csv
import sys
import matplotlib
matplotlib.use('TkAgg')
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python histogram.py dataset_train.csv")
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

    # Group data by house and feature
    data_by_house = {house: {f: [] for f in features} for house in houses}
    for row in rows:
        house = row["Hogwarts House"]
        if house not in houses:
            continue
        for f in features:
            val = row[f]
            if val is not None and val != "":
                try:
                    data_by_house[house][f].append(float(val))
                except ValueError:
                    pass

    # Plot histograms for all courses
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    fig.suptitle("Score Distribution by House for Each Course", fontsize=14)

    for i, feature in enumerate(features):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx][col_idx] if n_rows > 1 else axes[col_idx]

        for house in houses:
            values = data_by_house[house][feature]
            if len(values) > 0:
                ax.hist(values, bins=20, alpha=0.5, label=house,
                        color=colors[house], edgecolor="black", linewidth=0.3)

        ax.set_title(feature, fontsize=9, pad=6)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    # Hide unused subplots
    for i in range(n_features, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx][col_idx] if n_rows > 1 else axes[col_idx]
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    best_feature = find_most_homogeneous(features, houses, data_by_house)
    print(f"\nThe course with the most homogeneous score distribution across"
          f" all four houses is '{best_feature}'.")


def ft_mean(values):
    return sum(values) / len(values)


def ft_std(values):
    m = ft_mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def find_most_homogeneous(features, houses, data_by_house):
    best_feature = None
    best_score = float("inf")

    for feature in features:
        house_data = [data_by_house[h][feature] for h in houses
                      if len(data_by_house[h][feature]) > 1]
        if len(house_data) < 2:
            continue

        house_means = [ft_mean(d) for d in house_data]
        house_stds = [ft_std(d) for d in house_data]
        avg_within_std = ft_mean(house_stds)

        if avg_within_std == 0:
            continue

        # Ratio: spread of means relative to average within-house spread
        # A low ratio means all houses have similar distributions
        score = ft_std(house_means) / avg_within_std
        if score < best_score:
            best_score = score
            best_feature = feature

    return best_feature


if __name__ == "__main__":
    main()
