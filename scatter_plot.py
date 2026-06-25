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
    else:
        feature_x = features[0]
        feature_y = features[1]

    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    colors = {
        "Gryffindor": "#e74c3c",
        "Hufflepuff": "#f1c40f",
        "Ravenclaw": "#3498db",
        "Slytherin": "#2ecc71"
    }

    fig, ax = plt.subplots(figsize=(10, 8))

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
    ax.set_title(f"Scatter Plot: {feature_x} vs {feature_y}", fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
