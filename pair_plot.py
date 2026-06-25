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


def get_values_by_house(rows, feature, house):
    vals = []
    for row in rows:
        if row["Hogwarts House"] != house:
            continue
        v = row[feature]
        if v != "":
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return vals


def ft_min(values):
    m = values[0]
    for v in values:
        if v < m:
            m = v
    return m


def ft_max(values):
    m = values[0]
    for v in values:
        if v > m:
            m = v
    return m


def make_histogram(ax, rows, feature, houses, colors):
    """Diagonal cells: overlapping histograms per house."""
    all_vals = []
    for house in houses:
        vals = get_values_by_house(rows, feature, house)
        all_vals.extend(vals)

    if not all_vals:
        return

    lo = ft_min(all_vals)
    hi = ft_max(all_vals)
    bins = 20
    width = (hi - lo) / bins if hi != lo else 1

    for house in houses:
        vals = get_values_by_house(rows, feature, house)
        if not vals:
            continue
        counts = [0] * bins
        for v in vals:
            idx = int((v - lo) / width)
            if idx >= bins:
                idx = bins - 1
            counts[idx] += 1
        bin_edges = [lo + i * width for i in range(bins)]
        ax.bar(bin_edges, counts, width=width * 0.9,
               color=colors[house], alpha=0.5, align="edge")


def make_scatter(ax, rows, feat_x, feat_y, houses, colors):
    """Off-diagonal cells: scatter plot per house."""
    for house in houses:
        x_vals = []
        y_vals = []
        for row in rows:
            if row["Hogwarts House"] != house:
                continue
            vx = row[feat_x]
            vy = row[feat_y]
            if vx != "" and vy != "":
                try:
                    x_vals.append(float(vx))
                    y_vals.append(float(vy))
                except ValueError:
                    pass
        ax.scatter(x_vals, y_vals, color=colors[house], alpha=0.4,
                   s=2, linewidths=0)


def main():
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset_train.csv")
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

    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(n * 1.5, n * 1.5),
                             constrained_layout=True)
    fig.suptitle("Pair Plot - Hogwarts Dataset", fontsize=14, y=1.01)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                make_histogram(ax, rows, features[i], houses, colors)
            else:
                make_scatter(ax, rows, features[j], features[i], houses, colors)

            # Feature labels on the outer edges only
            if j == 0:
                ax.set_ylabel(features[i], fontsize=5, rotation=45,
                              ha="right", va="center")
            if i == n - 1:
                ax.set_xlabel(features[j], fontsize=5, rotation=45,
                              ha="right", va="top")

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors[h], markersize=6, label=h)
        for h in houses
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8,
               bbox_to_anchor=(1.08, 1.0))

    plt.show()


if __name__ == "__main__":
    main()
