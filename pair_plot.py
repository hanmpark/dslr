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


def ft_mean(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)


def ft_variance(values):
    if len(values) < 2:
        return 0
    mean = ft_mean(values)
    total = 0
    for v in values:
        total += (v - mean) ** 2
    return total / len(values)


def ft_f_ratio(rows, feature, houses):
    """
    Fisher's F-ratio: between-group variance / within-group variance.
    High value means houses are well separated on this feature.
    """
    groups = {}
    all_vals = []
    for house in houses:
        vals = get_values_by_house(rows, feature, house)
        groups[house] = vals
        all_vals.extend(vals)

    if len(all_vals) < 2:
        return 0

    global_mean = ft_mean(all_vals)

    between = 0
    for house in houses:
        g = groups[house]
        if not g:
            continue
        between += len(g) * (ft_mean(g) - global_mean) ** 2
    between /= len(houses)

    within = 0
    for house in houses:
        g = groups[house]
        if not g:
            continue
        within += ft_variance(g)
    within /= len(houses)

    if within == 0:
        return 0
    return between / within


def ft_correlation(x_vals, y_vals):
    """Pearson correlation coefficient."""
    n = len(x_vals)
    if n == 0:
        return 0
    mean_x = ft_mean(x_vals)
    mean_y = ft_mean(y_vals)
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


def print_feature_analysis(rows, features, houses):
    """Compute and print feature selection analysis from data."""
    REDUNDANCY_THRESHOLD = 0.85
    SEPARATION_THRESHOLD = 1.0  # F-ratio below this = poor separation

    # 1. F-ratio per feature
    f_scores = []
    for feat in features:
        score = ft_f_ratio(rows, feat, houses)
        f_scores.append((score, feat))
    f_scores.sort(reverse=True)

    print("\nFeature separation scores (F-ratio: higher = better house separation):")
    print(f"  {'Feature':<35} {'F-ratio':>8}  Verdict")
    print("  " + "-" * 60)
    for score, feat in f_scores:
        if score >= SEPARATION_THRESHOLD:
            verdict = "keep"
        else:
            verdict = "weak — consider dropping"
        print(f"  {feat:<35} {score:>8.2f}  {verdict}")

    # 2. Redundant pairs (high |r|)
    redundant_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            x_vals, y_vals = get_paired_values(rows, features[i], features[j])
            r = ft_correlation(x_vals, y_vals)
            if abs(r) >= REDUNDANCY_THRESHOLD:
                redundant_pairs.append((abs(r), r, features[i], features[j]))
    redundant_pairs.sort(reverse=True)

    if redundant_pairs:
        print("\nHighly correlated pairs (|r| >= {:.2f}) — redundant features:".format(
            REDUNDANCY_THRESHOLD))
        print(f"  {'Feature 1':<30} {'Feature 2':<30} {'r':>7}")
        print("  " + "-" * 70)
        for _, r, f1, f2 in redundant_pairs:
            print(f"  {f1:<30} {f2:<30} {r:>7.4f}")

    # 3. Final recommendation
    weak = {feat for score, feat in f_scores if score < SEPARATION_THRESHOLD}
    redundant_to_drop = set()
    for _, _, f1, f2 in redundant_pairs:
        # Drop the one with the lower F-ratio
        score1 = next(s for s, f in f_scores if f == f1)
        score2 = next(s for s, f in f_scores if f == f2)
        redundant_to_drop.add(f1 if score1 < score2 else f2)

    drop = weak | redundant_to_drop
    keep = [feat for _, feat in f_scores if feat not in drop]

    print("\nRecommended features for logistic regression:")
    print("  Keep  :", ", ".join(keep) if keep else "none")
    print("  Drop  :", ", ".join(sorted(drop)) if drop else "none")
    print()


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
    fig, axes = plt.subplots(n, n, figsize=(n * 1.5, n * 1.5))
    fig.suptitle("Pair Plot — Hogwarts Dataset", fontsize=14, y=1.01)

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

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

    plt.tight_layout()
    plt.show()

    print_feature_analysis(rows, features, houses)


if __name__ == "__main__":
    main()
