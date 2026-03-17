import csv
import sys


def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


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


def read_csv(filename):
  with open(filename, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    data = {header: [] for header in headers}

    for row in reader:
      for header in headers:
        value = row[header]
        if value is not None and value != "" and is_float(value):
          data[header].append(float(value))

  numeric_data = {}
  for key, values in data.items():
    if ft_count(values) > 0:
        numeric_data[key] = values
  return numeric_data


def compute_stats(values):
  sorted_values = sorted(values)
  return {
    "Count": ft_count(values),
    "Mean": ft_sum(values) / ft_count(values) if ft_count(values) > 0 else 0,
    "Std": (ft_sum((x - ft_sum(values) / ft_count(values)) ** 2 for x in values) / ft_count(values)) ** 0.5 if ft_count(values) > 0 else 0,
    "Min": sorted_values[0] if ft_count(values) > 0 else 0,
    "25%": sorted_values[int(0.25 * ft_count(values))] if ft_count(values) > 0 else 0,
    "50%": sorted_values[int(0.5 * ft_count(values))] if ft_count(values) > 0 else 0,
    "75%": sorted_values[int(0.75 * ft_count(values))] if ft_count(values) > 0 else 0,
    "Max": sorted_values[-1] if ft_count(values) > 0 else 0
  }


def print_stats(stats_by_feature):
  stat_names = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
  headers = list(stats_by_feature.keys())

  first_col_width = 10
  col_width = 15

  print("".ljust(first_col_width), end="")
  for h in headers:
    print(f"{h[:col_width-1]:>{col_width}}", end="")
  print()

  for stat in stat_names:
    print(f"{stat:<{first_col_width}}", end="")
    for h in headers:
      value = stats_by_feature[h][stat]
      if value is None:
        print(f"{'NaN':>{col_width}}", end="")
      else:
        print(f"{value:>{col_width}.6f}", end="")
    print()


def main():
  if len(sys.argv) != 2:
    print("Usage: python describe.py dataset_train.csv")
    sys.exit(1)

  filename = sys.argv[1]
  data = read_csv(filename)

  stats_by_feature = {}
  for feature, values, in data.items():
    stats_by_feature[feature] = compute_stats(values)

  print_stats(stats_by_feature)


if __name__ == "__main__":
  main()
