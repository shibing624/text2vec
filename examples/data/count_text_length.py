import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def count_text_length(files):
    lengths = defaultdict(int)
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                text1_length, text2_length = len(obj["text1"]), len(obj["text2"])
                lengths[text1_length] += 1
                lengths[text2_length] += 1
    return lengths


def plot_histogram(length_counts):
    sorted_lengths = sorted(length_counts.items(), key=lambda x: x[0])
    total_counts = sum([count for _, count in sorted_lengths])
    cumulative_counts = np.cumsum([count for _, count in sorted_lengths]) / total_counts

    # Find the index where the cumulative percentage reaches 95%
    cutoff_index = np.where(cumulative_counts >= 0.90)[0][0]
    cutoff_length = sorted_lengths[cutoff_index][0]

    truncated_data = {k: v for k, v in length_counts.items() if k <= 800}

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(truncated_data.keys(), truncated_data.values())
    ax.set_xlabel('Text Length')
    ax.set_ylabel('Count')
    ax.set_title('Text Length Distribution')

    # Add 90% cumulative line
    ax.axvline(cutoff_length, color='r', linestyle='--', label=f'90% cumulative ({cutoff_length})')
    ax.legend()

    plt.show()


def main():
    files = [filename for filename in os.listdir('.') if filename.endswith('.jsonl')]
    result = count_text_length(files)

    for length, count in sorted(result.items(), key=lambda x: x[0]):
        print(f"Length: {length}, Count: {count}")

    # Plot histogram
    plot_histogram(result)


if __name__ == "__main__":
    main()
