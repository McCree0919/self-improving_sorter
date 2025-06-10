# utils.py

import random
import time
import tracemalloc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. 数据生成 ---
def generate_input(n, dist_type="piecewise"):
    if dist_type == "uniform":
        return [random.uniform(0, 1) for _ in range(n)]
    elif dist_type == "piecewise":
        return [random.uniform(i, i + 1) for i in range(n)]
    elif dist_type == "gaussian":
        return [random.gauss(i, 1) for i in range(n)]
    elif dist_type == "beta":
        return [random.betavariate(2, 5) for _ in range(n)]
    else:
        raise ValueError("Unknown distribution type.")

# --- 2. 排序算法 ---
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def heap_sort(arr):
    import heapq
    return list(heapq.nsmallest(len(arr), arr))

# --- 3. 时间和内存测量 ---
def measure_time_memory(func, *args):
    tracemalloc.start()
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end - start), (peak / 1024)  # 返回秒和KB

# --- 4. 绘图工具 ---
def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False


def plot_bar_chart(x_labels, values, ylabel, title, rotation=45):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=x_labels, y=values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


def plot_line_chart(x, ys, labels, ylabel, title):
    plt.figure(figsize=(8, 6))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker='o', label=label)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlabel("Data Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Bucket Index")
    plt.ylabel("Data Index")
    plt.tight_layout()
    plt.show()
