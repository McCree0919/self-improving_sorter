import time
import tracemalloc
import random
import pickle
import step1, step2, step3, step4, step5

# === 控制变量：生成一次统一测试数据并保存 ===


def load_test_data(filename="test_data.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

# === 插入排序 ===
def insertion_sort(arr):
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# === 快速排序 ===
def quicksort(arr):
    arr = arr.copy()
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x < pivot]
    greater_equal = [x for x in arr[1:] if x >= pivot]
    return quicksort(less) + [pivot] + quicksort(greater_equal)

# === 归并排序 ===
def mergesort(arr):
    arr = arr.copy()
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# === 堆排序 ===
import heapq
def heapsort(arr):
    arr = arr.copy()
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]

# === 自改进排序 ===
def self_improving_sort(data, v_list, di_trees):
    buckets = step5.bucket_classify(data, di_trees, v_list)
    for bucket in buckets:
        insertion_sort(bucket)
    result = []
    for bucket in buckets:
        result.extend(bucket)
    return result

# === 训练阶段（只运行一次）===
def measure_training(n_train):
    def training():
        training_data = step1.collect_training_data(n_train)
        v_list = step2.build_v_list(training_data, n_train)
        _, prob_matrix = step3.estimate_distributions(training_data, v_list, n_train)
        di_trees = step4.build_all_di_trees(prob_matrix)
        return v_list, di_trees

    elapsed, _, peak = measure_time_memory(training)
    v_list, di_trees = training()
    print(f"\n[训练阶段] n = {n_train}")
    print(f"训练时间：{elapsed:.6f} 秒，内存峰值：{peak:.2f} KB")
    return v_list, di_trees

# === 性能测量函数 ===
def measure_time_memory(func, *args):
    tracemalloc.start()
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end - start), (current / 1024), (peak / 1024)

# === 稳态阶段测试对比 ===
def test_with_fixed_data(v_list, di_trees, filename="test_data.pkl", runs=3):
    print(f"\n[测试阶段] 使用文件数据：{filename}，运行 {runs} 次")
    original_data = load_test_data(filename)

    algorithms = [
        ("自改进排序", lambda data: self_improving_sort(data, v_list, di_trees)),
        ("快速排序", quicksort),
        ("归并排序", mergesort),
        ("堆排序", heapsort),
        ("插入排序", insertion_sort),
    ]

    print(f"{'算法':<20} {'平均时间(秒)':<18} {'平均峰值内存(KB)':<20}")
    print("-" * 60)

    for name, func in algorithms:
        total_time = 0
        total_mem = 0
        for _ in range(runs):
            data = original_data.copy()
            elapsed, _, peak = measure_time_memory(func, data)
            total_time += elapsed
            total_mem += peak
        print(f"{name:<20} {total_time / runs:<18.6f} {total_mem / runs:<20.2f}")

# === 主程序入口 ===
if __name__ == "__main__":
    test_data_size = 100000
    training_size = 100000
    runs = 3
    filename = "test_data.pkl"

    v_list, di_trees = measure_training(training_size)
    test_with_fixed_data(v_list, di_trees, filename, runs=runs)
