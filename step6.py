import step1,step2,step3,step4,step5

import time
import tracemalloc
import random
import step1, step2, step3, step4, step5
import sys

sys.setrecursionlimit(100000)  # 解决构树递归深度问题

# --- 插入排序 ---
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# --- 自改进排序主函数（只执行稳态部分） ---
def self_improving_sort(data, v_list, di_trees):
    buckets = step5.bucket_classify(data, di_trees, v_list)
    for bucket in buckets:
        insertion_sort(bucket)
    result = []
    for bucket in buckets:
        result.extend(bucket)
    return result

# --- 快速排序 ---
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x < pivot]
    greater_equal = [x for x in arr[1:] if x >= pivot]
    return quicksort(less) + [pivot] + quicksort(greater_equal)

# --- 测量时间 + 内存 ---
def measure_time_memory(func, *args):
    tracemalloc.start()
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (end - start), (current / 1024), (peak / 1024)

# ✅ 单独运行一次训练阶段，返回排序器所需结构
def train_sorter(n):
    training_data = step1.collect_training_data(n)
    v_list = step2.build_v_list(training_data, n)
    _, prob_matrix = step3.estimate_distributions(training_data, v_list, n)
    di_trees = step4.build_all_di_trees(prob_matrix)
    return v_list, di_trees

# ✅ 测量训练时间 + 空间
def measure_training(n):
    elapsed, _, peak = measure_time_memory(train_sorter, n)
    v_list, di_trees = train_sorter(n)  # 再跑一次以获得结构（首次用于计时）
    return v_list, di_trees, elapsed, peak

# ✅ 主函数：只训练一次，测试多轮
def benchmark_algorithms(n, runs=5):
    print(f"\n[阶段一] 自改进排序训练阶段：")
    v_list, di_trees, t_train, m_train = measure_training(n)
    print(f"训练时间：{t_train:.6f} 秒，训练内存峰值：{m_train:.2f} KB")

    # ✅ 测试数据生成器（每轮生成一个新输入）
    def data_gen():
        return step1.generate_input(n)

    # ✅ 不再重复训练，复用 v_list 和 di_trees
    algorithms = [
        ("自改进排序（稳态）", lambda data: self_improving_sort(data, v_list, di_trees)),
        ("Python sorted()", lambda data: sorted(data)),
        ("快速排序", lambda data: quicksort(data))
    ]

    print(f"\n[阶段二] 各排序算法（稳态）性能对比：")
    print(f"{'算法':<22} {'平均时间(秒)':<18} {'平均峰值内存(KB)':<20}")
    print("-" * 60)

    for name, func in algorithms:
        time_total = 0
        mem_total = 0
        for _ in range(runs):
            data = data_gen()
            elapsed, _, peak = measure_time_memory(func, data)
            time_total += elapsed
            mem_total += peak
        print(f"{name:<22} {time_total / runs:<18.6f} {mem_total / runs:<20.2f}")

# --- 程序入口 ---
if __name__ == "__main__":
    benchmark_algorithms(n=1000, runs=5)

