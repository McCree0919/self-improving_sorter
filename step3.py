import step1
import step2

import bisect


import bisect
import numpy as np

def estimate_distributions(training_data, v_list, n):
    """
    统计每个位置 x_i 落入 V-list 的哪个区间，返回频率矩阵和概率矩阵
    """
    lambda_rounds = len(training_data)
    num_intervals = len(v_list) - 1

    # ✅ 关键点1：使用 v_list 控制第二维
    freq_matrix = np.zeros((n, num_intervals), dtype=np.uint16)

    for data in training_data:
        for i, x in enumerate(data):
            k = bisect.bisect_right(v_list, x) - 1
            k = max(0, min(k, num_intervals - 1))  # ✅ 边界保护
            freq_matrix[i, k] += 1

    # ✅ 关键点2：使用 float32 降低内存（8GB 降到 4GB）
    prob_matrix = freq_matrix.astype(np.float32) / lambda_rounds

    # ✅ 返回 list 保持兼容性
    return freq_matrix.tolist(), prob_matrix.tolist()

if __name__ == "__main__":
    n = 10
    training_data = step1.collect_training_data(n)
    v_list = step2.build_v_list(training_data, n)
    freq_matrix, prob_matrix = estimate_distributions(training_data, v_list, n)

    print("频率矩阵 freq_matrix[位置][区间]:")
    for i, row in enumerate(freq_matrix):
        print(f"位置{i}: {row}")

    print("\n概率矩阵 prob_matrix[位置][区间]:")
    for i, row in enumerate(prob_matrix):
        print(f"位置{i}: {row}")
