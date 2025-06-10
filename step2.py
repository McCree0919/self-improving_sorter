import step1

"""
def build_v_list(training_data, v_length=None):
    
    输入：
    - training_data: 二维数组（lambda_rounds × n）
    - v_length: V-list 长度（默认为 n）

    输出：
    - 带哨兵的 V-list，长度为 v_length + 2（含 -inf 和 +inf）
    
    merged = []
    for data in training_data:
        merged.extend(data)
    merged.sort()

    if v_length is None:
        v_length = len(training_data[0])

    step = len(merged) // (v_length + 1)
    v_list = [float('-inf')]
    for i in range(1, v_length + 1):
        v_list.append(merged[i * step])
    v_list.append(float('inf'))
    return v_list
"""

import numpy as np

def build_v_list(training_data, v_length):
    merged = []
    for row in training_data:
        merged.extend(row)

    percentiles = np.linspace(0, 100, v_length + 1)[1:-1]
    boundaries = np.percentile(merged, percentiles).tolist()
    return [float('-inf')] + boundaries + [float('inf')]


if __name__ == "__main__":
    n = 10
    training_data = step1.collect_training_data(n)
    v_list = build_v_list(training_data, n)
    print("V-list (带哨兵):", v_list)

