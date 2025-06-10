# debug_bucket_flow.py

import numpy as np
import step1
import step2
import step3
import step4
import step5
from utils import generate_input, plot_heatmap


def debug_bucket_flow(n=2000, dist_type="piecewise", lambda_rounds=10):
    print("\n[🔍 Step 1] 采集训练数据")
    training_data = [generate_input(n, dist_type) for _ in range(lambda_rounds)]
    print("[✓] 训练组数:", len(training_data))
    print("[✓] 每组长度:", len(training_data[0]))

    print("\n[🔍 Step 2] 构建 V-list")
    v_list = step2.build_v_list(training_data, v_length=n)
    print("[✓] V-list 长度:", len(v_list))
    print("[✓] 是否递增:", all(v_list[i] < v_list[i+1] for i in range(len(v_list)-1)))

    print("\n[🔍 Step 3] 概率估计 + Step 4 构建 Di 树")
    _, prob_matrix = step3.estimate_distributions(training_data, v_list, n)
    di_trees = step4.build_all_di_trees(prob_matrix)
    print("[✓] Di 树数量:", len(di_trees))

    print("\n[🔍 Step 5] 桶划分")
    test_data = generate_input(n, dist_type)
    print("[✓] 测试数据长度:", len(test_data))

    if len(di_trees) != len(test_data):
        print("[❌ 错误] Di 树数量和测试数据不匹配！")
    else:
        print("[✓] Di 树与测试数据一一对应")

    buckets = step5.bucket_classify(test_data, di_trees, v_list)
    bucket_sizes = [len(b) for b in buckets]
    print("[✓] 总桶数:", len(bucket_sizes))
    print("[✓] 最大桶大小:", max(bucket_sizes))
    print("[✓] 非空桶数:", sum(1 for x in bucket_sizes if x > 0))
    print("[✓] 均值/标准差:", f"{np.mean(bucket_sizes):.2f} / {np.std(bucket_sizes):.2f}")

    heatmap_data = np.zeros((1, len(bucket_sizes)))
    heatmap_data[0, :] = bucket_sizes
    plot_heatmap(heatmap_data, title=f"Bucket Heatmap ({dist_type}, n={n})")


if __name__ == "__main__":
    debug_bucket_flow(n=2000, dist_type="piecewise", lambda_rounds=10)