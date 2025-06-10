
import random
import pickle

def generate_and_save_data(n, filename="test_data.pkl", seed=42):
    random.seed(seed)
    data = [random.uniform(i, i + 1) for i in range(n)]
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ 测试数据已生成并保存到 {filename}")

if __name__ == "__main__":
    test_data_size = 10000
    training_size = 100000
    runs = 3
    filename = "test_data.pkl"

    generate_and_save_data(test_data_size, filename)



"""
# test_step2_distribution.py

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import step1
import step2

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
        raise ValueError("Unknown distribution type")

def tv_list(n, dist_type="piecewise"):
    print(f"\n=== 测试分布类型: {dist_type} ===")
    training_data = [generate_input(n, dist_type) for _ in range(10)]  # lambda_rounds=10
    v_list = step2.build_v_list(training_data, n)

    merged = []
    for data in training_data:
        merged.extend(data)

    # 按V-list分类
    bucket_counts = [0 for _ in range(len(v_list)-1)]
    for x in merged:
        for i in range(len(v_list) - 1):
            if v_list[i] <= x < v_list[i+1]:
                bucket_counts[i] += 1
                break

    # 输出统计信息
    print(f"最大桶大小: {max(bucket_counts)}")
    print(f"最小桶大小: {min(bucket_counts)}")
    print(f"平均桶大小: {np.mean(bucket_counts):.2f}")
    print(f"标准差: {np.std(bucket_counts):.2f}")

    # 绘制分桶图
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(range(len(bucket_counts))), y=bucket_counts)
    plt.xlabel("Bucket Index")
    plt.ylabel("Number of Samples")
    plt.title(f"Bucket Distribution for {dist_type.capitalize()} Distribution")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n = 1000  # 测试数据长度

    for dist in ["uniform", "piecewise", "gaussian", "beta"]:
        tv_list(n, dist)
        """



