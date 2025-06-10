import random
import math

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

def collect_training_data(n, lambda_rounds=None, dist_type="piecewise"):
    if lambda_rounds is None:
        lambda_rounds = math.ceil(math.log2(n))
    return [generate_input(n, dist_type) for _ in range(lambda_rounds)]


if __name__ == "__main__":
    n = 10  # 采样间隔
    training_data = collect_training_data(n)

    for i, data in enumerate(training_data):
        print(f"第 {i+1} 轮训练数据: {data}")
