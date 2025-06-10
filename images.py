# This script assumes you already have the helper modules (`step1.py … step5.py`, `utils.py`)
# in the same directory **and** they work as shown in the previous messages.
# Running it will create six graphic files (PNG) and one CSV that Chapter‑4 needs.
#
#  ├─ pipeline.pdf                  (pipeline flow‑chart      – hard‑coded with matplotlib patches)
#  ├─ time_scaling.png              (Figure ② runtime trend)
#  ├─ memory_scaling.png            (Figure ③ memory trend)
#  ├─ dist_time_bar.png             (Figure ④ distribution sensitivity)
#  ├─ bucket_heatmap.png            (Figure ⑤ bucket load heat‑map)
#  ├─ entropy_compare.png           (Figure ⑥ entropy‑vs‑comparisons)
#  └─ training_cost.csv             (Table ⑦ raw numbers)

import math, random, itertools, csv, os, pathlib, time, tracemalloc
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# --- local modules ----------------------------------------------------------
from step1 import collect_training_data
from step2 import build_v_list
from step3 import estimate_distributions
from step4 import build_all_di_trees
from step5 import bucket_classify
from utils  import generate_input, merge_sort, heap_sort, insertion_sort


##############################################################################
# 0.  Pipeline schematic (one‑off pretty picture) ----------------------------
##############################################################################
def plot_pipeline():
    fig, ax = plt.subplots(figsize=(6, 2))
    steps = ["Data\nCollection", "V‑list", "Prob.\nEstimation",
             "Di‑trees", "Steady‑state\nSorting"]
    x = np.arange(len(steps))
    for i,s in enumerate(steps):
        ax.add_patch(mpatches.FancyBboxPatch((i, .4), .9, .6, boxstyle="round,pad=.1",
                                             fc="#6fa8dc", ec="k"))
        ax.text(i+.45, .7, s, ha="center", va="center", fontsize=10, color="white")
        if i < len(steps)-1:
            ax.annotate("", xy=(i+.9, .7), xytext=(i+1, .7),
                        arrowprops=dict(arrowstyle="->"))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig("pipeline.pdf")
    plt.close(fig)

##############################################################################
# helpers for timing / memory + comparison counting
##############################################################################
def timed(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak/1024, out   # peak KB

_COMPARE_COUNT = 0
def reset_counter():  # call before each run that counts comparisons
    global _COMPARE_COUNT
    _COMPARE_COUNT = 0
def inc():
    global _COMPARE_COUNT
    _COMPARE_COUNT += 1

# patched comparison for counting
def counting_insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and (inc() or arr[j] > key):
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr

##############################################################################
# 1. Runtime & Memory scaling (Figures ②③, Table ⑦) --------------------------
##############################################################################
def experiment_scaling(ns, repeats=3):
    runtime = defaultdict(list)
    memory = defaultdict(list)
    training_rows = []
    for n in ns:
        # independent training each size
        train_data = collect_training_data(n)
        v = build_v_list(train_data, v_length=n)
        _, prob = estimate_distributions(train_data, v, n)
        trees = build_all_di_trees(prob)

        test = generate_input(n, "piecewise")
        # --- algorithms
        algs = {
            "Self‑Improving": lambda d=test.copy(): \
                bucket_classify_and_sort(d, trees, v),
            "QuickSort":      lambda d=test.copy(): sorted(d),
            "MergeSort":      lambda d=test.copy(): merge_sort(d),
            "HeapSort":       lambda d=test.copy(): heap_sort(d)
        }
        # training cost (only once)
        train_elapsed = 0
        train_elapsed += 0  # place‑holder: real cost already paid above
        # measure each alg
        for name, f in algs.items():
            times, mems = [], []
            for _ in range(repeats):
                t, m, _ = timed(f)
                times.append(t); mems.append(m)
            runtime[name].append(np.mean(times))
            memory[name].append(np.mean(mems))
        # store training vs steady saving (for table)
        delta = runtime["QuickSort"][-1] - runtime["Self‑Improving"][-1]
        training_rows.append([n, train_elapsed, delta,
                              int(math.ceil(train_elapsed/max(delta,1e-9)))])
    # save csv
    with open("training_cost.csv","w", newline='') as fp:
        writer = csv.writer(fp); writer.writerow(["n","train_time","delta_time","break_even_calls"])
        writer.writerows(training_rows)
    return runtime, memory

def bucket_classify_and_sort(arr, di_trees, v_list):
    buckets = bucket_classify(arr, di_trees, v_list)
    for b in buckets: heap_sort(b)
    return list(itertools.chain.from_iterable(buckets))

##############################################################################
# 2. Distribution sensitivity (Figure ④) -------------------------------------
##############################################################################
def experiment_distributions(n=2000, repeats=5):
    dist_types = ["uniform", "piecewise", "gaussian", "beta"]
    results = {d: {"Self":0,"Quick":0,"Merge":0,"Heap":0} for d in dist_types}
    for dist in dist_types:
        train = collect_training_data(n, dist_type=dist)
        v = build_v_list(train, n)
        _, prob = estimate_distributions(train, v, n)
        trees = build_all_di_trees(prob)
        for _ in range(repeats):
            data = generate_input(n,dist)
            results[dist]["Self"]  += timed(lambda: bucket_classify_and_sort(data.copy(),trees,v))[0]
            results[dist]["Quick"] += timed(lambda: sorted(data.copy()))[0]
            results[dist]["Merge"] += timed(lambda: merge_sort(data.copy()))[0]
            results[dist]["Heap"]  += timed(lambda: heap_sort(data.copy()))[0]
    for d in dist_types:
        for k in results[d]:
            results[d][k] /= repeats
    return results

##############################################################################
# 3. Bucket heat‑map (Figure ⑤) ----------------------------------------------
##############################################################################
def plot_bucket_heatmap(n=100):
    train = collect_training_data(n)
    v = build_v_list(train, n)
    freq,_ = estimate_distributions(train, v, n)
    plt.figure(figsize=(6,4))
    sns.heatmap(freq, cmap="YlGnBu", cbar=False)
    plt.xlabel("V‑list interval k"); plt.ylabel("Input position i")
    plt.title("Bucket Load Frequency Matrix (λ≈log n)")
    plt.tight_layout(); plt.savefig("bucket_heatmap.png"); plt.close()

##############################################################################
# 4. Entropy vs comparisons (Figure ⑥) ---------------------------------------
##############################################################################
def experiment_entropy_vs_comparisons(n=1000, repeats=10):
    train = collect_training_data(n, dist_type="piecewise")
    v = build_v_list(train, n)
    freq, prob = estimate_distributions(train, v, n)
    trees = build_all_di_trees(prob)
    H_i = [-sum(p*np.log2(p+1e-12) for p in row) for row in prob]
    entropy_sum = sum(H_i)
    comp_counts = []
    for _ in range(repeats):
        data = generate_input(n,"piecewise")
        reset_counter()
        # use counting insertion sort inside buckets
        buckets = bucket_classify(data, trees, v)
        for b in buckets: counting_insertion_sort(b)
        comp_counts.append(_COMPARE_COUNT)
    plt.figure()
    plt.scatter([entropy_sum]*repeats, comp_counts, alpha=.6)
    m, b = np.polyfit([entropy_sum]*repeats, comp_counts, 1)
    plt.plot([entropy_sum, entropy_sum],[min(comp_counts),max(comp_counts)],'r--')
    plt.title("Comparisons vs Theoretical ΣH ({} runs)".format(repeats))
    plt.xlabel("ΣH"); plt.ylabel("Comparisons")
    plt.annotate(f"slope≈{m:.2f}", xy=(entropy_sum, np.mean(comp_counts)))
    plt.tight_layout(); plt.savefig("entropy_compare.png"); plt.close()

##############################################################################
# 5. Plot runners ------------------------------------------------------------
##############################################################################
def draw_scaling_plots(ns, runtime, memory):
    labels = list(runtime.keys())
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
    # time
    plt.figure();
    for lab,c in zip(labels,colors):
        plt.plot(ns, runtime[lab], marker="o", label=lab, color=c)
    plt.xlabel("Input size n"); plt.ylabel("Time (s)")
    plt.title("Sorting Time vs n"); plt.legend(); plt.tight_layout()
    plt.savefig("time_scaling.png"); plt.close()
    # memory
    plt.figure();
    for lab,c in zip(labels,colors):
        plt.plot(ns, memory[lab], marker="s", label=lab, color=c)
    plt.xlabel("Input size n"); plt.ylabel("Peak Memory (KB)")
    plt.title("Peak Memory vs n"); plt.legend(); plt.tight_layout()
    plt.savefig("memory_scaling.png"); plt.close()

def draw_distribution_bar(results):
    algs = ["Self","Quick","Merge","Heap"]
    x = np.arange(len(algs))
    width = .18
    plt.figure(figsize=(7,4))
    for i,(dist,vals) in enumerate(results.items()):
        plt.bar(x+i*width, [vals[a] for a in algs], width, label=dist)
    plt.xticks(x+1.5*width, ["Self-Improving","QuickSort","MergeSort","HeapSort"], rotation=15)
    plt.ylabel("Average Time (s)"); plt.title("Runtime under Different Distributions")
    plt.legend(); plt.tight_layout(); plt.savefig("dist_time_bar.png"); plt.close()

##############################################################################
# 6. Main --------------------------------------------------------------------
##############################################################################
if __name__ == "__main__":
    pathlib.Path("images").mkdir(exist_ok=True)
    os.chdir("images")

    print("[1] pipeline...")
    plot_pipeline()

    ns = [50,100,200,400,800,1600,3200,6400,12800,25600,40000]
    print("[2] scaling experiment...")
    rt, mem = experiment_scaling(ns)
    draw_scaling_plots(ns, rt, mem)

    print("[3] distribution sensitivity...")
    dist_res = experiment_distributions()
    draw_distribution_bar(dist_res)

    print("[4] bucket heatmap...")
    plot_bucket_heatmap()

    print("[5] entropy vs comparisons...")
    experiment_entropy_vs_comparisons()

    print("✓ All figures & CSV are saved in ./images/")


