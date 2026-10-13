import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def setup_output_dirs(script_name):
    dirs = {
        "root":    f"{script_name}-Results",
        "plots":   f"{script_name}-Results/plots",
        "models":  f"{script_name}-Results/models",
        "history": f"{script_name}-Results/history",
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs

def get_next_run_number(script_name):
    DIRS = {
        "base": f"{script_name}-Results",
    }
    DIRS["history"] = os.path.join(DIRS["base"], "history")
    existing = glob.glob(os.path.join(DIRS["history"], "loss_history_run*.csv"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        base = os.path.basename(f)
        try:
            num = int(base.replace("loss_history_run", "").replace(".csv", ""))
            nums.append(num)
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")

    for idx, run_num in enumerate(sorted(merged["run"].unique())):
        subset = merged[merged["run"] == run_num].reset_index(drop=True)
        ax.semilogy(
            subset["epoch"],
            subset["loss_total"],
            color=cmap(idx % 10),
            linewidth=1.5,
            label=f"Run {run_num:03d}"
        )

    ax.set_xlabel("Epoch / Step")
    ax.set_ylabel("Total Loss (log scale)")
    ax.set_title("Loss History — All Runs")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    merged_plot = os.path.join(DIRS["history"], "loss_history_all.png")
    plt.savefig(merged_plot, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Merged history plot saved -> {merged_plot}")
