import numpy as np
import matplotlib.pyplot as plt

def main():
    # 尝试加载两组历史数据
    try:
        uniform_data = np.load("loss_history_uniform.npz")
        rar_data = np.load("loss_history_rar.npz")
    except FileNotFoundError as e:
        print(f"Error: Could not load comparison data: {e}")
        print("Please ensure you run both 'python train.py --mode uniform' and 'python train.py --mode rar' first.")
        return

    # 提取数据
    steps_uni = uniform_data["steps"]
    train_loss_uni = uniform_data["train_loss"]
    val_loss_uni = uniform_data["val_loss"]

    steps_rar = rar_data["steps"]
    train_loss_rar = rar_data["train_loss"]
    val_loss_rar = rar_data["val_loss"]

    # 创建 1x2 对比画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Training Loss 对比
    ax1.plot(steps_uni, train_loss_uni, 'b-', label='Uniform Baseline', alpha=0.8)
    ax1.plot(steps_rar, train_loss_rar, 'r-', label='RAR Adaptive', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Convergence (Log Scale)')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()

    # 2. Validation Loss 对比 (在固定的 100x100 均匀网格上计算)
    ax2.plot(steps_uni, val_loss_uni, 'b-', label='Uniform Baseline', alpha=0.8)
    ax2.plot(steps_rar, val_loss_rar, 'r-', label='RAR Adaptive', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('PDE Validation Loss (Uniform Grid)')
    ax2.set_title('Validation Loss Accuracy Comparison (Log Scale)')
    ax2.grid(True, which="both", ls="--")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=150)
    print("Saved comparison plot to loss_comparison.png")

if __name__ == '__main__':
    main()
