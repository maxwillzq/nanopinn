import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def load_params(path):
    """从 npz 加载权重并重建 PyTree"""
    data = jnp.load(path)
    num_layers = len([k for k in data.keys() if k.startswith("W_")])
    params = []
    for idx in range(num_layers):
        params.append({
            'W': data[f"W_{idx}"],
            'b': data[f"b_{idx}"]
        })
    return params

def forward(params, t, x):
    """单点预测"""
    z = jnp.array([[t], [x]])
    for layer in params[:-1]:
        z = jnp.tanh(jnp.dot(layer['W'], z) + layer['b'])
    out = jnp.dot(params[-1]['W'], z) + params[-1]['b']
    return out[0, 0]

forward_v = jax.vmap(forward, in_axes=(None, 0, 0))

def main():
    params = load_params("pinn_params.npz")

    # 创建用于绘图的可视化网格
    n_t, n_x = 200, 200
    t_ticks = np.linspace(0, 1, n_t)
    x_ticks = np.linspace(-1, 1, n_x)
    T, X = np.meshgrid(t_ticks, x_ticks)
    
    # 展平以便向量化预测
    t_flat = T.flatten()
    x_flat = X.flatten()
    u_flat = forward_v(params, t_flat, x_flat)
    U = np.array(u_flat).reshape(n_x, n_t)

    # 创建双图画布
    fig = plt.figure(figsize=(14, 5))

    # 1. 速度场时空热力图
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(U, extent=[0, 1, -1, 1], cmap='rainbow', aspect='auto', origin='lower')
    fig.colorbar(im, ax=ax1, label='u(t, x)')
    
    # 叠加自适应采样的共轭点
    data = jnp.load("pinn_params.npz")
    if "t_col" in data and "x_col" in data:
        t_col = data["t_col"]
        x_col = data["x_col"]
        ax1.scatter(t_col, x_col, color='black', alpha=0.15, s=0.3, label='Collocation Points')
        ax1.legend(loc='upper right')

    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Space (x)')
    ax1.set_title("Burgers' Equation Flow Field Colormap")


    # 2. 特定时间的速度剖面曲线
    ax2 = fig.add_subplot(122)
    target_times = [0.0, 0.2, 1.0 / np.pi, 0.5, 1.0]
    for t_val in target_times:
        t_arr = jnp.ones_like(x_ticks) * t_val
        u_val = forward_v(params, t_arr, x_ticks)
        label = f"t = {t_val:.2f}"
        if np.isclose(t_val, 1.0 / np.pi):
            label = f"t = 1/π ≈ {t_val:.3f} (Break)"
        ax2.plot(x_ticks, u_val, label=label)

    
    ax2.set_xlabel('Space (x)')
    ax2.set_ylabel('Velocity (u)')
    ax2.set_title('Velocity Profiles at Specific Timestamps')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("burgers_shock_wave.png", dpi=150)
    print("Saved plot to burgers_shock_wave.png")

if __name__ == '__main__':
    main()
