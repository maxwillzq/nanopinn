import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# ----------------- 加载模型结构 -----------------
def load_params(path):
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
    z = jnp.array([[t], [x]])
    for layer in params[:-1]:
        z = jnp.tanh(jnp.dot(layer['W'], z) + layer['b'])
    out = jnp.dot(params[-1]['W'], z) + params[-1]['b']
    return out[0, 0]

forward_v = jax.vmap(forward, in_axes=(None, 0, 0))

def residual(params, t, x, nu):
    u = forward(params, t, x)
    u_t = jax.grad(forward, argnums=1)(params, t, x)
    u_x = jax.grad(forward, argnums=2)(params, t, x)
    u_xx = jax.grad(jax.grad(forward, argnums=2), argnums=2)(params, t, x)
    return u_t + u * u_x - nu * u_xx

residual_v = jax.vmap(residual, in_axes=(None, 0, 0, None))

def main():
    # 1. 加载两个模型的权重
    try:
        params_uni = load_params("pinn_params_uniform.npz")
        params_rar = load_params("pinn_params_rar.npz")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please train both models first.")
        return

    nu = 0.01 / jnp.pi
    
    # 2. 构造一个在 t = 1.0 时非常密集的空间网格 x ∈ [-0.2, 0.2]（激波中心区域）
    x_dense = np.linspace(-0.2, 0.2, 1000)
    t_dense = np.ones_like(x_dense) * 1.0

    # 3. 计算预测的速度值
    u_uni = forward_v(params_uni, t_dense, x_dense)
    u_rar = forward_v(params_rar, t_dense, x_dense)

    # 4. 计算物理残差的绝对值
    f_uni = np.abs(residual_v(params_uni, t_dense, x_dense, nu))
    f_rar = np.abs(residual_v(params_rar, t_dense, x_dense, nu))

    # 5. 开始画图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：速度剖面对比（看看谁的激波过渡更陡峭、更准确）
    ax1.plot(x_dense, u_uni, 'b--', label='Uniform Baseline (Fixed)', alpha=0.8)
    ax1.plot(x_dense, u_rar, 'r-', label='Hybrid RAR (Adaptive)', alpha=0.8)
    ax1.set_xlabel('Space (x) near Shock')
    ax1.set_ylabel('Velocity (u) at t = 1.0')
    ax1.set_title('Velocity Profile Comparison at t = 1.0 (Zoomed)')
    ax1.grid(True)
    ax1.legend()

    # 右图：物理残差绝对值分布（看看激波处的控制方程是否得到了完美满足）
    ax2.plot(x_dense, f_uni, 'b--', label='Uniform Baseline (Fixed)', alpha=0.8)
    ax2.plot(x_dense, f_rar, 'r-', label='Hybrid RAR (Adaptive)', alpha=0.8)
    ax2.set_xlabel('Space (x) near Shock')
    ax2.set_ylabel('Absolute PDE Residual |f(t,x)|')
    ax2.set_yscale('log')
    ax2.set_title('PDE Residual Profile at t = 1.0 (Log Scale)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("shock_resolution_comparison.png", dpi=150)
    print("Saved local comparison plot to shock_resolution_comparison.png")

if __name__ == '__main__':
    main()
