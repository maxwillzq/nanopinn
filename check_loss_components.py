import jax
import jax.numpy as jnp
import numpy as np

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

def forward_single(params, t, x):
    z = jnp.array([[t], [x]])
    for layer in params[:-1]:
        z = jnp.tanh(jnp.dot(layer['W'], z) + layer['b'])
    out = jnp.dot(params[-1]['W'], z) + params[-1]['b']
    return out[0, 0]

forward_v = jax.vmap(forward_single, in_axes=(None, 0, 0))

def residual(params, t, x, nu):
    u = forward_single(params, t, x)
    u_t = jax.grad(forward_single, argnums=1)(params, t, x)
    u_x = jax.grad(forward_single, argnums=2)(params, t, x)
    u_xx = jax.grad(jax.grad(forward_single, argnums=2), argnums=2)(params, t, x)
    return u_t + u * u_x - nu * u_xx

residual_v = jax.vmap(residual, in_axes=(None, 0, 0, None))

def main():
    # 1. 加载参数
    try:
        params_uni = load_params("pinn_params_uniform.npz")
        params_rar = load_params("pinn_params_rar.npz")
    except FileNotFoundError as e:
        print(f"Error: {e}. Train both models first.")
        return

    nu = 0.01 / jnp.pi

    # 2. 构造与训练一致的 IC/BC 数据
    N_ic = 100
    x_ic = jnp.linspace(-1, 1, N_ic)
    t_ic = jnp.zeros(N_ic)
    u_ic = -jnp.sin(jnp.pi * x_ic)

    N_bc = 100
    t_bc_half = jnp.linspace(0, 1, N_bc // 2)
    t_bc = jnp.concatenate([t_bc_half, t_bc_half])
    x_bc = jnp.concatenate([jnp.ones(N_bc // 2) * -1.0, jnp.ones(N_bc // 2) * 1.0])
    u_bc = jnp.zeros(N_bc)

    # 3. 构造一个包含 10000 个点的均匀时空网格用来评估全局 PDE 残差
    t_flat = np.random.uniform(0, 1, 10000)
    x_flat = np.random.uniform(-1, 1, 10000)

    # 4. 计算各个损失项
    def get_components(params):
        # PDE Loss
        f_pred = residual_v(params, t_flat, x_flat, nu)
        mse_f = jnp.mean(f_pred ** 2)
        
        # IC Loss
        u_ic_pred = forward_v(params, t_ic, x_ic)
        mse_ic = jnp.mean((u_ic_pred - u_ic) ** 2)
        
        # BC Loss
        u_bc_pred = forward_v(params, t_bc, x_bc)
        mse_bc = jnp.mean((u_bc_pred - u_bc) ** 2)
        
        return float(mse_f), float(mse_ic), float(mse_bc)

    uni_f, uni_ic, uni_bc = get_components(params_uni)
    rar_f, rar_ic, rar_bc = get_components(params_rar)

    print("\n================== LOSS COMPONENTS BREAKDOWN ==================")
    print(f"{'Loss Component':<25} | {'Uniform Baseline':<20} | {'Hybrid RAR (Ours)':<20}")
    print("-" * 75)
    print(f"{'PDE Residual Loss (MSE)':<25} | {uni_f:<20.6e} | {rar_f:<20.6e}")
    print(f"{'Initial Condition (IC)':<25} | {uni_ic:<20.6e} | {rar_ic:<20.6e}")
    print(f"{'Boundary Condition (BC)':<25} | {uni_bc:<20.6e} | {rar_bc:<20.6e}")
    print("===============================================================\n")

if __name__ == '__main__':
    main()
