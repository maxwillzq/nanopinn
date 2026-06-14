import jax
import jax.numpy as jnp
import optax
import numpy as np
import argparse


def init_weights(layer_sizes, key):
    """手工初始化神经网络参数 (PyTree)"""
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
        w_key, b_key = jax.random.split(keys[i])
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        W = jax.random.uniform(w_key, (out_dim, in_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim, 1))
        params.append({'W': W, 'b': b})
    return params

def forward(params, t, x):
    """单点前向传播 (t, x) -> u"""
    z = jnp.array([[t], [x]])
    for layer in params[:-1]:
        z = jnp.tanh(jnp.dot(layer['W'], z) + layer['b'])
    out = jnp.dot(params[-1]['W'], z) + params[-1]['b']
    return out[0, 0]

# 向量化前向传播
forward_v = jax.vmap(forward, in_axes=(None, 0, 0))

def residual(params, t, x, nu):
    """物理控制偏微分方程 (PDE) 残差计算"""
    u = forward(params, t, x)
    u_t = jax.grad(forward, argnums=1)(params, t, x) # ∂u/∂t
    u_x = jax.grad(forward, argnums=2)(params, t, x) # ∂u/∂x
    u_xx = jax.grad(jax.grad(forward, argnums=2), argnums=2)(params, t, x) # ∂²u/∂x²
    return u_t + u * u_x - nu * u_xx

# 向量化物理残差计算
residual_v = jax.vmap(residual, in_axes=(None, 0, 0, None))

def loss_fn(params, t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_col, x_col, nu):
    """综合损失函数"""
    # 初始条件损失
    u_ic_pred = forward_v(params, t_ic, x_ic)
    loss_ic = jnp.mean((u_ic_pred - u_ic) ** 2)

    # 边界条件损失
    u_bc_pred = forward_v(params, t_bc, x_bc)
    loss_bc = jnp.mean((u_bc_pred - u_bc) ** 2)

    # 物理残差残差损失
    f_pred = residual_v(params, t_col, x_col, nu)
    loss_physics = jnp.mean(f_pred ** 2)

    return loss_ic + loss_bc + loss_physics

def save_params(params, path):
    """只保存模型参数到 npz 文件"""
    flat_params = {}
    for idx, layer in enumerate(params):
        flat_params[f"W_{idx}"] = layer['W']
        flat_params[f"b_{idx}"] = layer['b']
    jnp.savez(path, **flat_params)
    print(f"Saved model parameters to {path}")

def save_collocation_history(points_history, path):
    """保存共轭点坐标历史到 npz 文件"""
    jnp.savez(path, **points_history)
    print(f"Saved collocation history to {path}")

def adaptive_resample(params, t_adapt, x_adapt, nu, key, top_k=1500, noise_std=0.005):
    """自适应重采样机制 (RAR)：仅在自适应点集内部淘汰，找出残差最大的 top_k 个点在其周围分裂，并替换残差最小的点"""
    f_preds = residual_v(params, t_adapt, x_adapt, nu)
    abs_residuals = jnp.abs(f_preds)
    
    # 按照残差大小排序，在自适应集内部提取 top_k 个困难点
    top_indices = jnp.argsort(abs_residuals)[-top_k:]
    t_hard = t_adapt[top_indices]
    x_hard = x_adapt[top_indices]
    
    # 局部高斯扰动繁衍新点
    k1, k2 = jax.random.split(key)
    t_new = t_hard + jax.random.normal(k1, shape=t_hard.shape) * noise_std
    x_new = x_hard + jax.random.normal(k2, shape=x_hard.shape) * noise_std
    
    # 替换自适应集内部残差最小的 top_k 个点，以维持 Static Shapes
    bottom_indices = jnp.argsort(abs_residuals)[:top_k]
    t_adapt = t_adapt.at[bottom_indices].set(t_new)
    x_adapt = x_adapt.at[bottom_indices].set(x_new)
    
    t_adapt = jnp.clip(t_adapt, 0.0, 1.0)
    x_adapt = jnp.clip(x_adapt, -1.0, 1.0)
    
    return t_adapt, x_adapt

def main():
    # 命令行解析模式
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rar", choices=["uniform", "rar"])
    args = parser.parse_args()
    mode = args.mode

    # 超参数与初始化设置
    key = jax.random.PRNGKey(42)
    layer_sizes = [2, 20, 20, 20, 20, 1]
    nu = 0.01 / jnp.pi
    lr = 1e-3
    steps = 15000

    params = init_weights(layer_sizes, key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # 1. 初始条件采样: t=0, x ∈ [-1, 1], u = -sin(pi * x)
    N_ic = 100
    x_ic = jnp.linspace(-1, 1, N_ic)
    t_ic = jnp.zeros(N_ic)
    u_ic = -jnp.sin(jnp.pi * x_ic)

    # 2. 边界条件采样: x ∈ {-1, 1}, t ∈ [0, 1], u = 0
    N_bc = 100
    t_bc_half = jnp.linspace(0, 1, N_bc // 2)
    t_bc = jnp.concatenate([t_bc_half, t_bc_half])
    x_bc = jnp.concatenate([jnp.ones(N_bc // 2) * -1.0, jnp.ones(N_bc // 2) * 1.0])
    u_bc = jnp.zeros(N_bc)

    # 3. 物理控制共轭采样点 (训练采样)
    # 我们采用混合自适应采样 (Hybrid RAR)：50% 固定背景点，50% 自适应追踪点
    N_bg = 5000
    N_adapt = 5000
    
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # 固定背景点 (整个训练过程中不变)
    t_bg = jax.random.uniform(k1, (N_bg,), minval=0.0, maxval=1.0)
    x_bg = jax.random.uniform(k2, (N_bg,), minval=-1.0, maxval=1.0)
    # 自适应追踪点 (会动态重采样)
    t_adapt = jax.random.uniform(k3, (N_adapt,), minval=0.0, maxval=1.0)
    x_adapt = jax.random.uniform(k4, (N_adapt,), minval=-1.0, maxval=1.0)

    # 4. 固定的测试网格 (100x100 = 10000点，用于计算客观 Validation Loss)
    val_n = 100
    t_val_ticks = jnp.linspace(0.0, 1.0, val_n)
    x_val_ticks = jnp.linspace(-1.0, 1.0, val_n)
    T_val, X_val = jnp.meshgrid(t_val_ticks, x_val_ticks)
    t_val_grid = T_val.flatten()
    x_val_grid = X_val.flatten()

    # 用于记录收敛历史数据
    history_steps = []
    history_train_loss = []
    history_val_loss = []

    # 记录点集时空演化历史 (合并背景和自适应点，仅用于 rar 模式)
    points_history = {
        "t_step_0": jnp.concatenate([t_bg, t_adapt]),
        "x_step_0": jnp.concatenate([x_bg, x_adapt])
    }

    # 编译训练单步更新函数
    @jax.jit
    def train_step(params, opt_state, t_col, x_col):
        loss_val, grads = jax.value_and_grad(loss_fn)(
            params, t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_col, x_col, nu
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    print(f"Starting training loop in '{mode}' mode...")
    for step in range(1, steps + 1):
        # 合并背景点与自适应追踪点，保持总点数 10000 恒定
        t_col = jnp.concatenate([t_bg, t_adapt])
        x_col = jnp.concatenate([x_bg, x_adapt])

        params, opt_state, loss = train_step(params, opt_state, t_col, x_col)
        
        # 每 100 步记录一次全局物理残差与训练损失
        if step % 100 == 0:
            f_val_pred = residual_v(params, t_val_grid, x_val_grid, nu)
            val_loss = jnp.mean(f_val_pred ** 2)
            
            history_steps.append(step)
            history_train_loss.append(float(loss))
            history_val_loss.append(float(val_loss))

        if step % 1000 == 0:
            print(f"Step {step}/{steps} - Loss: {loss:.5e}")
            
            if mode == "rar":
                # 自适应重采样：仅在自适应组 N_adapt 内部进行淘汰替换
                key, subkey = jax.random.split(key)
                t_adapt, x_adapt = adaptive_resample(
                    params, t_adapt, x_adapt, nu, subkey,
                    top_k=1500,  # 每次更新 5000 点中的 1500 点
                    noise_std=0.005
                )
                
                # 每 3000 步记录合并后的完整坐标快照
                if step % 3000 == 0:
                    points_history[f"t_step_{step}"] = jnp.concatenate([t_bg, t_adapt])
                    points_history[f"x_step_{step}"] = jnp.concatenate([x_bg, x_adapt])

    # 保存 Loss 收敛历史
    history_path = f"loss_history_{mode}.npz"
    np.savez(
        history_path,
        steps=np.array(history_steps),
        train_loss=np.array(history_train_loss),
        val_loss=np.array(history_val_loss)
    )
    print(f"Saved loss history to {history_path}")

    # 保存模型权重参数
    save_params(params, f"pinn_params_{mode}.npz")
    if mode == "rar":
        save_collocation_history(points_history, "collocation_history.npz")


if __name__ == "__main__":
    main()
