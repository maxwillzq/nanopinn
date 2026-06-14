import jax
import jax.numpy as jnp
import optax
import numpy as np

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
    """将参数序列化并保存为 npz 文件"""
    flat_params = {}
    for idx, layer in enumerate(params):
        flat_params[f"W_{idx}"] = layer['W']
        flat_params[f"b_{idx}"] = layer['b']
    jnp.savez(path, **flat_params)
    print(f"Saved parameters to {path}")

def main():
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

    # 3. 物理控制共轭点采样
    N_col = 10000
    k1, k2 = jax.random.split(key)
    t_col = jax.random.uniform(k1, (N_col,), minval=0.0, maxval=1.0)
    x_col = jax.random.uniform(k2, (N_col,), minval=-1.0, maxval=1.0)

    # 编译训练单步更新函数
    @jax.jit
    def train_step(params, opt_state, t_col, x_col):
        loss_val, grads = jax.value_and_grad(loss_fn)(
            params, t_ic, x_ic, u_ic, t_bc, x_bc, u_bc, t_col, x_col, nu
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    print("Starting training loop...")
    for step in range(1, steps + 1):
        params, opt_state, loss = train_step(params, opt_state, t_col, x_col)
        if step % 1000 == 0:
            print(f"Step {step}/{steps} - Loss: {loss:.5e}")

    save_params(params, "pinn_params.npz")

if __name__ == "__main__":
    main()
