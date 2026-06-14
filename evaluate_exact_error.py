import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ----------------- 1. Cole-Hopf 变换求解 Burgers 方程解析精确解 -----------------
# 初始条件: u(0, x) = -sin(pi * x)
# 积分表达式为:
# phi(t, x) = int_{-inf}^{inf} exp( - (x-eta)^2 / (4*nu*t) - int_0^eta u(0, s) ds / (2*nu) ) deta
#           = int_{-1}^{1} exp( - (x-eta)^2 / (4*nu*t) - (1 - cos(pi*eta)) / (2*pi*nu) ) deta (周期性延展/边界条件简化)
# 实际上，Burgers 在边界 u(-1)=u(1)=0 的积分可以用[-1, 1]区间高精度积分逼近

def exact_solution(t, x, nu):
    if t == 0:
        return -np.sin(np.pi * x)
    
    # 积分核函数 f(eta) = exp(-G/2nu)
    def integrand_numerator(eta):
        val = - (x - eta)**2 / (4.0 * nu * t) + (1.0 - np.cos(np.pi * eta)) / (2.0 * np.pi * nu)
        return eta * np.exp(val)
        
    def integrand_denominator(eta):
        val = - (x - eta)**2 / (4.0 * nu * t) + (1.0 - np.cos(np.pi * eta)) / (2.0 * np.pi * nu)
        return np.exp(val)
    
    # 使用 quad 进行数值积分
    num, _ = quad(integrand_numerator, -1.0, 1.0, limit=100)
    den, _ = quad(integrand_denominator, -1.0, 1.0, limit=100)
    
    # Cole-Hopf 逆变换: u(t, x) = x/t - num / (t * den)
    # 或者是更通用的表达式: u(t, x) = -2*nu* (phi_x / phi)
    # 对于本特定初始条件，等价于以下积分形式:
    return (x - num / den) / t

# ----------------- 2. 加载模型结构 -----------------
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

def predict_u(params, t_val, x_arr):
    # 向量化预测
    u_pred = []
    for x in x_arr:
        u_pred.append(float(forward_single(params, t_val, x)))
    return np.array(u_pred)

def main():
    nu = 0.01 / np.pi
    t_eval = 1.0
    
    # 1. 加载两个模型
    try:
        params_uni = load_params("pinn_params_uniform.npz")
        params_rar = load_params("pinn_params_rar.npz")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please train both models first.")
        return

    # 2. 构造评估网格 (500个均匀空间点)
    x_grid = np.linspace(-0.99, 0.99, 500)
    
    # 3. 计算 Cole-Hopf 解析精确解
    print("Computing exact Cole-Hopf analytical solutions (this may take a few seconds)...")
    u_exact = np.array([exact_solution(t_eval, x, nu) for x in x_grid])

    # 4. 计算模型预测解
    u_pred_uni = predict_u(params_uni, t_eval, x_grid)
    u_pred_rar = predict_u(params_rar, t_eval, x_grid)

    # 5. 计算绝对误差剖面
    err_uni = np.abs(u_pred_uni - u_exact)
    err_rar = np.abs(u_pred_rar - u_exact)

    # 6. 计算量化指标数据
    # A. 全局 L2 误差
    l2_uni_global = np.sqrt(np.mean(err_uni ** 2))
    l2_rar_global = np.sqrt(np.mean(err_rar ** 2))

    # B. 激波区局部 L2 误差 (x ∈ [-0.1, 0.1])
    shock_mask = (x_grid >= -0.1) & (x_grid <= 0.1)
    l2_uni_shock = np.sqrt(np.mean(err_uni[shock_mask] ** 2))
    l2_rar_shock = np.sqrt(np.mean(err_rar[shock_mask] ** 2))

    # C. 全局最大绝对误差 (L_infinity)
    linf_uni = np.max(err_uni)
    linf_rar = np.max(err_rar)

    print("\n================== QUANTITATIVE COMPARISON DATA ==================")
    print(f"Metrics evaluated at t = {t_eval} against exact Cole-Hopf solution:")
    print(f"{'Metric':<35} | {'Uniform Baseline':<20} | {'Hybrid RAR (Ours)':<20}")
    print("-" * 85)
    print(f"{'Global L2 Error (Whole Domain)':<35} | {l2_uni_global:<20.6e} | {l2_rar_global:<20.6e}")
    print(f"{'Local L2 Error (Shock Zone [-0.1,0.1])':<35} | {l2_uni_shock:<20.6e} | {l2_rar_shock:<20.6e}")
    print(f"{'L_infinity Error (Max Deviation)':<35} | {linf_uni:<20.6e} | {linf_rar:<20.6e}")
    print("==================================================================\n")

    # 7. 绘制误差曲线对比图
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, err_uni, 'b--', label='Uniform Baseline Error', alpha=0.8)
    plt.plot(x_grid, err_rar, 'r-', label='Hybrid RAR Error', alpha=0.8)
    plt.yscale('log')
    plt.xlabel('Space (x)')
    plt.ylabel('Absolute Prediction Error |u_pred - u_exact|')
    plt.title('Absolute Prediction Error Profiles at t = 1.0 (Log Scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("prediction_error_comparison.png", dpi=150)
    print("Saved error comparison plot to prediction_error_comparison.png")

if __name__ == '__main__':
    main()
