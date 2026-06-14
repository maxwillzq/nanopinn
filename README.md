# nanoPINN 🧠🌊

A minimalist, framework-free, single-file implementation of **Physics-Informed Neural Networks (PINNs)** solving the nonlinear Burgers' equation using **JAX**.

Inspired by Andrej Karpathy's `nanoGPT` philosophy, this repository strips away all the heavy abstraction of high-level neural network libraries (no Flax, no Equinox, no Haiku). It implements everything—weight initialization, MLP forward pass, automatic differentiation of PDEs, and the training loop—from scratch in a single clean Python file.

---

## 🎯 Why This Matters

Standard deep learning frameworks often obscure how PINNs actually enforce physics. By exposing the raw parameter dictionaries (`PyTrees`) and leveraging JAX's functional purity, **code translates directly to math**. 

Instead of treating the partial differential equation (PDE) as a black-box constraint, you can see exactly how `jax.grad` continuously differentiates the neural network's outputs with respect to space and time to formulate the physics loss.

---

## 📐 The Physics: Burgers' Equation

We solve the 1D viscous Burgers' equation, a fundamental PDE in fluid mechanics that models the formation and propagation of sharp **shock waves**:

$$u_t + u u_x - \nu u_{xx} = 0, \quad x \in [-1, 1], \quad t \in [0, 1]$$

### The Challenge
We set the viscosity $\nu = 0.01 / \pi$. At this extremely low viscosity, the nonlinear convective term ($u u_x$) dominates. As time evolves from an initial sine wave $u(0,x) = -\sin(\pi x)$, the wave peaks sprint forward while the troughs slide backward. 

Without friction, this would cause an impossible physical "wave breaking" (gradient catastrophe) at $t = 1/\pi \approx 0.32$. The tiny viscosity term $\nu u_{xx}$ awakens at the very last moment to fight back, creating a **near-vertical, razor-sharp shock wave** at $x=0$. 

Traditional numerical methods (like finite difference) notoriously oscillate and fail here due to the Gibbs phenomenon. **nanoPINN solves it smoothly without a single spatial mesh grid.**

---

## 🛠️ Inside the Code

### 1. Zero-Abstraction MLP Forward Pass
No `nn.Linear`. A layer is just a raw dictionary containing a weight matrix `W` and a bias vector `b`. The forward pass is pure matrix math:

```python
def forward(params, t, x):
    z = jnp.array([[t], [x]])
    for layer in params[:-1]:
        z = jnp.tanh(jnp.dot(layer['W'], z) + layer['b'])
    out = jnp.dot(params[-1]['W'], z) + params[-1]['b']
    return out[0, 0]
```

### 2. Elegant Higher-Order Differentiation
JAX allows us to write partial derivatives exactly like they look on paper. To compute $u_{xx}$ (the second derivative of the network output with respect to $x$), we simply chain `jax.grad`:

```python
u_t = jax.grad(forward, argnums=1)(params, t, x) # ∂u/∂t
u_x = jax.grad(forward, argnums=2)(params, t, x) # ∂u/∂x
u_xx = jax.grad(jax.grad(forward, argnums=2), argnums=2)(params, t, x) # ∂²u/∂x²

residual = u_t + u * u_x - nu * u_xx
```

### 3. XLA Fusion Accelerator
Using `@jax.jit`, the entire training step—including the forward pass, data loss, dual-layer automatic differentiation, and Adam weight updates—is fused and compiled down to optimal CPU/GPU machine code.

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python train.py
```
The model will distribute 10,000 unsupervised collocation points across the space-time domain and optimize the network globally. Within 15,000 steps, you will see the total loss plunge, and the network will learn to reconstruct the sharp physical shock wave.

### Generate Visualization
```bash
python plot.py
```
This will generate `burgers_shock_wave.png` showing the flow field colormap and velocity profiles.

![Burgers Shock Wave](burgers_shock_wave.png)

---

## 📄 License
MIT
