# Physics‑Informed Neural Networks for Non‑Newtonian Channel Flow

This project investigates two physics‑informed neural network frameworks for solving **steady incompressible non‑Newtonian flow in a two‑dimensional channel**:

- **Standard Physics‑Informed Neural Network (PINN)** based on a Fully Connected Neural Network (FCNN)
- **Physics‑Informed Graph Neural Network (PINN‑GNN)** that incorporates mesh connectivity

Both approaches aim to approximate the **velocity field** and **pressure distribution** while satisfying the governing physical laws of fluid mechanics.

---

# 1. Problem Description

The goal is to predict the velocity components $(u,v)$ and pressure $p$ for a steady incompressible non‑Newtonian flow inside a rectangular channel.

**Domain dimensions**

- Channel length: $L = 10.0\,m$
- Channel height: $H = 1.0\,m$

The models learn the solution field directly from the governing equations and boundary conditions without relying on traditional numerical solvers.

---

# 2. Governing Equations

The flow is governed by the **steady incompressible Navier–Stokes equations**.

## 2.1 Continuity Equation

$$
\nabla \cdot \mathbf{u} = 0
$$

## 2.2 Momentum Equation

$$
\rho (\mathbf{u} \cdot \nabla \mathbf{u}) =
- \nabla p + \nabla \cdot \boldsymbol{\tau}
$$

### Expanded Form

**x‑momentum**

$$
\rho \left(u \frac{\partial u}{\partial x} +v \frac{\partial u}{\partial y}\right)+\frac{\partial p}{\partial x}-
\left(
\frac{\partial \tau_{xx}}{\partial x} +
\frac{\partial \tau_{xy}}{\partial y}
\right)
= 0
$$

**y‑momentum**

$$
\rho \left(u \frac{\partial v}{\partial x} +v \frac{\partial v}{\partial y}\right)+\frac{\partial p}{\partial y}-
\left(
\frac{\partial \tau_{xy}}{\partial x} +
\frac{\partial \tau_{yy}}{\partial y}
\right)
= 0
$$

---

# 3. Non‑Newtonian Constitutive Model

The fluid viscosity is modeled using the **Carreau–Yasuda model**, which captures shear‑thinning behavior commonly observed in complex fluids.

The viscous stress tensor is defined as

$$
\boldsymbol{\tau} = \eta(\dot{\gamma}) \, \dot{\boldsymbol{\gamma}}
$$

## 3.1 Carreau–Yasuda Viscosity Model

$$
\eta(\dot{\gamma}) =
\eta_\infty +
(\eta_0 - \eta_\infty)
\left[
1 + (\lambda \dot{\gamma})^a
\right]^{\frac{n-1}{a}}
$$

where the viscosity depends on the magnitude of the **shear rate**.

## 3.2 Shear Rate Magnitude

$$
\dot{\gamma} =
\sqrt{
2\left(\frac{\partial u}{\partial x}\right)^2 +
2\left(\frac{\partial v}{\partial y}\right)^2 +
\left(
\frac{\partial u}{\partial y} +
\frac{\partial v}{\partial x}
\right)^2
}
$$

---

# 4. Physical Parameters

| Parameter | Value | Description |
|-----------|------|-------------|
| $\rho$ | 1.0 | Fluid density |
| $\eta_0$ | 0.01 | Zero‑shear viscosity |
| $\eta_\infty$ | 0.005 | Infinite‑shear viscosity |
| $\lambda$ | 1.0 | Relaxation time |
| $n$ | 0.5 | Power‑law index |
| $a$ | 2.0 | Yasuda parameter |

---

# 5. Boundary Conditions

![Mesh Geometry](Input/mesh_geometry.png)

The computational domain is subject to the following boundary conditions.

### Inlet

A parabolic velocity profile is imposed:

$$
u = 0.5(1.0 - 4y^2), \qquad v = 0
$$

### Walls

A **no‑slip boundary condition** is enforced:

$$
u = 0, \qquad v = 0
$$

### Outlet

The outlet pressure is fixed:

$$
p = 0
$$

Boundary conditions are enforced differently depending on the learning framework, either as **soft constraints** within the loss function or as **hard constraints** embedded in the network architecture.

---

# 6. Modeling Approaches

Two physics‑informed neural architectures are considered.

---

# 6.1 Standard Physics‑Informed Neural Network (PINN)

In the classical PINN framework, the solution fields are approximated using a **fully connected neural network**.

## Network Architecture

- **Input:** spatial coordinates $(x,y)$
- **Output:** velocity components $(u,v)$ and pressure $p$
- **Model type:** Multilayer Perceptron (MLP)
- **Hidden layers:** three layers $[64,64,64]$
- **Hardware:** GPU (CUDA) when available, otherwise CPU

## Automatic Differentiation

All spatial derivatives required for the PDE residuals are computed using **PyTorch automatic differentiation**:

- Gradients are taken with respect to $(x,y)$
- `create_graph=True` enables higher‑order derivatives needed for stress divergence

## Boundary Condition Enforcement

Boundary conditions are implemented as **soft constraints** using penalty terms within the loss function.

## Loss Function

The total loss consists of a **data term** and a **physics term**:

$$
\mathcal{L} = \mathcal{L}_{data} + \mathcal{L}_{pde}
$$

where

$$
\mathcal{L}_{data} =
\mathcal{L}_u +
\mathcal{L}_v +
\mathcal{L}_p
$$

$$
\mathcal{L}_{pde} =
\mathcal{L}_{continuity} +
\mathcal{L}_{momentum_x} +
\mathcal{L}_{momentum_y}
$$

All components are computed using **Mean Squared Error (MSE)**.

## Training Procedure

Optimization is performed in two stages:

**Stage 1 — Adam optimizer**

- Initial learning rate: $10^{-3}$
- Provides stable initial convergence

**Stage 2 — L‑BFGS optimizer**

- Learning rate: $0.01$
- Maximum iterations: 2000
- Provides quasi‑Newton refinement for higher accuracy

---

# 6.2 Physics‑Informed Graph Neural Network (PINN‑GNN)

The second approach integrates **mesh topology information** using a graph neural network.

Unlike the FCNN‑based PINN, this method directly exploits the **connectivity of the computational mesh**.

## Network Architecture

- **Input:** node coordinates $(x,y)$ with Fourier feature embedding
- **Graph representation:** normalized adjacency matrix from mesh connectivity
- **Hidden layers:** three gated graph layers $[64,64,64]$
- **Gating mechanism:** dual U–V gates
- **Output:** physical fields $(u,v,p)$

## Gradient Evaluation

Spatial derivatives are computed using **least‑squares gradient reconstruction** based on neighboring nodes.

### Reconstruction Procedure

1. Identify neighboring nodes through mesh connectivity  
2. Construct local coordinate difference matrices  
3. Solve a least‑squares problem to estimate spatial gradients  

## Boundary Condition Enforcement

Boundary conditions are implemented as **hard constraints** directly within the network outputs.

## Physics‑Informed Loss

The governing equations are enforced at interior nodes through the PDE residual loss:

$$
\mathcal{L} =
\mathcal{L}_{continuity}
+
\mathcal{L}_{momentum_x}
+
\mathcal{L}_{momentum_y}
$$

Each term corresponds to the **MSE of the corresponding PDE residual**.

## Training Strategy

Training follows the same two‑stage optimization procedure:

1. **Adam optimizer** for initial convergence  
2. **L‑BFGS optimizer** for high‑accuracy refinement

---

# 6.3 Ill‑Posed Formulation and Inlet Profile Inference

Beyond the standard forward problem, the framework can also address an **inverse (ill‑posed) problem**.

In this formulation, the **inlet velocity profile is assumed to be unknown**. Instead of prescribing the analytical inlet condition, the model is provided with only **sparse observational data**, such as a limited number of velocity measurements inside the domain.

During training, the network simultaneously:

- enforces the **Navier–Stokes equations**
- matches the **available measurement data**
- reconstructs the **unknown inlet velocity distribution**

The governing physics acts as a strong regularization mechanism, allowing both **PINN (FCNN)** and **PINN‑GNN** to recover the inlet velocity profile even when only limited data are available.

This capability highlights the effectiveness of physics‑informed neural networks for solving **inverse problems and ill‑posed physical systems**.

---

# 7. Post‑Processing and Validation

The predicted fields are validated against **CFD reference solutions**.

The post‑processing pipeline generates:

- Velocity contour plots $(u,v)$
- Pressure field visualization
- Effective viscosity distribution $\eta$
- Relative error maps
- $L_2$ error norms

Additionally, the **predicted inlet velocity profile** is compared with the analytical parabolic profile to assess reconstruction accuracy.
