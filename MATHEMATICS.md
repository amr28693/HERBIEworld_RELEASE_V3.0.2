# MATHEMATICS.md — The Complete Equations of HERBIE World

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     ███╗   ███╗ █████╗ ████████╗██╗  ██╗███████╗███╗   ███╗ █████╗       ║
║     ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║██╔════╝████╗ ████║██╔══██╗      ║
║     ██╔████╔██║███████║   ██║   ███████║█████╗  ██╔████╔██║███████║      ║
║     ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║██╔══╝  ██║╚██╔╝██║██╔══██║      ║
║     ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║███████╗██║ ╚═╝ ██║██║  ██║      ║
║     ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝      ║
║                                                                           ║
║              Full Wave Equations for Artificial Life                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Contents

1. [The Nonlinear Schrödinger Equation (NLSE)](#the-nonlinear-schrödinger-equation)
2. [The Korteweg-de Vries Equation (KdV)](#the-korteweg-de-vries-equation)
3. [The Complete Herbie: All Equations](#the-complete-herbie)
4. [Split-Step Fourier Method](#numerical-methods)
5. [Derived Quantities](#derived-quantities)
6. [Coupling Terms](#coupling-terms)

---

## The Nonlinear Schrödinger Equation

The NLSE is the foundation of all creature dynamics:

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │   i ∂ψ/∂t = -½ ∇²ψ + V(x)ψ + g|ψ|²ψ │
                    │                                     │
                    └─────────────────────────────────────┘
```

### Terms

| Term | Meaning | Physical Interpretation |
|------|---------|------------------------|
| i ∂ψ/∂t | Time evolution | How the wavefunction changes |
| -½ ∇²ψ | Kinetic/dispersion | Tendency to spread out |
| V(x)ψ | Potential | External forces (food, boundaries) |
| g\|ψ\|²ψ | Nonlinearity | Self-interaction (focusing/defocusing) |

### Regimes

- **g > 0**: Self-focusing (attractive) — soliton formation
- **g < 0**: Self-defocusing (repulsive) — dispersion
- **g = 0**: Linear Schrödinger — pure spreading

---

## The Korteweg-de Vries Equation

The KdV equation governs soliton signaling in neural channels:

```
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │   ∂u/∂t + 6u(∂u/∂x) + ∂³u/∂x³ = 0  │
                    │                                     │
                    └─────────────────────────────────────┘
```

### Terms

| Term | Meaning | Physical Interpretation |
|------|---------|------------------------|
| ∂u/∂t | Time evolution | Signal propagation |
| 6u(∂u/∂x) | Nonlinear steepening | Larger signals travel faster |
| ∂³u/∂x³ | Dispersion | Spreading tendency |

### The KdV Soliton Solution

```
    u(x,t) = (c/2) sech²[ √(c/12) (x - ct - x₀) ]
```

where c is the soliton velocity. Key property: **amplitude determines speed**.

---

## The Complete Herbie

A Herbie creature contains multiple coupled dynamical systems:

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │                        THE COMPLETE HERBIE                      │
    │                                                                 │
    │   ┌───────────────┐                                             │
    │   │  TORUS BRAIN  │ ← Central pattern generator                 │
    │   │    ψ_torus    │                                             │
    │   └───────┬───────┘                                             │
    │           │                                                     │
    │     ┌─────┴─────┐                                               │
    │     │           │                                               │
    │     ▼           ▼                                               │
    │  ┌──────┐   ┌──────┐                                            │
    │  │ KdV  │   │ KdV  │  ← Afferent/Efferent channels              │
    │  │Affer │   │Effer │                                            │
    │  └──┬───┘   └───┬──┘                                            │
    │     │           │                                               │
    │     ▼           ▼                                               │
    │   ┌───────────────┐                                             │
    │   │  BODY FIELD   │ ← Physical embodiment                       │
    │   │    ψ_body     │                                             │
    │   └───────┬───────┘                                             │
    │           │                                                     │
    │     ┌─────┴─────┬─────────┐                                     │
    │     │           │         │                                     │
    │     ▼           ▼         ▼                                     │
    │  ┌──────┐   ┌──────┐  ┌──────┐                                  │
    │  │ LIMB │   │ LIMB │  │ LIMB │  ← KdV appendages                │
    │  │  u₁  │   │  u₂  │  │  u₃  │                                  │
    │  └──────┘   └──────┘  └──────┘                                  │
    │                                                                 │
    │   ┌───────────────┐                                             │
    │   │   SKELETON    │ ← Vibrational modes                         │
    │   │   (damped)    │                                             │
    │   └───────────────┘                                             │
    │                                                                 │
    │   ┌───────────────┐   ┌───────────────┐                         │
    │   │  LEFT HAND    │   │  RIGHT HAND   │  ← Gripper KdV          │
    │   │    u_L(x)     │   │    u_R(x)     │                         │
    │   └───────────────┘   └───────────────┘                         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Torus Brain Equations

### The Ring NLSE

The torus brain lives on a 1D ring (θ ∈ [0, 2π]):

```
    i ∂ψ/∂t = -½ ∂²ψ/∂θ² + g|ψ|²ψ + A(θ,t)
```

where:
- θ is the angular position on the ring
- g is the nonlinearity (modulated by state)
- A(θ,t) is the afferent input term

### Nonlinearity Modulation

```
    g_target = 1.2 + 0.6×audio - 0.5×hunger - 1.2×dream_depth
    
    g(t+1) = α×g(t) + (1-α)×g_target    where α = 0.95
```

### Circulation (Phase Winding)

```
                       1    2π
    circulation(t) = ───── ∮  (∂φ/∂θ) dθ
                      2π   0
    
    where φ = arg(ψ)
```

Circulation measures net angular momentum of the wavefunction.

### Coherence (Focus)

```
                      N
    coherence = N × Σ (Iᵢ / Σᵢ Iᵢ)²
                     i=1
    
    where I = |ψ|²
```

High coherence = localized density = focused attention.

### Arousal

```
    arousal = clip( (coherence/8) × (0.5 + 0.5×tanh(|circulation|)), 0, 1 )
```

### Directional Bias

```
    mean_x = Σᵢ Iᵢ×cos(θᵢ) / Σᵢ Iᵢ
    mean_y = Σᵢ Iᵢ×sin(θᵢ) / Σᵢ Iᵢ
    
    direction = arctan2(mean_y, mean_x)
    strength = √(mean_x² + mean_y²)
```

---

## Body Field Equations

### The 2D NLSE with Potential

```
    i ∂ψ/∂t = -½ ∇²ψ + V(x,y)ψ + g|ψ|²ψ
```

### The Potential Function

```
    V(x,y) = V_basins + V_skin + V_objects + V_ghosts
```

#### Body Basins (Anatomical Structure)

```
    V_basins = Σ -Aᵢ × exp(-|r - rᵢ|² / 2σᵢ²)
              basins
    
    Basins: core, head, tail, limb_origins
```

#### Skin Boundary (Containment)

```
    V_skin = 7.0 × exp(-(d_boundary)² / 0.5)
    
    where d_boundary = distance to body edge
```

This creates a repulsive barrier preventing wavefunction escape.

#### Object Potential (Food/Barriers)

```
    For food (compliance > 0.5):
        V_food = -strength × exp(-d² / radius²)
        
        where strength = compliance × 0.6 × (1 + energy/max_energy)
                       × (1 + 1.5 × hunger)
        
        Capped at 5.0 (below skin strength 7.0)
    
    For barriers (compliance < 0.5):
        V_barrier = +(1 - compliance) × 5.0 × exp(-d² / 0.6)
```

#### Ghost Potential

```
    V_ghost = -influence × (coherence - 0.3) × 2.0 × exp(-d² / 25)
    
    Coherent ghosts attract, dispersing ghosts repel
```

### Hunger-Modulated Coupling

```
    V_effective = V_total × (0.4 + 0.4 × hunger)
    
    Max coupling: 0.8 (to preserve skin containment)
```

### Momentum Extraction

```
    p_x = Im( Σ ψ* × (∂ψ/∂x) )
    p_y = Im( Σ ψ* × (∂ψ/∂y) )
    
    momentum = [p_x, p_y] × normalization
```

---

## KdV Channel Equations

### Neural Channel Dynamics

```
    ∂u/∂t = -6u(∂u/∂x) - ∂³u/∂x³
```

### Soliton Nucleation

When a signal is injected:
```
    u(x) += amplitude × exp(-((x - x₀)/width)²)
    
    where width = 0.35 / (amplitude + 0.25)
```

### Arrival Detection

A signal "arrives" when:
```
    u(x_end) > threshold  AND  peak is traveling toward end
```

---

## Limb Equations

### Limb KdV

Each limb is a 1D KdV system:

```
    ∂u/∂t + g×u×(∂u/∂x) + dispersion×(∂³u/∂x³) = 0
```

### Efferent Injection

Motor commands from torus inject at limb base:
```
    u[0:3] += amplitude × exp(-i/2) × cos(phase)
```

### Pulse Tracking

```
    pulse_position = Σ xᵢ×|uᵢ| / Σ |uᵢ|
    pulse_amplitude = max(|u|)
```

### Limb Thrust

```
    thrust = pulse_amplitude × |angular_velocity| × 0.4 × direction
```

---

## Gripper (Hand) Equations

### Gripper KdV

```
    ∂u/∂t + g×u×(∂u/∂x) + dispersion×(∂³u/∂x³) = 0
    
    g = 0.8, dispersion = 0.1, damping = 0.97
```

### Grip Reflex

```
    tip_activation = mean(|u[-5:]|)
    
    if tip_activation > 0.1:
        attempt_grip()
```

### Arm Swing Physics

```
    wave_momentum = Σ uᵢ × xᵢ
    angular_accel = wave_momentum × 0.02
    spring_force = (base_angle - angle) × 0.1
    
    angular_velocity += angular_accel + spring_force - damping×velocity
    angle += angular_velocity
```

---

## Metabolism Equations

### Hunger Dynamics

```
    age_factor = 1.0 + 0.4 × (age / max_age)
    base_rate = 0.0003 × age_factor × metabolism_rate
    movement_cost = 0.00008 × velocity² × metabolism_rate
    
    hunger += base_rate + movement_cost
```

### Eating Reward

```
    contact = max(proximity_contact, body_contact × 2.0)
    
    where proximity_contact = exp(-(dist/reach_radius)²)
          reach_radius = 5.0 + object_size
    
    if contact > 0.03 and compliance > 0.5:
        extraction = min(contact × 1.5, energy)
        reward = extraction × (compliance - 0.25)
        
    hunger -= reward × 0.4 × vigor
```

---

## Ghost Equations

### Ghost NLSE (Free Evolution)

```
    i ∂ψ/∂t = -½ ∂²ψ/∂θ² + g|ψ|²ψ
    
    No external input. No body. Just wavefunction.
```

### Ghost Movement

```
    circulation = mean(diff(unwrap(arg(ψ))))
    
    speed = |circulation| × 0.12
    angle = age × circulation × 0.1
    
    velocity = 0.95×velocity + [cos(angle), sin(angle)] × speed × 0.05
    position += velocity
```

### Ghost Dispersal

Ghost dies when:
- Coherence too low (energy spread thin)
- Singularity (NaN/Inf from blowup)

---

## Numerical Methods

### Split-Step Fourier Method

For NLSE: i∂ψ/∂t = -½∇²ψ + g|ψ|²ψ + Vψ

```
    1. Half-step nonlinear:
       ψ ← ψ × exp(-i × g × |ψ|² × Δt/2)
    
    2. Full-step linear (in Fourier space):
       ψ̂ ← FFT(ψ)
       ψ̂ ← ψ̂ × exp(-i × k² × Δt/2)
       ψ ← IFFT(ψ̂)
    
    3. Half-step nonlinear + potential:
       ψ ← ψ × exp(-i × g × |ψ|² × Δt/2)
       ψ ← ψ × exp(-i × V × Δt)
```

### For KdV

```
    1. Linear step (dispersion):
       û ← FFT(u)
       û ← û × exp(-i × k³ × Δt)
       u ← IFFT(û)
    
    2. Nonlinear step:
       du/dx ← IFFT(i × k × FFT(u))
       u ← u - 6 × u × (du/dx) × Δt
```

---

## Coupling Summary

### Signal Flow

```
    Audio Input ──────────────────────┐
                                      │
    ┌─────────────────────────────────▼─────────────────────────────────┐
    │                           CREATURE                                 │
    │                                                                    │
    │   Environment ──→ Afferent KdV ──→ TORUS ──→ Efferent KdV ──→ Body │
    │                         │            │              │              │
    │                         └────────────┼──────────────┘              │
    │                                      │                             │
    │                                      ▼                             │
    │                               BODY FIELD ←── Object Potentials    │
    │                                      │                             │
    │                               ┌──────┼──────┐                      │
    │                               │      │      │                      │
    │                               ▼      ▼      ▼                      │
    │                             LIMBS  HANDS  SKELETON                 │
    │                               │      │                             │
    │                               └──────┴──────→ Movement, Actions    │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              World Interaction
```

---

## Constants

```python
# Torus Brain
N_torus = 64                    # Ring discretization
dt = 0.05                       # Time step
G_SMOOTH = 0.95                 # Nonlinearity smoothing

# Body Field
Nx, Ny = 32, 32                 # Body grid
BODY_L = 16.0                   # Body field extent

# KdV Channels
N_kdv = 64                      # Channel discretization
kdv_length = 2π                 # Channel length

# Metabolism
base_hunger_rate = 0.0003
movement_cost_coef = 0.00008
eating_reduction_coef = 0.4

# Containment
skin_strength = 7.0
max_food_potential = 5.0

# EM Field (V2.6)
em_resolution = 32              # EM grid
c = 2.0                         # Speed of light (grid/timestep)
epsilon_0 = 1.0                 # Permittivity
coupling_strength = 0.3         # EM-creature coupling
```

---

## Electromagnetic Field (V2.6)

### Maxwell's Equations in 2D

With magnetic field B perpendicular to the plane (B = Bz ẑ):

```
    ∂Ex/∂t = c²(∂Bz/∂y) - Jx/ε₀      (Ampère-Maxwell, x)
    ∂Ey/∂t = -c²(∂Bz/∂x) - Jy/ε₀     (Ampère-Maxwell, y)
    ∂Bz/∂t = ∂Ex/∂y - ∂Ey/∂x         (Faraday)
    
    ∇·E = ρ/ε₀                        (Gauss - enforced spectrally)
```

### Creature Charge

Creatures source the EM field as charge distributions:

```
    charge = 0.5 + 0.5 × min(coherence/5, 1) × concentration_factor
    
    where coherence = torus coherence
          concentration_factor = body field localization
    
    ρ(x,y) = Σ charge_i × Gaussian(x - pos_i, σ)
            creatures
```

### Current from Motion

Moving charges create current:

```
    J(x,y) = Σ charge_i × velocity_i × Gaussian(x - pos_i, σ)
            creatures
```

### Lorentz Force

Creatures feel the field:

```
    F = q(E + v × B)
    
    In 2D: Fx = q(Ex + vy × Bz)
           Fy = q(Ey - vx × Bz)
```

### Ghost EM Contributions

Ghosts create subtle EM disturbances:

```
    ghost_charge = coherence × 0.2
    
    Circulation creates rotating current:
    Jx += circulation × (-dy) × Gaussian / (r² + 1)
    Jy += circulation × dx × Gaussian / (r² + 1)
```

### Operating Modes

```
    OFF          - No EM computation
    ELECTROSTATIC - Solve ∇²φ = -ρ/ε₀, E = -∇φ (cheapest)
    FULL_MAXWELL  - Full FDTD evolution (most accurate)
    HYBRID        - Low-res Maxwell + interpolation
```

### Environment Variables

```bash
    HERBIE_EM_MODE=ELECTROSTATIC   # OFF, ELECTROSTATIC, FULL_MAXWELL
    HERBIE_EM_RESOLUTION=32        # Grid resolution (32 = fast, 64 = detailed)
```

---

```
    "The creature is not simulated by these equations.
     The creature IS these equations."
    
                                        — HERBIE World Mathematics
```
