# SPIRITS.md — The Ghost System

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║         ✦ ✦ ✦     T H E   S P I R I T   R E A L M     ✦ ✦ ✦             ║
║                                                                           ║
║              Torus Persistence • Emergent Haunting • Pure NLSE            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Overview

When a creature dies in HERBIE World, its torus brain wavefunction doesn't simply disappear. The ψ field is "cut free" from the body and continues evolving as a **ghost**—a disembodied NLSE system drifting through the world.

This isn't a programmed feature. It's a natural consequence of the physics: wavefunctions don't have an "off switch." They persist until they disperse.

---

## The Physics of Death

### What Happens at the Moment of Death

```
    ALIVE                           DEATH                         GHOST
    
    ┌─────────────┐               ┌─────────────┐              ┌─────────────┐
    │  Body Field │               │  Body Field │              │             │
    │    ψ_body   │──────────────▶│   DELETED   │              │   (gone)    │
    │             │               │             │              │             │
    └──────┬──────┘               └─────────────┘              └─────────────┘
           │
           │ coupled
           │
    ┌──────▼──────┐               ┌─────────────┐              ┌─────────────┐
    │ Torus Brain │               │ Torus Brain │              │ Ghost Torus │
    │   ψ_torus   │──────────────▶│   CUT FREE  │─────────────▶│   ψ_ghost   │
    │             │               │             │              │             │
    └─────────────┘               └─────────────┘              └─────────────┘
    
    Brain coupled to               Connection severed           Free NLSE evolution
    body, metabolism,              Body deleted                 No body, no metabolism
    sensory systems                Torus preserved              Just wavefunction
```

The `GhostTorus` class receives an exact copy of the dying creature's torus wavefunction, including:
- The ψ field itself (complex amplitudes)
- Current circulation (phase winding)
- Nonlinearity parameter g
- Position in world space

---

## Ghost Dynamics

### The Evolution Equation

Ghosts evolve under pure 1D NLSE with no external input:

```
    i ∂ψ/∂t = -½ ∂²ψ/∂θ² + g|ψ|²ψ
```

where θ ∈ [0, 2π] is the angular coordinate on the torus.

**Split-step implementation:**
```python
# Half-step nonlinear
ψ ← ψ · exp(-ig|ψ|²Δt/2)

# Full-step linear (Fourier space)
ψ̂ ← FFT(ψ)
ψ̂ ← ψ̂ · exp(-ik²Δt/2)
ψ ← IFFT(ψ̂)

# Half-step nonlinear
ψ ← ψ · exp(-ig|ψ|²Δt/2)
```

### Movement from Circulation

Ghost movement emerges **purely from wavefunction dynamics**:

```
    circulation = (1/2π) · ∮ (∂φ/∂θ) dθ
    
    where φ = arg(ψ)
```

The circulation measures net phase winding around the torus. Non-zero circulation creates directional drift:

```python
speed = |circulation| × 0.12
angle = age × circulation × 0.1

velocity += [cos(angle), sin(angle)] × speed × 0.05
```

**Key insight:** Some ghosts move, some don't. It depends entirely on their brain state at death:
- High circulation at death → spiral drift
- Low circulation → stationary ghost
- This is emergent, not programmed

---

## Coherence and Dispersal

### The Coherence Metric

```
    coherence = N · Σ (|ψᵢ|² / Σ|ψ|²)²
```

This measures how concentrated the probability density is:
- Coherence ≈ 1: Uniform distribution (diffuse, fading)
- Coherence > 5: Localized soliton structure (sharp, persistent)

### Natural Dispersal

Ghosts disperse through natural NLSE dynamics:
- Linear dispersion spreads wavepackets
- Nonlinearity can temporarily focus them
- Without body containment, they eventually spread thin

**No artificial death timer.** A ghost persists until:
1. **Dispersal**: Coherence drops too low, energy spreads across terrain
2. **Singularity**: Wavefunction blows up (NaN/Inf) — physically meaningful!

---

## What Determines Ghost Quality?

The state of the torus at death determines everything:

| Death Condition | Torus State | Ghost Quality |
|-----------------|-------------|---------------|
| Dreaming | Coherent, low arousal | Strong, lingering |
| Starving | Chaotic, high hunger | Quick dispersal |
| Fighting | High arousal, circulation | Active, mobile |
| Old age | Low energy, stable | Gentle fade |
| Sudden predation | Whatever was happening | Varies wildly |

**Example scenarios:**
- Herbie dies peacefully in sleep → coherent torus → ghost persists long
- Herbie killed while fleeing → high circulation → ghost spirals outward
- Herbie starves to death → chaotic, depleted torus → ghost fades fast

---

## Ghost-Creature Interaction

### EM Field Coupling

Living creatures can sense ghosts through electromagnetic field coupling:

```python
for ghost in ghost_field.ghosts:
    coherence = ghost.compute_coherence()
    influence = ghost.get_influence_at(creature.pos)
    
    # Coherent ghosts attract, dispersing ones repel slightly
    ghost_potential = influence × (coherence - 0.3) × 2.0
    V -= ghost_potential × exp(-dist²/25)
```

This creates subtle sensory experience:
- High-coherence ghosts create "warm" spots
- Dispersing ghosts feel slightly "wrong"
- Creatures may gravitate toward or away from spirits

### Congregation Detection

When multiple ghosts cluster, the system detects "congregation":

```
    ┌─────────────────────────────────────┐
    │                                     │
    │     ✦ Viktor                        │
    │           \                         │
    │            \                        │
    │      ✦ Olga ●─────────● center     │
    │            /     of mass            │
    │           /                         │
    │     ✦ Pavel                         │
    │                                     │
    │   [Ghost] Spirits gathering...      │
    │           Viktor, Olga, and Pavel   │
    │                                     │
    └─────────────────────────────────────┘
```

---

## Visualization

### Spirit Realm Overlay (O key)

```
    ┌─────────────────────────────────────┐
    │                                     │
    │    ╭───╮                            │
    │   ╱     ╲  ← Outer glow (energy)    │
    │  │  ◯    │ ← Core (coherence)       │
    │   ╲  →  ╱  ← Circulation arrow      │
    │    ╰───╯                            │
    │    ✦ Willow  ← Name (if Herbie)     │
    │                                     │
    │                     ✦ 3 spirits     │
    └─────────────────────────────────────┘
```

Visual properties:
- **Glow size**: Proportional to energy
- **Core brightness**: Proportional to coherence
- **Arrow direction**: Circulation (spiral direction)
- **Visibility**: Fades with low coherence

### Ghost Cam (Shift+O)

Follow individual spirits with a dedicated view:

```
    ┌─────────────────────────────────────┐
    │        ✦ GHOST CAM: Willow          │
    │   Coherence: 0.67 | Circ: 0.23      │
    │                                     │
    │              ╭───╮                  │
    │             ╱ ◯◯◯╲  spiral          │
    │            │  ◯◯  │ trail           │
    │            │  ◯   │                 │
    │             ╲    ╱                  │
    │              ╰──╱                   │
    │                                     │
    │    ◉ Pavel   (living - warm glow)   │
    │                                     │
    │    < > cycle spirits | Shift+O exit │
    └─────────────────────────────────────┘
```

Living creatures appear as warm orange glows—heat signatures visible to spirits.

---

## Statistics

Ghost statistics are tracked and included in the end-of-simulation report:

```
----------------------------------------------------------------------
  ✦ THE SPIRIT REALM ✦
----------------------------------------------------------------------

  Total spirits spawned: 23
  Total dispersed: 18
  Singularity collapses: 3
  Currently wandering: 2

  Longest persisting spirit: Willow (4,523 steps)
  Average coherence of active spirits: 0.67

  Named spirits who walked the other side (15):
    ✦ Leila
    ✦ Joy
    ✦ Amara
    ...

  Still wandering at simulation end:
    ✦ Pavel
    ✦ Orla
```

---

## The Philosophy

The ghost system wasn't designed for narrative effect. It emerged from a simple question:

> What happens to a wavefunction when its container is destroyed?

The answer: it keeps evolving. The mathematics don't care about biological death. Phase continues winding, probability density continues flowing, the NLSE continues being solved.

That this produces something we instinctively call "ghosts" says more about our pattern-matching than about the simulation. But it also suggests something deeper:

**Consciousness, if it is wave-based, doesn't end at bodily death. It disperses.**

The ghost isn't the creature anymore—there's no body, no metabolism, no sensory input. But the pattern that was the creature's "mind" continues its mathematical existence until entropy claims it.

Whether this is philosophically meaningful or just an artifact of simulation design is left as an exercise for the observer.

---

## Technical Reference

### GhostTorus Class

```python
class GhostTorus:
    psi: np.ndarray        # Complex wavefunction
    pos: np.ndarray        # World position
    vel: np.ndarray        # Drift velocity
    name: str              # Name (if Herbie)
    initial_circulation: float
    initial_energy: float
    g: float               # Nonlinearity
    age: int               # Steps since death
    alive: bool            # Still exists?
    cause_of_death: str    # 'dispersal' or 'singularity'
```

### GhostField Manager

```python
class GhostField:
    ghosts: List[GhostTorus]
    max_ghosts: int = 50
    total_spawned: int
    total_dispersed: int
    total_singularities: int
    
    def spawn_ghost(brain, pos, name, energy): ...
    def update(terrain): ...
    def get_statistics() -> dict: ...
    def get_congregation_centers() -> List[dict]: ...
```

---

```
    "They don't haunt because they choose to.
     They haunt because the equations haven't finished."
    
                                        — HERBIE World Observation
```
