# EMBRYOLOGY.md — Soliton-Based Morphogenesis

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     ███████╗███╗   ███╗██████╗ ██████╗ ██╗   ██╗ ██████╗                 ║
║     ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚██╗ ██╔╝██╔═══██╗                ║
║     █████╗  ██╔████╔██║██████╔╝██████╔╝ ╚████╔╝ ██║   ██║                ║
║     ██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══██╗  ╚██╔╝  ██║   ██║                ║
║     ███████╗██║ ╚═╝ ██║██████╔╝██║  ██║   ██║   ╚██████╔╝                ║
║     ╚══════╝╚═╝     ╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝                 ║
║                                                                           ║
║       From Single Soliton to Complex Organism via Symmetry Breaking       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Overview

Herbie embryos develop through a **soliton-based morphogenesis** process. Starting from a single localized wavefunction (the "zygote"), the embryo evolves through NLSE dynamics, undergoing symmetry breaking events that establish body plan, organ primordia, and trait modifications.

This is not a programmed growth sequence—it's emergent pattern formation in a nonlinear field.

---

## The Developmental Stages

```
    Stage 0: ZYGOTE              Stage 1: CLEAVAGE           Stage 2: MORULA
    (0-10% development)          (10-20%)                    (20-30%)
    
         ●                          ●●                         ●●●
      Single                      ●●                          ●●●●
      soliton                    First                        Solid
                                divisions                      ball
    
    
    Stage 3: BLASTULA           Stage 4: GASTRULA           Stage 5: NEURULA
    (30-45%)                    (45-60%)                    (60-80%)
    
        ○○○                         ◐◐                          ═══
       ○   ○                       ◐  ◐                        ═ ● ═
        ○○○                         ◐◐                          ═══
      Hollow                     Symmetry                     Body plan
      sphere                     breaking                     established
    
    
    Stage 6: ORGANOGENESIS      Stage 7: FETAL              Stage 8: READY
    (80-100%)                   (refinement)                (birth imminent)
    
        ◉═◉                         ●═●                         ●═●
       ═●●●═                       ●●●●●                       ●●●●●
        ◉═◉                         ●═●                         ●═●
      Organ                       Growth &                    Ready for
      primordia                   maturation                  birth
```

---

## The Physics of Development

### The Embryo Field

The embryo is represented as a 2D NLSE field on a 32×32 grid:

```
    i ∂ψ/∂t = -½ ∇²ψ + g|ψ|²ψ
```

where:
- ψ(x,y,t) is the complex embryonic wavefunction
- g is the inherited nonlinearity (from parents)
- Boundary conditions are periodic (torus topology)

### Initial Condition: The Zygote

```python
def initialize_zygote(self):
    """
    Create single Gaussian soliton representing fertilized egg.
    """
    x = np.linspace(-π, π, N)
    y = np.linspace(-π, π, N)
    X, Y = np.meshgrid(x, y)
    
    r² = X² + Y²
    ψ = 1.5 × exp(-r²/0.5) × exp(i × 0)
```

This creates a single, centered, circularly-symmetric wavepacket—the embryonic "cell."

---

## Stage-by-Stage Development

### Stage 0: Zygote (0-10%)

```
    Initial state:
    
         ░░░░░░░░
        ░░░░░░░░░░
       ░░░░████░░░░
       ░░░██████░░░
       ░░░██████░░░
       ░░░░████░░░░
        ░░░░░░░░░░
         ░░░░░░░░
    
    Single localized soliton
    High radial symmetry
    Maximum coherence
```

The field is initialized and begins evolving. No structure yet—pure potential.

### Stage 1: Cleavage (10-20%)

```python
def do_cleavage(self):
    """
    Simulate cell division via soliton splitting.
    """
    # Find current density peak
    I = |ψ|²
    peak_idx = argmax(I)
    
    # Add perturbation to induce splitting
    perturbation = 0.15 × exp(i × random_phase)
    ψ[peak_idx ± offset] += perturbation
```

```
    After cleavage:
    
         ░░░░░░░░
        ░░░░░░░░░░
       ░░░██░░██░░░
       ░░████████░░
       ░░████████░░
       ░░░██░░██░░░
        ░░░░░░░░░░
         ░░░░░░░░
    
    2-cell stage: soliton has split
    Bilateral symmetry emerging
```

### Stage 2: Morula (20-30%)

Continued division creates a solid cluster of density peaks:

```
         ░░░░░░░░
        ░░░░░░░░░░
       ░░██░██░██░░
       ░░████████░░
       ░░████████░░
       ░░██░██░██░░
        ░░░░░░░░░░
         ░░░░░░░░
    
    8-16 "cells" (density peaks)
    Still radially symmetric
```

### Stage 3: Blastula (30-45%)

A cavity forms in the center as density redistributes:

```
         ░░░░░░░░
        ░░██████░░
       ░░██░░░░██░░
       ░░██░░░░██░░
       ░░██░░░░██░░
       ░░░██████░░░
        ░░░░░░░░░░
         ░░░░░░░░
    
    Hollow sphere (blastocoel)
    Density concentrated at boundary
```

### Stage 4: Gastrula (45-60%)

**Critical stage**: Symmetry breaking occurs.

```python
def do_gastrulation(self):
    """
    Break radial symmetry to establish body axis.
    """
    # Environmental + genetic bias
    bias_angle = maternal_stress × 0.5 + random(-0.3, 0.3)
    
    # Apply asymmetric perturbation
    X, Y = meshgrid(...)
    asymmetry = 0.1 × cos(arctan2(Y, X) - bias_angle)
    ψ *= (1 + asymmetry)
    
    self.bilateral_symmetry = measure_bilateral()
```

```
         ░░░░░░░░
        ░░░████░░░
       ░░░██████░░░
       ░░████████░░   ← Primary axis forms
       ░░░██████░░░
        ░░░░████░░░   ← "Head" vs "tail" emerges
         ░░░░░░░░
    
    Bilateral symmetry established
    Anterior-posterior axis defined
```

### Stage 5: Neurula (60-80%)

Body plan crystallizes:

```python
def do_neurulation(self):
    """
    Establish body plan and proto-organ basins.
    """
    # Create dorsal-ventral axis
    dorsal_ventral_bias = ...
    
    # Count emerging basins (proto-organs)
    self.n_basins = count_local_maxima(|ψ|²)
    self.basin_positions = find_basin_centers()
```

```
         ░░░░░░░░
        ░░░░██░░░░
       ░░░██████░░░     ← Head region
       ░░████████░░
       ░══════════░     ← Body axis (notochord analog)
       ░░████████░░
       ░░░██████░░░     ← Tail region
        ░░░░░░░░░░
    
    Clear head/tail differentiation
    Midline structure (neural tube analog)
```

### Stage 6: Organogenesis (80-100%)

Organ primordia form as stable attractor basins:

```
         ░░░░░░░░
        ░░◉░██░◉░░      ← Sensory organs
       ░░░██████░░░
       ░░█◉████◉█░░     ← Limb buds
       ░══════════░
       ░░█◉████◉█░░     ← Limb buds  
       ░░░██████░░░
        ░░░░◉░░░░░      ← Tail structure
    
    ◉ = distinct basin (proto-organ)
    █ = body mass
    ═ = axial structure
```

### Stage 7-8: Fetal → Ready

Final growth and refinement. Field energy stabilizes, basins lock in place, ready for instantiation as creature.

---

## Environmental Influences

### Maternal Stress

The mother's state during pregnancy affects development:

```python
def set_maternal_environment(mother_hunger, mother_stress):
    """
    Maternal state modulates developmental noise.
    """
    self.maternal_stress = 0.7 × old_stress + 0.3 × (hunger + stress/2)
    self.environmental_noise = 0.05 + 0.1 × maternal_stress
```

Effects:
- High stress → increased developmental noise → more variation
- Starvation → asymmetric development → potential abnormalities
- Calm pregnancy → stable, symmetric development

### Perturbations

Random perturbations simulate environmental factors:

```python
if random() < environmental_noise:
    apply_perturbation()
    
def apply_perturbation(self):
    """
    Random environmental influence on development.
    """
    # Random localized kick
    kick_pos = random_position()
    kick_strength = random(0.05, 0.15)
    
    ψ[kick_pos] += kick_strength × exp(i × random_phase)
    
    log_event('perturbation', details={'position': kick_pos})
```

---

## Trait Modifiers from Development

Development doesn't just create structure—it modifies trait expression:

```python
def compute_trait_modifiers(self):
    """
    Developmental trajectory affects final phenotype.
    """
    modifiers = {}
    
    # Symmetry affects coordination
    modifiers['speed'] = 0.8 + 0.4 × bilateral_symmetry
    
    # Basin count affects complexity
    if n_basins > 6:
        modifiers['curiosity'] = 1.1  # More neural complexity
    
    # Stress during development
    if developmental_stress > 0.5:
        modifiers['metabolism'] = 0.9  # Thrifty phenotype
    
    return modifiers
```

This creates **phenotypic plasticity**:
- Same genotype + different development = different creature
- Environment during gestation matters

---

## Developmental Logging

Every significant event is recorded:

```python
@dataclass
class DevelopmentalEvent:
    step: int
    stage: DevelopmentalStage
    event_type: str  # 'division', 'symmetry_break', 'basin_form', 'perturbation'
    details: Dict
```

Example log:
```
Step    Stage         Event              Details
───────────────────────────────────────────────────────
0       ZYGOTE        initialization     mass=2.25
15      CLEAVAGE      division           cells=2
28      CLEAVAGE      division           cells=4
42      MORULA        transition         symmetry=0.92
65      BLASTULA      cavity_form        cavity_size=0.34
89      GASTRULA      symmetry_break     axis_angle=0.23
112     GASTRULA      perturbation       pos=(12,18)
145     NEURULA       basin_form         n_basins=4
178     ORGANOGENESIS basin_form         n_basins=6
200     READY         birth_ready        final_mass=4.12
```

---

## CE Theory Alignment

The embryonic development system embodies Collapse-Emergence principles:

### Symmetry Breaking = Collapse Events

Each developmental transition represents information → structure:

```
    τ (information)                    t (structure)
         │                                  │
         │  ░░░░░░░░                        │
         │  Radial                          │  ═══════
         │  symmetry                        │  Bilateral
         │  (all axes                       │  (one axis
         │   equivalent)                    │   selected)
         │                                  │
         ▼         COLLAPSE                 ▼
    ─────●──────────────────────────────────●─────────
              Symmetry-breaking event
```

### Morphospace Gradients

Development explores morphospace, settling into attractor basins:

```
    Morphospace
         │
     ╭───┼───────────────╮
     │   │    ↘          │
     │   │      ↘        │  Developmental
     │   │        ◉      │  trajectory
     │   │     attractor │
     ╰───┴───────────────╯
         
    Final form = attractor basin in developmental space
```

### Path Dependence

The same genetics can produce different outcomes based on developmental path:

```
    Genotype A ──┬── Low stress path ──→ Phenotype α
                 │
                 └── High stress path ──→ Phenotype α'
                 
    α ≠ α' even though genetics identical
```

---

## Visualization

### Embryo View (toggle during pregnancy)

```
    ┌─────────────────────────────────────┐
    │  EMBRYO: Unnamed (Orla's child)     │
    │  Stage: GASTRULA (52%)              │
    │                                     │
    │    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
    │    ░░░░░░░░████████░░░░░░░░░░░░    │
    │    ░░░░░░████████████░░░░░░░░░░    │
    │    ░░░░████████████████░░░░░░░░    │
    │    ░░░░████████████████░░░░░░░░    │
    │    ░░░░░░████████████░░░░░░░░░░    │
    │    ░░░░░░░░████████░░░░░░░░░░░░    │
    │    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░    │
    │                                     │
    │  Bilateral sym: 0.78                │
    │  Basins: 4                          │
    │  Maternal stress: 0.12              │
    └─────────────────────────────────────┘
```

---

## Technical Reference

### EmbryoField Class

```python
class EmbryoField:
    N: int = 32                    # Grid size
    g: float                       # Nonlinearity (inherited)
    psi: np.ndarray                # Complex wavefunction
    
    stage: DevelopmentalStage
    dev_step: int
    total_steps: int = 200
    
    bilateral_symmetry: float
    radial_symmetry: float
    n_basins: int
    basin_positions: List[Tuple]
    
    maternal_stress: float
    environmental_noise: float
    trait_modifiers: Dict[str, float]
    events: List[DevelopmentalEvent]
    
    def step() -> Optional[DevelopmentalEvent]
    def is_ready() -> bool
    def get_final_traits() -> Dict
```

---

```
    "Every creature is a solution to the same equation,
     but the path through morphospace is different each time.
     Development is not destiny—it's negotiation."
    
                                        — HERBIE World Embryology
```
