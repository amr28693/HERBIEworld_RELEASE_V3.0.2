# SCIENCE.md — Scientific Foundations of HerbieWorld

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     What This Simulation Is, What It Demonstrates, and What It Isn't      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Overview

HerbieWorld is an artificial life simulation exploring an alternative computational substrate for cognition and development. Rather than neural networks or rule-based AI, creatures in HerbieWorld are governed by **nonlinear wave equations**—specifically the Nonlinear Schrödinger Equation (NLSE) and Korteweg-de Vries (KdV) equation.

This is a research platform for investigating:
- Emergent behavior from continuous field dynamics
- Wave-based sensorimotor coupling
- Soliton-mediated morphogenesis
- Population genetics in artificial ecosystems

---

## Core Hypothesis

**Cognition and behavior can emerge from the dynamics of nonlinear wave equations without explicit symbolic processing or trained weights.**

This is distinct from:
- Neural networks (which learn weight matrices)
- Symbolic AI (which manipulates discrete symbols)
- Cellular automata (which use discrete state updates)

Instead, HerbieWorld creatures are continuous field systems where behavior emerges from:
- Wavefunction interference patterns
- Soliton formation and propagation
- Symmetry breaking events
- Attractor dynamics in nonlinear PDEs

---

## The Wave Mechanics

### The Torus Brain (Central Pattern Generator)

Each Herbie has a "torus brain"—a 1D ring of 64 nodes evolving under the NLSE:

```
i ∂ψ/∂t = -½ ∂²ψ/∂θ² + V(θ)ψ + g|ψ|²ψ
```

**What it does:**
- Receives afferent (sensory) inputs as localized perturbations
- Generates efferent (motor) outputs from wavefunction circulation
- Forms stable attractor basins representing behavioral "modes"

**What it demonstrates:**
- Sensorimotor integration without explicit encoding
- Continuous state spaces vs discrete symbols
- Attractor dynamics as primitive decision-making

**What it does NOT claim:**
- This is not a model of biological neurons
- This does not claim the NLSE describes actual brain dynamics
- The "cognition" here is minimal and reactive, not general intelligence

### The Body Field

Each creature has a 2D 32×32 NLSE field representing somatic state:

```
i ∂ψ/∂t = -½ ∇²ψ + V(x,y)ψ + g|ψ|²ψ
```

**What it does:**
- Encodes body state as wavefunction density distribution
- Couples to metabolism, arousal, and environmental stimuli
- Generates morphological dynamics (body shape fluctuations)

**What it demonstrates:**
- Continuous representation of body state
- Wave-mediated proprioception
- Energy dynamics through field conservation

### KdV Soliton Channels

Afferent signaling uses the KdV equation:

```
∂u/∂t + 6u(∂u/∂x) + ∂³u/∂x³ = 0
```

**What it does:**
- Propagates sensory signals as stable soliton pulses
- Maintains signal integrity over "neural" distances
- Allows collision and interaction of signals

**What it demonstrates:**
- Information transmission via solitary waves
- Nonlinear signal processing without amplifiers
- Potential relevance to axonal signal propagation theories

---

## Soliton-Based Embryogenesis

**This is a key novel feature.** When Herbies reproduce, offspring develop through a morphogenetic process governed by NLSE dynamics.

### The Process

1. **Zygote**: Single Gaussian soliton (fertilized "egg")
2. **Cleavage**: Soliton splitting through instability
3. **Gastrulation**: Symmetry breaking (radial → bilateral)
4. **Neurulation**: Body axis establishment
5. **Organogenesis**: Stable basin formation (proto-organs)
6. **Birth**: Mature wavefunction transferred to new creature

### What This Demonstrates

- **Path-dependent phenotype**: The same genotype can produce different phenotypes depending on developmental trajectory
- **Environmental coupling**: Maternal stress perturbs the embryonic field, affecting development
- **Symmetry breaking as differentiation**: Morphological complexity emerges from symmetry-breaking events in the field

### What This Does NOT Claim

- This is not a model of actual embryogenesis
- The "organs" are abstract basins, not functional structures
- This does not claim biological morphogenesis uses NLSE dynamics

### Scientific Precedent

The use of nonlinear PDEs for morphogenesis has precedent:
- Turing patterns (reaction-diffusion)
- Gierer-Meinhardt models
- More recently: morphogen gradient interpretation

The contribution here is using NLSE specifically, allowing:
- Soliton-based cell division analogs
- Phase coherence as a developmental signal
- Basin formation through self-focusing dynamics

---

## Population Genetics

HerbieWorld implements **Mendelian genetics** with realistic population dynamics.

### Features

- Diploid genomes with dominant/recessive alleles
- Mendelian segregation and independent assortment
- Random mutation during reproduction
- Sexual selection through resonance-based mate choice

### Analysis Outputs

The simulation tracks and reports:
- **Allele frequency trajectories** over time
- **Hardy-Weinberg equilibrium tests** (χ² statistics)
- **Selection coefficients** for each allele
- **Inbreeding coefficients** (F statistics)
- **Linkage disequilibrium** between loci

### What This Demonstrates

- Genetic drift in small populations
- Directional selection from differential survival
- Assortative mating effects from behavioral similarity

### Caveat

The genetics system is currently **disconnected from wave dynamics**. Genes affect phenotypic parameters (speed, metabolism) but do not directly modify the NLSE coefficients. This is a known limitation and area for future work.

---

## Emergent Behaviors

The following behaviors emerge from wave dynamics without explicit programming:

### Observed Emergent Phenomena

| Behavior | Mechanism | Robustness |
|----------|-----------|------------|
| **Nesting** | Placement memory + territorial return | Consistent |
| **Pair bonding** | Torus resonance synchronization | Consistent |
| **Lineage territories** | Offspring inherit parent location bias | Consistent |
| **Tool-assisted defense** | Hands + grip + counter-attack calculation | Consistent |
| **Art/smearing** | Probabilistic object manipulation | Occasional |
| **Burial behavior** | Response to nearby death | Occasional |

### What "Emergent" Means Here

These behaviors are not scripted. They arise from:
- Wave dynamics creating stable attractors
- Sensory coupling to environment
- Memory formation through basin persistence

However, the *potential* for these behaviors is enabled by the system architecture. This is not claiming they would emerge from any arbitrary wave system.

---

## Information-Theoretic Analysis

The simulation includes tools for information-geometric analysis:

### Metrics Computed

- **Fisher Information**: Sensitivity of population distribution to parameters
- **KL Divergence**: Distribution change between time steps
- **Transfer Entropy**: Directed information flow between species

### What This Enables

- Detecting phase transitions in ecosystem dynamics
- Quantifying "surprise" in population events
- Measuring causal influence between species

---

## Limitations and Honest Caveats

### Computational

- **Very expensive**: Full NLSE evolution is O(N log N) per field per step
- **Population limited**: ~100 creatures maximum on consumer hardware
- **Time-limited**: Extended runs require HPC resources

### Scientific

- **Not biologically realistic**: The wave equations are not models of biological systems
- **Cognition is minimal**: Creatures exhibit reactive behavior, not planning or reasoning
- **Genetics disconnected**: Mendelian system doesn't feed into wave parameters yet

### Theoretical

- **No formal proofs**: Emergence claims are empirical observations, not theorems
- **Parameter sensitivity**: Behavior depends on tuned constants
- **Interpretation challenges**: What does a "soliton thought" mean?

---

## What To Observe

If you're running the simulation for research purposes, look for:

### Short-term (steps 0-1000)
- Genesis event and predator culling
- Nest establishment patterns
- Initial bonding attempts

### Medium-term (steps 1000-5000)
- First successful reproduction
- Lineage formation
- Genetic drift in allele frequencies
- Population stabilization

### Long-term (steps 5000+)
- Multi-generational family structure
- Territorial separation of lineages
- Selection pressure effects on traits
- Hardy-Weinberg deviation (or maintenance)

### In the Ghost Realm
- Spirit coherence decay
- Correlation between life events and ghost persistence
- Singularity collapses

---

## Related Work

This simulation draws inspiration from:

- **Soliton theory**: Zabusky & Kruskal (1965), NLSE solitons
- **Artificial life**: Tierra (Ray), Avida, Polyworld
- **Morphogenesis**: Turing (1952), Gierer-Meinhardt, Levin (bioelectricity)
- **Embodied cognition**: Dynamical systems approaches to mind
- **Information geometry**: Fisher information, natural gradients

---

## Citation

If you use HerbieWorld in research, please cite:

```
HerbieWorld: Wave-Based Artificial Life Simulation
Version 3.0 (2026)
https://github.com/[repository]
```

---

## Contact

For questions about the science or implementation:
[Your contact information]

---

*HerbieWorld is a research tool for exploring wave-based computation in artificial life. It makes specific, testable claims about emergent behavior from nonlinear dynamics, while acknowledging the significant gap between these abstract models and biological reality.*
