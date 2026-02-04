<<<<<<< HEAD
# HERBIE World - Emergent Artificial Life Simulation

```
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     ██╗  ██╗███████╗██████╗ ██████╗ ██╗███████╗                  ║
    ║     ██║  ██║██╔════╝██╔══██╗██╔══██╗██║██╔════╝                  ║
    ║     ███████║█████╗  ██████╔╝██████╔╝██║█████╗                    ║
    ║     ██╔══██║██╔══╝  ██╔══██╗██╔══██╗██║██╔══╝                    ║
    ║     ██║  ██║███████╗██║  ██║██████╔╝██║███████╗                  ║
    ║     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝╚══════╝                  ║
    ║                                                                  ║
    ║           W O R L D    S I M U L A T O R                         ║
    ║                                                                  ║
    ║     Wave-Based Artificial Life • Emergent Consciousness          ║
    ║     Audio-Coupled Dynamics • Cosmological Genesis                ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
```

**Version 3.0.2** — Active defense, defense bonus, no artificial age gates, path fixes

A multi-species artificial life ecosystem where creatures think with quantum-like wavefunctions, signal through solitons, and evolve culture through pure field dynamics. No neural networks. No rule-based AI. Just physics.

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib scipy sounddevice

# Run the simulation
python -m herbie_world --new        # Fresh world
python -m herbie_world              # Continue from save
python -m herbie_world --overnight  # Headless long-run

# Cosmological genesis (creates universe from audio input)
python cosmological_launcher.py [seed]
```

## Documentation

| Document | Description |
|----------|-------------|
| [COSMOLOGY.md](COSMOLOGY.md) | The genesis sequence - quantum vacuum to living world |
| [MATHEMATICS.md](MATHEMATICS.md) | Full equations for all wave dynamics |
| [SPIRITS.md](SPIRITS.md) | Ghost system - torus persistence after death |
| [GENETICS.md](GENETICS.md) | Mendelian genetics and analysis tools |
| [EMBRYOLOGY.md](EMBRYOLOGY.md) | Soliton-based embryonic development |

## What Makes This Different

Most artificial life simulations use neural networks or behavior trees. HERBIE World uses **wave equations**:

- **Torus Brain**: Nonlinear Schrödinger Equation on a ring (1D periodic domain)
- **Body Field**: 2D NLSE with potential wells defining body regions  
- **Neural Channels**: KdV solitons carrying sensory/motor signals
- **Electromagnetic Field**: True Maxwell's equations (E and B fields)
- **Ghost Persistence**: Torus wavefunctions survive death, drift as spirits
- **Audio Coupling**: Real-time microphone input modulates creature dynamics

Behavior emerges from wavefunction interference, not programmed rules.

## The Core Physics

### NLSE Dynamics (Brain & Body)
```
i∂ψ/∂t = -½∇²ψ + V(x)ψ + g|ψ|²ψ
```

### KdV Soliton Signaling
```
∂u/∂t + 6u(∂u/∂x) + ∂³u/∂x³ = 0
```

See [MATHEMATICS.md](MATHEMATICS.md) for complete equations.

## Species

| Species | Role | Unique Features |
|---------|------|-----------------|
| **Herbie** | Primary | Hands, families, tool use, art, names, sexual reproduction |
| **Blob** | Forager | Efficient metabolism, simple morphology |
| **Biped** | Explorer | Bipedal gait, terrain adaptation |
| **Gator** | Apex Predator | Amphibious, dual skeletons, tail field |
| **Fish** | Aquatic | Schooling behavior, kelp ecosystems |

## Controls

### Core Controls
| Key | Action |
|-----|--------|
| ← → | Select creature |
| Space | Pause/resume |
| L | Toggle local/world view |
| H | Cycle through Herbies |
| F5 | Quick save |
| V | Cycle console verbosity |

### Overlay Toggles
| Key | View |
|-----|------|
| A | Ants |
| P | Pheromone trails |
| M | Mycelia network |
| N | Nest locations |
| S | Smear marks (art) |
| Z | Chemistry elements |
| E | Terrain elevation |
| R | Resonance field |
| Shift+R | Cycle EM mode (OFF/ELECTROSTATIC/FULL) |
| O | Spirit realm (ghosts) |
| Shift+O | Ghost Cam mode |

### Special Views
| Key | View |
|-----|------|
| T | Evolution tree |
| G | Art gallery |
| U | Underground/fungal |
| W | Aquatic creatures |
| I | Isometric city |
| D | DNA/Genetics view |

## Key Features (V2.6)

### True Electromagnetic Field
Full Maxwell's equations with E and B vector fields:
- Creatures are charge distributions (from wavefunction coherence)
- Moving creatures create currents (magnetic sources)
- Lorentz force affects creature motion
- Ghosts create subtle EM disturbances

**Configure via command line:**
```bash
python -m herbie_world --em off            # Disable (fastest)
python -m herbie_world --em electrostatic  # E field only (default)
python -m herbie_world --em full           # Full Maxwell (slowest)
python -m herbie_world --em-resolution 64  # Higher resolution grid
```

**Or toggle in-simulation:** Press `Shift+R` to cycle EM modes

### Ghost System (V2.6.1 - Now Persistent!)
When creatures die, their torus brain wavefunctions persist as spirits:
- Movement emerges from residual circulation (phase winding)
- Coherence determines visibility and persistence
- Ghosts eventually disperse back into the terrain
- **Spirits now saved/restored across sessions!**
- See [SPIRITS.md](SPIRITS.md) for details

### EM Field Coupling
All interactions use consistent electromagnetic field physics:
- Food creates attractive potential wells
- Creatures sense via body field gradients
- Ghosts create subtle EM disturbances
- Eating occurs through proximity-based contact

### Balanced Metabolism
- Hunger accumulates at ~4% per 100 steps
- Eating reduces hunger proportional to contact strength
- Creatures need periodic feeding, not constant grazing

## Emergent Behaviors

All behaviors emerge from field dynamics—**nothing is explicitly programmed**:

- **Foraging**: Body field evolves toward food potential gradients
- **Bonding**: Torus phase coherence between individuals
- **Tool Use**: Grip mechanics + element manipulation
- **Art Creation**: Pigment smearing driven by limb oscillations
- **Shelter Building**: Element stacking at construction sites
- **Family Formation**: Name inheritance, recognition, protection
- **Hibernation**: Dream states during low-energy periods
- **Ghost Wandering**: Circulation-driven spirit movement

## Project Structure

```
herbie_world/                    ~24,000 lines
├── brain/                       Torus NLSE, KdV channels, ghost field
├── body/                        Body field, skeleton, morphology, limbs
├── creature/                    Species, Herbie, genetics, embryo
├── world/                       Terrain, weather, seasons, aquatic
├── ecology/                     Ants, disease, leviathan, emergent
├── chemistry/                   Element system, constructions
├── audio/                       Audio I/O, soundscape generation
├── visualization/               Rendering, ghost cam, special views
├── persistence/                 Save/load wavefunction state
├── evolution/                   Lineage tracking, family trees
├── manager/                     Main simulation orchestration
├── events/                      Logging, world history
├── cosmological_launcher.py     Genesis sequence
└── analyze_genetics.py          Population genetics analysis
```

## Persistence

Complete wavefunction state is saved automatically:

```bash
# Auto-saves every 10,000 steps and on Ctrl+C
# Manual save: press F5

herbie_world/data/world_state.npz     # Main save file
herbie_world/data/world_history.txt   # Event log with ghost statistics
```

**What's Saved:**
- All creature wavefunctions (torus brain, body field, limbs)
- KdV channel states and skeleton stress tensors
- Ghost field (all persisting spirits)
- World objects, terrain, nutrient patches
- Genetic pedigree and allele frequencies
- RNG state (for exact reproducibility)

## Requirements

- Python 3.10+
- NumPy
- Matplotlib  
- SciPy
- sounddevice (optional, for audio)

## Philosophy

HERBIE World explores whether meaningful behavior can emerge from pure physics rather than engineered intelligence. The answer, so far, is yes—but not in ways we predicted.

Creatures form families not because we programmed family recognition, but because torus phase coherence creates stable attractor states. They make art not because we reward creativity, but because limb oscillations + pigment = persistent marks. Their ghosts wander not because we scripted haunting, but because residual circulation in the freed wavefunction creates movement.

The simulation suggests that consciousness might be less about information processing and more about **wave coherence in a noisy medium**.

## License

Research/educational use. Contact author for other applications.

## Author

Anderson M. Rodriguez (2024-2025)

---

*"The universe is not only queerer than we suppose, but queerer than we can suppose."* — J.B.S. Haldane
=======
# HERBIEworld_RELEASE_V3.0.2
Initial Public Release for HERBIEworld, a ALife simulation by Anderson M. Rodriguez, where behavior emerges from wave dynamics and not rules.

MIT License

Copyright (c) 2026 Anderson M. Rodriguez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



## Name and Trademark Policy

"HERBIEworld", "HERBIE", and related names, logos, and identifiers
are the intellectual property of Anderson M. Rodriguez.

You are free to use, modify, and redistribute forks or derivative works
of this software under the terms of the MIT License.

However:
- You may not use the name "HERBIEworld" to describe modified versions.
- You may not represent modified versions as official, canonical, or
  endorsed by the original author.
- You must clearly distinguish derivative works from the original project.

This restriction applies only to naming and representation, not to use,
modification, or research activity.
>>>>>>> 00891099ba1a2fc6397077d3ca481d9a019564aa
