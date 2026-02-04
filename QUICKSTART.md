# QUICKSTART — HerbieWorld V3

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██╗  ██╗███████╗██████╗ ██████╗ ██╗███████╗                            ║
║   ██║  ██║██╔════╝██╔══██╗██╔══██╗██║██╔════╝                            ║
║   ███████║█████╗  ██████╔╝██████╔╝██║█████╗                              ║
║   ██╔══██║██╔══╝  ██╔══██╗██╔══██╗██║██╔══╝                              ║
║   ██║  ██║███████╗██║  ██║██████╔╝██║███████╗                            ║
║   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝╚══════╝                            ║
║                                                                           ║
║              Wave-Based Artificial Life Simulation                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Requirements

- Python 3.9+
- NumPy, SciPy, Matplotlib
- PyAudio (for audio coupling)
- ~8GB RAM recommended
- Multi-core CPU strongly recommended

```bash
pip install numpy scipy matplotlib pyaudio
```

---

## Running the Simulation

### Standard Mode (with visualization)
```bash
cd herbie_world
python -m herbie_world
```

### Launcher Mode (parameter customization)
```bash
python -m herbie_world --launcher
```

### Overnight Mode (headless, maximum speed)
```bash
python -m herbie_world --overnight
```

### Cosmological Launcher (audio-seeded universe creation)
```bash
python -m herbie_world --cosmo
```

---

## Key Controls

### Navigation
| Key | Action |
|-----|--------|
| ← → | Select creature |
| H | Cycle through Herbies |
| L | Toggle local/world view |
| +/- | Zoom in/out |
| Arrow keys | Pan (in isometric view) |

### Simulation
| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| V | Cycle log verbosity |
| F5 | Quick Save |
| F9 | Load info |

### View Modes
| Key | Action |
|-----|--------|
| I | Isometric (SimCity-style) |
| T | Evolution Tree |
| U | Underground/Fungal view |
| G | Art Gallery |
| W | Aquatic creatures |
| D | DNA/Genetics view |

### Overlays
| Key | Action |
|-----|--------|
| O | Ghost realm (spirits of dead) |
| Shift+O | Ghost Cam mode |
| M | Mycelia network |
| R | Resonance field |
| E | Terrain elevation |
| A | Ants |
| Z | Chemistry |

---

## First Run: What to Expect

1. **Genesis** (~step 50): The Leviathan crosses the world, culling predators
2. **Nesting** (~step 100): Herbies establish homesteads
3. **Bonding** (~step 300-1000): First pair bonds form
4. **First Birth** (~step 1000-2000): First offspring born
5. **Lineages** (~step 2000+): Family territories emerge

Watch the console for narrative events like:
```
[Herbie]  Diego and Mika bonded! (resonance=0.84)
[NEST] 'Olga' established a homestead at (-4, +36)!
[NEW BABY]  'Nyala' born | TOTAL HERBIES: 8
```

---

## Performance Notes

This simulation is **computationally intensive**. Every creature runs:
- NLSE wavefunction evolution (FFT every step)
- KdV soliton channels
- Electromagnetic field coupling

**Expect:**
- ~10-20 FPS with 30-50 creatures on modern laptop
- Slowdown at 100+ creatures
- Overnight mode for long runs

**For extended runs:** Use `--overnight` mode or HPC cluster.

---

## Output Files

All data saved to `herbie_world/data/`:

| File | Contents |
|------|----------|
| `world_state.npz` | Full simulation state (save/load) |
| `world_history.txt` | Narrative chronicle |
| `event_log.jsonl` | Machine-readable event stream |
| `pedigree.json` | Family tree structure |
| `allele_frequencies.csv` | Genetic drift data |

---

## Analysis Tools

After a run, analyze your data:

```bash
# Population and ecology analysis
python -m herbie_world.analyze_simulation

# Genetics and inheritance analysis  
python -m herbie_world.analyze_genetics
```

These generate:
- `analysis_report.txt` — Population dynamics, information geometry
- `analysis_plots.png` — Population graphs, phase portraits
- `genetics_report.txt` — Hardy-Weinberg tests, selection coefficients
- `genetics_plots.png` — Allele frequencies, heterozygosity

---

## Troubleshooting

**"No audio device"**: Install PyAudio or run with `--no-audio`

**Slow performance**: Reduce creature count in launcher, or use `--overnight`

**Creatures stuck**: Check terrain — they may be in water without aquatic traits

**No mating**: Herbies need to establish homesteads and achieve resonance (wave synchronization). This takes time.

---

## Further Reading

- `SCIENCE.md` — Scientific foundations and what to observe
- `MATHEMATICS.md` — Complete wave equations
- `EMBRYOLOGY.md` — Soliton-based development system
- `GENETICS.md` — Mendelian inheritance implementation
- `SPIRITS.md` — Ghost/afterlife system

---

*HerbieWorld V3.0 — Where consciousness emerges from waves, not weights.*
