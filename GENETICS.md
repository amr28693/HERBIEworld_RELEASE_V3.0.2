# GENETICS.md — Mendelian Genetics & Population Analysis

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     ██████╗ ███████╗███╗   ██╗███████╗████████╗██╗ ██████╗███████╗       ║
║    ██╔════╝ ██╔════╝████╗  ██║██╔════╝╚══██╔══╝██║██╔════╝██╔════╝       ║
║    ██║  ███╗█████╗  ██╔██╗ ██║█████╗     ██║   ██║██║     ███████╗       ║
║    ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝     ██║   ██║██║     ╚════██║       ║
║    ╚██████╔╝███████╗██║ ╚████║███████╗   ██║   ██║╚██████╗███████║       ║
║     ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝╚══════╝       ║
║                                                                           ║
║         Diploid Inheritance • Allele Frequencies • Trait Expression       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Overview

Herbies (and potentially other species) implement **diploid Mendelian genetics** with:
- 12 genes controlling various traits
- Dominant/recessive allele mechanics
- Sexual reproduction with meiotic recombination
- Mutation during reproduction
- Complete pedigree tracking

---

## The Genetic Architecture

### Genes and Traits

| Gene | Trait | Dominant (AA/Aa) | Recessive (aa) |
|------|-------|------------------|----------------|
| `size` | Body size | Large | Small |
| `speed` | Movement speed | Fast | Slow |
| `metabolism` | Metabolic rate | High | Low |
| `fertility` | Reproduction rate | High | Low |
| `longevity` | Lifespan | Long | Short |
| `aggression` | Conflict behavior | Aggressive | Passive |
| `sociality` | Group preference | Social | Solitary |
| `curiosity` | Exploration | Curious | Cautious |
| `pattern` | Visual pattern | Spotted | Solid |
| `coloration` | Color intensity | Bright | Dull |
| `audio_sensitivity` | Sound response | Sensitive | Deaf |
| `grip_strength` | Hand strength | Strong | Weak |

### Genotype Representation

Each individual carries two alleles per gene:

```python
class HerbieGenetics:
    genotype: Dict[str, Tuple[str, str]]
    # Example:
    # {
    #     'size': ('A', 'a'),      # Heterozygous
    #     'speed': ('A', 'A'),     # Homozygous dominant
    #     'metabolism': ('a', 'a'), # Homozygous recessive
    #     ...
    # }
```

### Dominance Mechanics

```
    Genotype → Phenotype
    
    AA (homozygous dominant)  → Dominant trait (1.0)
    Aa (heterozygous)         → Dominant trait (0.75)  [incomplete dominance]
    aa (homozygous recessive) → Recessive trait (0.0)
```

The 0.75 for heterozygotes creates **incomplete dominance**—heterozygotes show slightly less extreme phenotypes than homozygous dominants.

---

## Sexual Reproduction

### Mating Requirements

For Herbies to reproduce:
1. One must be Provider (male), one Carrier (female)
2. Both must be adults (age > maturity threshold)
3. Carrier cannot already be pregnant
4. Both must have sufficient energy
5. Torus phase coherence creates attraction

### Meiosis Simulation

When reproduction occurs, gametes are formed:

```python
def form_gamete(parent_genetics) -> Dict[str, str]:
    """
    Simulate meiosis: randomly select one allele per gene.
    """
    gamete = {}
    for gene, (allele1, allele2) in parent_genetics.genotype.items():
        # Random segregation
        gamete[gene] = random.choice([allele1, allele2])
    return gamete
```

### Fertilization

```
    Mother gamete:  {size: 'A', speed: 'a', metabolism: 'A', ...}
                         ↓
    ─────────────────────●─────────────────────
                         ↑
    Father gamete:  {size: 'a', speed: 'A', metabolism: 'a', ...}
    
    
    Offspring:      {size: ('A','a'), speed: ('a','A'), metabolism: ('A','a'), ...}
```

### Mutation

During gamete formation, mutations can occur:

```python
MUTATION_RATE = 0.02  # 2% per gene

for gene in gamete:
    if random.random() < MUTATION_RATE:
        gamete[gene] = 'A' if gamete[gene] == 'a' else 'a'
```

---

## Trait Expression

### From Genotype to Phenotype

```python
def get_phenotype_value(gene_name: str) -> float:
    """
    Convert genotype to phenotype value (0.0 - 1.0).
    """
    allele1, allele2 = self.genotype[gene_name]
    
    if allele1 == 'A' and allele2 == 'A':
        return 1.0   # Full dominant expression
    elif allele1 == 'a' and allele2 == 'a':
        return 0.0   # Full recessive expression
    else:
        return 0.75  # Heterozygote (incomplete dominance)
```

### Trait Modifiers

Phenotype values modify creature parameters:

```python
# Example: size trait
base_size = species.base_size
size_phenotype = genetics.get_phenotype_value('size')
actual_size = base_size * (0.7 + 0.6 * size_phenotype)
# Range: 0.7 × base (aa) to 1.3 × base (AA)

# Example: speed trait
base_speed = species.base_speed
speed_phenotype = genetics.get_phenotype_value('speed')
actual_speed = base_speed * (0.6 + 0.8 * speed_phenotype)
# Range: 0.6 × base (aa) to 1.4 × base (AA)
```

---

## Pedigree Tracking

### The Pedigree Structure

Every Herbie birth is recorded:

```python
{
    'id': 'H12345_042',
    'name': 'Willow',
    'generation': 3,
    'mother': 'H12340_028',
    'father': 'H12341_029',
    'mother_name': 'Orla',
    'father_name': 'Pavel',
    'birth_step': 4523,
    'genotype': {...},
    'death_step': None,  # Filled when dies
    'cause_of_death': None,
    'children': ['H12350_051', 'H12352_053']
}
```

### Pedigree File

Saved to `herbie_world/data/pedigree.json`:

```json
{
  "individuals": {
    "H12345_042": {
      "name": "Willow",
      "generation": 3,
      "parents": ["H12340_028", "H12341_029"],
      "children": ["H12350_051"],
      "birth_step": 4523,
      "death_step": 8901,
      "genotype": {
        "size": ["A", "a"],
        "speed": ["A", "A"],
        ...
      }
    },
    ...
  },
  "allele_frequencies": {
    "5000": {"size_A": 0.65, "size_a": 0.35, ...},
    "10000": {"size_A": 0.58, "size_a": 0.42, ...}
  }
}
```

---

## Population Genetics Analysis

### Running the Analysis

```bash
python analyze_genetics.py
```

This generates:
- `genetics_report.txt` — Human-readable analysis
- `genetics_plots.png` — Visualization of allele dynamics
- `allele_frequencies.csv` — Raw frequency data

### What's Analyzed

1. **Allele Frequency Dynamics**
   - Track A vs a frequency for each gene over time
   - Detect selection pressure (frequency changes)
   - Identify fixation events (allele reaches 100%)

2. **Hardy-Weinberg Equilibrium**
   ```
   Expected frequencies:
     p² (AA) + 2pq (Aa) + q² (aa) = 1
   
   where p = freq(A), q = freq(a)
   ```
   
   Deviations indicate:
   - Selection pressure
   - Non-random mating
   - Genetic drift
   - Population structure

3. **Heterozygosity**
   ```
   H = 2pq = proportion of heterozygotes
   ```
   
   High H → genetic diversity
   Low H → inbreeding or selection

4. **Generation Statistics**
   - Average traits per generation
   - Fitness correlations
   - Lineage success rates

### Example Report Output

```
======================================================================
                 HERBIE GENETICS ANALYSIS REPORT
======================================================================

SIMULATION SUMMARY
  Total individuals tracked: 847
  Generations observed: 12
  Final population: 23

ALLELE FREQUENCIES (Final)
  Gene          A freq    a freq    H-W χ²
  ────────────────────────────────────────
  size          0.62      0.38      2.31
  speed         0.71      0.29      1.87*
  metabolism    0.45      0.55      0.92
  fertility     0.58      0.42      3.45*
  ...

  * Significant deviation from Hardy-Weinberg (p < 0.05)

SELECTION DETECTED
  speed: Significant increase in 'A' allele (Δp = +0.21)
         Fast individuals may have survival advantage

  metabolism: Drift toward 'a' allele (Δp = -0.15)
              Low metabolism favorable in resource-limited environment

INBREEDING COEFFICIENT
  F = 0.12 (moderate inbreeding detected)
  Common ancestor pairs: Pavel-Orla lineage (47% of population)

NOTABLE LINEAGES
  Most successful: Pavel → 23 living descendants
  Extinct lines: Haru (no surviving offspring after Gen 4)
```

### Visualization

The `genetics_plots.png` includes:

```
    ┌─────────────────────────────────────┐
    │  Allele Frequency Dynamics          │
    │                                     │
    │  1.0 ─┬─────────────────────────    │
    │       │   ___________               │
    │       │  /           \___  size_A   │
    │  0.5 ─┼─/                 \___      │
    │       │/                      \___  │
    │  0.0 ─┴─────────────────────────    │
    │       0    2000   4000   6000       │
    │                 Step                 │
    └─────────────────────────────────────┘
    
    ┌─────────────────────────────────────┐
    │  Heterozygosity Over Time           │
    │                                     │
    │  0.5 ─┬─────────────────────────    │
    │       │ ____                        │
    │       │/    \____                   │
    │  0.25─┼          \____              │
    │       │               \____         │
    │  0.0 ─┴─────────────────────────    │
    │       Gen 0  Gen 3  Gen 6  Gen 9    │
    └─────────────────────────────────────┘
```

---

## Evolutionary Dynamics

### Natural Selection

Selection emerges from differential survival:

```
    Environment                    Trait Favored
    ─────────────────────────────────────────────
    Food scarce                    Low metabolism (aa)
    Predators common               Fast speed (AA/Aa)
    Large territory                High curiosity (AA/Aa)
    Dense population               High sociality (AA/Aa)
    Harsh seasons                  High longevity (AA/Aa)
```

### Genetic Drift

In small populations, random sampling causes frequency changes:
- Founder effects from initial spawning
- Bottlenecks after predator events
- Random loss of rare alleles

### Gene Flow

Migration events introduce new alleles:
- Seasonal migrations mix populations
- Rare long-distance dispersal

---

## DNA View (D key)

The visualization includes a genetics overlay:

```
    ┌─────────────────────────────────────┐
    │  WILLOW - Gen 3                     │
    │                                     │
    │  size:        ██░░ Aa (Large)       │
    │  speed:       ████ AA (Fast)        │
    │  metabolism:  ░░░░ aa (Low)         │
    │  fertility:   ██░░ Aa (High)        │
    │  longevity:   ████ AA (Long)        │
    │  aggression:  ░░░░ aa (Passive)     │
    │  ...                                │
    │                                     │
    │  Parents: Pavel × Orla              │
    │  Children: 3 (Sage, River, Brook)   │
    └─────────────────────────────────────┘
```

---

## Integration with Other Systems

### Embryonic Development

Genetics influence development trajectory:
- `size` genotype affects embryo growth rate
- `metabolism` genotype affects developmental timing
- Environmental stress can modify expression

### Ghost Persistence

Interestingly, genetics may correlate with ghost quality:
- High `longevity` individuals have more stable torus dynamics
- This isn't programmed—it emerges from correlated parameters

### Tool Use

`grip_strength` genotype affects:
- Maximum object weight that can be held
- Weapon effectiveness in combat
- Construction ability

---

## Future Directions

The genetics system is designed for expansion:

1. **Epistasis**: Gene interactions (e.g., size + metabolism)
2. **Pleiotropy**: Single genes affecting multiple traits
3. **Quantitative traits**: Polygenic inheritance
4. **Sexual selection**: Mate choice based on phenotype
5. **Speciation**: Reproductive isolation from divergence

---

```
    "Evolution is not a force but an outcome.
     Selection is not a mechanism but a pattern.
     Genetics is not destiny but tendency."
    
                                        — HERBIE World Genetics Module
```
