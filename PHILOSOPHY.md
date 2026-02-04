# PHILOSOPHY.md — Why Waves, Not Weights

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     What Makes HerbieWorld Different From Other Artificial Life Sims     ║
║                                                                           ║
║              A Document for ALife Researchers and Curious Minds           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## The Problem With Most ALife

Most artificial life simulations, despite their beauty and complexity, share a common architecture: **discrete state machines with hardcoded behavioral rules**.

Consider how behavior typically works in ALife:

```
IF hungry AND food_nearby THEN move_toward_food
IF predator_nearby THEN flee
IF mate_available AND energy > threshold THEN reproduce
```

Even sophisticated systems like neural networks are, at their core, weighted sums followed by nonlinear activations—discrete operations on discrete states, repeated.

This creates a fundamental gap: **biological cognition is continuous**. Neurons don't compute in clock cycles. Morphogenesis doesn't follow if-then rules. The physical world runs on differential equations, not decision trees.

HerbieWorld asks: *What if I took that seriously?*

---

## The HerbieWorld Approach

### No Behavioral Rules

There is no code in HerbieWorld that says:

```python
# THIS DOES NOT EXIST IN HERBIEWORLD
if creature.hungry and creature.sees_food:
    creature.move_toward(food)
```

Instead, there is a wave equation:

```python
# THIS IS WHAT ACTUALLY RUNS
psi_new = psi * np.exp(-1j * g * |psi|² * dt)  # Nonlinear phase rotation
psi_new = ifft(fft(psi) * propagator)           # Dispersive evolution
```

Behavior emerges from the dynamics of this equation—not from rules *about* behavior.

### How Movement Happens

A creature "wants" to move toward food not because of an explicit goal, but because:

1. **Food creates a potential well** V(x) in the creature's body field
2. **The wavefunction flows downhill** in this potential (quantum mechanics does this naturally)
3. **Wavefunction density shifts** toward the food location
4. **Position is computed** from the density centroid

The creature doesn't "decide" to move. The wave flows. Movement is a consequence.

### How Mating Happens

Two Herbies don't mate because `if ready_to_mate and partner_nearby`. They mate because:

1. **Torus brains are oscillating wavefunctions** with characteristic frequencies
2. **Nearby creatures' fields couple** through shared potential terms
3. **Resonance occurs** when frequencies align (like coupled pendulums)
4. **High resonance** = "bonding" = mating eligibility

The threshold isn't arbitrary—it's measuring actual wavefunction correlation. Two creatures that have spent time near each other, experiencing similar stimuli, will have more similar torus states. This *is* the relationship.

### How Development Happens

When a baby is conceived, it doesn't spawn with predefined traits. Instead:

1. **A single soliton forms** (the "zygote"—a localized wavepacket)
2. **NLSE dynamics evolve the field** through developmental time
3. **Symmetry breaking occurs** (radial → bilateral, via instabilities)
4. **Stable basins form** (the nonlinear term creates self-focusing structures)
5. **Basin count and positions** determine trait modifiers

The same parental genomes can produce different offspring phenotypes because development is **path-dependent**. A perturbation at the gastrulation stage (from maternal stress, random noise) permanently alters the trajectory through morphospace.

This isn't a metaphor. The embryo literally is a 32×32 NLSE field that evolves for 200 timesteps before birth.

---

## What's NOT Hardcoded

| Feature | Typical ALife | HerbieWorld |
|---------|---------------|-------------|
| Movement direction | `move_toward(goal)` | Wavefunction density gradient |
| Hunger response | `if hungry: seek_food` | Potential well attracts body field |
| Fear response | `if predator: flee` | Pain signal perturbs torus phase |
| Mate selection | `if compatible: bond` | Torus resonance correlation |
| Offspring traits | `child.trait = avg(parents) + noise` | Embryonic NLSE evolution |
| Memory | `memory_list.append(location)` | Basin persistence in torus |
| Death | `if health <= 0: die` | Energy conservation violation |
| Social behavior | `social_rules.evaluate()` | Field coupling between nearby creatures |

---

## The Equations Are Real

This isn't "inspired by physics." The equations are actual physics, implemented correctly:

### Nonlinear Schrödinger Equation (NLSE)

```
i ∂ψ/∂t = -½ ∇²ψ + V(x)ψ + g|ψ|²ψ
```

- Used in: Bose-Einstein condensates, optical fibers, water waves
- Solved via: Split-step Fourier method (FFT-based, spectrally accurate)
- Properties: Conserves norm (mass), supports soliton solutions

### Korteweg-de Vries Equation (KdV)

```
∂u/∂t + 6u(∂u/∂x) + ∂³u/∂x³ = 0
```

- Used in: Shallow water waves, plasma physics
- Solved via: Pseudospectral method
- Properties: Solitons pass through each other unchanged

### Maxwell's Equations (Electromagnetic Field)

```
∂E/∂t = c²∇×B - J/ε₀
∂B/∂t = -∇×E
```

- Full 2D electromagnetic field simulation
- Creatures carry charge based on their wavefunction coherence
- Enables field-mediated social coordination

These aren't approximations or simplifications for game purposes. They're the actual equations, solved with actual numerical methods used in computational physics.

---

## Why This Matters

### For Artificial Life Research

Most ALife explores **what behaviors emerge from rules**. HerbieWorld explores **what behaviors emerge from dynamics**.

This is a different question. Rules are imposed from outside. Dynamics are intrinsic to the system. When a Herbie moves toward food, it's not following a rule—it's obeying a conservation law.

### For Philosophy of Mind

If behavior can emerge from wave dynamics without explicit representation, what does that say about the necessity of symbolic cognition? HerbieWorld isn't claiming answers, but it provides a concrete system to think with.

### For Computational Modeling

The simulation demonstrates that continuous PDE-based agents are feasible (if expensive). This opens a design space between discrete automata and full physics simulation.

---

## What Emergence Looks Like Here

### Nesting Behavior

Not coded: "Herbies should build nests"

What happens:
1. Torus brain forms attractor basins from repeated experiences
2. Locations where food was found become basin centers
3. Creature movement is biased toward basin directions
4. Repeated visits reinforce the basin (self-organization)
5. Result: Creatures return to "home" locations

### Lineage Territories

Not coded: "Families should have territories"

What happens:
1. Offspring inherit parental torus state (partial basin copying)
2. Inherited basins bias movement toward parental territory
3. Siblings share basin structure, stay near each other
4. Different lineages have different inherited basins
5. Result: Family groups occupy distinct regions

### Tool-Assisted Defense

Not coded: "Herbies should use tools to fight"

What happens:
1. Hands grip objects based on tip activation (from torus efferent)
2. Grip strength varies with genome
3. When attacked, counter-damage scales with grip × tool_mass
4. Predators receive damage when attacking tool-holders
5. Result: Armed Herbies survive better (selection pressure)

---

## The Tradeoffs

### What This Approach Gains

- **Continuity**: No discrete state transitions
- **Physicality**: Behavior follows conservation laws
- **Emergence**: Complex from simple dynamics
- **Novelty**: Unexplored computational substrate

### What This Approach Loses

- **Speed**: FFTs every timestep are expensive
- **Interpretability**: What is a "thought" in a wavefunction?
- **Control**: Can't easily add specific behaviors
- **Scalability**: O(N log N) per field per creature

### The Honest Limitation

This is slow. Really slow. A laptop can handle ~50-100 creatures before frame rates suffer. Extended runs require cluster computing.

This is the cost of taking continuous dynamics seriously.

---

## Comparisons to Other Systems

### vs. Tierra / Avida (Self-replicating programs)

- Tierra: Discrete assembly instructions, digital genetics
- HerbieWorld: Continuous wavefunctions, wave-based genetics
- Shared: Emergent evolution, no hardcoded fitness

### vs. NEAT / Neural Evolution

- NEAT: Evolving network topologies, weight optimization
- HerbieWorld: Fixed wave equation, parameter inheritance
- Shared: Phenotype from genotype mapping

### vs. Polyworld

- Polyworld: Neural networks, energy economy, evolution
- HerbieWorld: Wave equations, energy conservation, evolution
- Shared: Embodied agents, ecological dynamics
- Different: Polyworld uses trained weights; HerbieWorld uses field dynamics

### vs. Lenia (Continuous cellular automata)

- Lenia: Continuous states, local update rules, beautiful morphologies
- HerbieWorld: Continuous states, global wave dynamics, embodied agents
- Shared: Aesthetic appreciation of continuous systems
- Different: Lenia is pattern-focused; HerbieWorld is agent-focused

### vs. Particle Life / Primordial Soup

- Particle Life: Discrete particles, force rules between types
- HerbieWorld: Continuous fields, wave equation evolution
- Shared: Emergent clustering, predator-prey-like dynamics
- Different: Particles are points; Herbies are extended wavefunctions

---

## The Core Claim

**HerbieWorld demonstrates that complex, lifelike behavior can emerge from the unmodified dynamics of nonlinear wave equations, without behavioral rules, reward functions, or trained parameters.**

This is not a claim about biological plausibility. Brains probably don't run NLSE dynamics (though axons do support soliton propagation, and there are wave theories of neural binding).

It's a claim about **computational sufficiency**: wave dynamics are rich enough to support cognition-like processes without additional symbolic machinery.

Whether this matters for understanding real minds is an open question. But it demonstrates that the design space for artificial life is larger than rule-systems and neural networks.

---

## For the Skeptic

"Isn't this just hiding the rules in the equation parameters?"

The equation parameters (g, V, dt) are simple constants. The NLSE with g > 0 forms solitons—this is mathematics, not a design choice. I didn't tune parameters to get "nesting behavior." I implemented NLSE correctly and nesting emerged.

"Aren't the sensory-motor mappings still hardcoded?"

The *coupling* between fields is specified (food creates potential wells, torus phase drives limbs). But the *behavior* that results is not. I specify that food attracts; I don't specify "move right when food is right."

"This is just physics simulation, not AI."

Yes. That's the point. The question is whether physics simulation can exhibit cognitive properties. HerbieWorld suggests it can—at least for minimal cognition in simple creatures.

---

## Try It Yourself

Run the simulation. Watch the creatures. Notice that:

1. No two creatures move identically (different wavefunction states)
2. Relationships form gradually (resonance builds over time)
3. Offspring resemble but differ from parents (developmental noise)
4. Death isn't instant (energy drains, field disperses)
5. Ghosts persist (the torus continues evolving after body death)

These aren't features I added. They're consequences of taking wave dynamics seriously.

---

*HerbieWorld: Where cognition is a wave, not a weight.*

---

## Further Reading

**On solitons and nonlinear waves:**
- Zabusky & Kruskal (1965) - Discovery of solitons
- Dauxois & Peyrard - Physics of Solitons

**On wave-based cognition theories:**
- Walter Freeman - Neurodynamics
- Giuseppe Vitiello - Quantum field theory of the brain

**On morphogenesis:**
- Turing (1952) - Chemical basis of morphogenesis
- Michael Levin - Bioelectricity and pattern formation

**On artificial life:**
- Langton - Artificial Life (foundational)
- Bedau - Open problems in artificial life

