"""
Electromagnetic Field - True Maxwell's Equations for HERBIE World V2.6

This implements a genuine EM field with:
- Electric field E(x,y) as a 2D vector field
- Magnetic field B(x,y) as a pseudoscalar in 2D (perpendicular to plane)
- Charge densities from creature wavefunctions
- Current densities from creature movement
- Finite propagation speed (light cone causality)

The EM field provides:
1. True field-mediated interaction at finite speed
2. Radiation from accelerating charges (moving creatures)
3. Standing wave patterns from boundary conditions
4. Faraday induction effects (changing B creates E)

Physics:
    Maxwell's equations in 2D (with B perpendicular to plane):
    
    ∂E/∂t = c²∇×B - J/ε₀           (Ampère-Maxwell)
    ∂B/∂t = -∇×E                    (Faraday)
    ∇·E = ρ/ε₀                      (Gauss)
    ∇·B = 0                         (no monopoles - automatic in 2D)

In 2D with B = B_z ẑ:
    ∂Ex/∂t = c²(∂B/∂y) - Jx/ε₀
    ∂Ey/∂t = -c²(∂B/∂x) - Jy/ε₀
    ∂B/∂t = ∂Ex/∂y - ∂Ey/∂x

CE Theory alignment:
- EM field is the "real" side of the information-matter duality
- Creatures are charge distributions whose wavefunctions source the field
- Retarded potentials create temporal structure (causality)
- Field energy represents physical degree of freedom bookkeeping

PERFORMANCE NOTE:
This can be computationally expensive. Use enable_em_field flag to toggle.
Start with coarse resolution (32x32) and see if stable before increasing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from enum import Enum


class EMFieldMode(Enum):
    """Operating modes for performance tuning."""
    OFF = 0           # No EM computation
    ELECTROSTATIC = 1 # Only E field, no time evolution (cheapest)
    FULL_MAXWELL = 2  # Full Maxwell evolution (most expensive)
    HYBRID = 3        # Full Maxwell at low resolution + interpolation


@dataclass
class EMFieldConfig:
    """Configuration for EM field computation."""
    mode: EMFieldMode = EMFieldMode.FULL_MAXWELL
    resolution: int = 32          # Grid resolution (32 = ~1000 cells, 64 = ~4000)
    c: float = 2.0                # Speed of light in grid units/timestep
    epsilon_0: float = 1.0        # Permittivity (affects field strength)
    mu_0: float = 1.0             # Permeability (affects wave speed)
    damping: float = 0.05         # Field energy dissipation per step (increased from 0.01)
    charge_spread: float = 2.0    # Gaussian spread of creature charges
    coupling_strength: float = 0.08  # How strongly creatures feel the field (reduced from 0.3)


class ElectromagneticField:
    """
    True 2D Maxwell electromagnetic field.
    
    Creatures are modeled as charge distributions:
    - Body field density → charge density
    - Body field momentum → current density
    - Torus coherence → charge magnitude
    
    The field evolves via FDTD (Finite Difference Time Domain) method.
    """
    
    def __init__(self, world_size: float, config: EMFieldConfig = None):
        """
        Initialize the EM field.
        
        Args:
            world_size: Physical size of the world
            config: Field configuration (uses defaults if None)
        """
        self.world_size = world_size
        self.config = config or EMFieldConfig()
        self.N = self.config.resolution
        self.dx = world_size / self.N
        self.dt = 0.5 * self.dx / self.config.c  # CFL condition for stability
        
        # Electric field: E = (Ex, Ey) at each grid point
        self.Ex = np.zeros((self.N, self.N))
        self.Ey = np.zeros((self.N, self.N))
        
        # Magnetic field: B = Bz (perpendicular to 2D plane)
        self.Bz = np.zeros((self.N, self.N))
        
        # Source terms (accumulated each step)
        self.rho = np.zeros((self.N, self.N))    # Charge density
        self.Jx = np.zeros((self.N, self.N))     # Current density x
        self.Jy = np.zeros((self.N, self.N))     # Current density y
        
        # Precompute wavenumbers for spectral operations
        kx = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, self.dx) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1  # Avoid division by zero
        
        # Statistics
        self.total_energy = 0.0
        self.max_E = 0.0
        self.max_B = 0.0
        self.step_count = 0
        
        # Coordinate grids for creature-field interaction
        x = np.linspace(0, world_size, self.N, endpoint=False)
        y = np.linspace(0, world_size, self.N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        
    def _world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world position to grid indices with wrapping."""
        x = pos[0] % self.world_size
        y = pos[1] % self.world_size
        i = int(x / self.dx) % self.N
        j = int(y / self.dx) % self.N
        return i, j
    
    def clear_sources(self):
        """Clear source terms for new timestep."""
        self.rho.fill(0)
        self.Jx.fill(0)
        self.Jy.fill(0)
    
    def add_creature_charge(self, pos: np.ndarray, charge: float, 
                            velocity: np.ndarray = None):
        """
        Add a creature as a charge distribution.
        
        Args:
            pos: World position
            charge: Charge magnitude (from wavefunction coherence)
            velocity: Creature velocity (creates current)
        """
        if self.config.mode == EMFieldMode.OFF:
            return
            
        if np.any(np.isnan(pos)):
            return
            
        # Gaussian charge distribution
        x0 = pos[0] % self.world_size
        y0 = pos[1] % self.world_size
        sigma = self.config.charge_spread * self.dx
        
        # Distance from charge center (with periodic wrapping)
        dx = self.X - x0
        dy = self.Y - y0
        # Wrap to nearest image
        dx = np.where(dx > self.world_size/2, dx - self.world_size, dx)
        dx = np.where(dx < -self.world_size/2, dx + self.world_size, dx)
        dy = np.where(dy > self.world_size/2, dy - self.world_size, dy)
        dy = np.where(dy < -self.world_size/2, dy + self.world_size, dy)
        
        r2 = dx**2 + dy**2
        gaussian = np.exp(-r2 / (2 * sigma**2))
        gaussian /= (np.sum(gaussian) + 1e-10)  # Normalize
        
        self.rho += charge * gaussian
        
        # Current from velocity
        if velocity is not None and np.linalg.norm(velocity) > 0.01:
            self.Jx += charge * velocity[0] * gaussian
            self.Jy += charge * velocity[1] * gaussian
    
    def add_ghost_charge(self, pos: np.ndarray, coherence: float, 
                         circulation: float):
        """
        Ghosts create subtle EM disturbances.
        
        Coherent ghosts → more charge
        Circulation → rotating dipole (creates magnetic field)
        """
        if self.config.mode == EMFieldMode.OFF:
            return
            
        # Ghost charge proportional to coherence
        ghost_charge = coherence * 0.2  # Weaker than living creatures
        self.add_creature_charge(pos, ghost_charge)
        
        # Circulation creates rotating current (magnetic source)
        if abs(circulation) > 0.01:
            x0 = pos[0] % self.world_size
            y0 = pos[1] % self.world_size
            sigma = self.config.charge_spread * self.dx * 2
            
            dx = self.X - x0
            dy = self.Y - y0
            dx = np.where(dx > self.world_size/2, dx - self.world_size, dx)
            dx = np.where(dx < -self.world_size/2, dx + self.world_size, dx)
            dy = np.where(dy > self.world_size/2, dy - self.world_size, dy)
            dy = np.where(dy < -self.world_size/2, dy + self.world_size, dy)
            
            r2 = dx**2 + dy**2
            gaussian = np.exp(-r2 / (2 * sigma**2))
            
            # Circular current pattern
            self.Jx += circulation * 0.1 * (-dy) * gaussian / (r2 + 1)
            self.Jy += circulation * 0.1 * dx * gaussian / (r2 + 1)
    
    def evolve(self):
        """
        Evolve the EM field by one timestep using FDTD.
        
        Implements Maxwell's equations:
            ∂Ex/∂t = c²(∂Bz/∂y) - Jx/ε₀
            ∂Ey/∂t = -c²(∂Bz/∂x) - Jy/ε₀
            ∂Bz/∂t = ∂Ex/∂y - ∂Ey/∂x
        """
        if self.config.mode == EMFieldMode.OFF:
            return
            
        if self.config.mode == EMFieldMode.ELECTROSTATIC:
            self._solve_poisson()
            return
        
        c2 = self.config.c ** 2
        eps = self.config.epsilon_0
        dt = self.dt
        
        # Spatial derivatives (central differences with periodic BC)
        def ddx(f):
            return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * self.dx)
        
        def ddy(f):
            return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * self.dx)
        
        # Update E field (Ampère-Maxwell)
        dBz_dy = ddy(self.Bz)
        dBz_dx = ddx(self.Bz)
        
        self.Ex += dt * (c2 * dBz_dy - self.Jx / eps)
        self.Ey += dt * (-c2 * dBz_dx - self.Jy / eps)
        
        # Update B field (Faraday)
        dEx_dy = ddy(self.Ex)
        dEy_dx = ddx(self.Ey)
        
        self.Bz += dt * (dEx_dy - dEy_dx)
        
        # Damping (energy dissipation)
        damping = 1.0 - self.config.damping
        self.Ex *= damping
        self.Ey *= damping
        self.Bz *= damping
        
        # Divergence cleaning (enforce Gauss's law spectrally)
        # This prevents charge from numerical errors
        self._clean_divergence()
        
        # Update statistics
        self._update_stats()
        self.step_count += 1
    
    def _solve_poisson(self):
        """
        Electrostatic mode: solve Poisson's equation for E.
        
        ∇²φ = -ρ/ε₀
        E = -∇φ
        """
        # Solve in Fourier space
        rho_k = np.fft.fft2(self.rho)
        
        # φ_k = ρ_k / (ε₀ * k²)
        phi_k = rho_k / (self.config.epsilon_0 * self.K2)
        phi_k[0, 0] = 0  # Zero mean potential
        
        # E = -∇φ in Fourier space: E_k = -ik * φ_k
        Ex_k = -1j * self.KX * phi_k
        Ey_k = -1j * self.KY * phi_k
        
        self.Ex = np.fft.ifft2(Ex_k).real
        self.Ey = np.fft.ifft2(Ey_k).real
        
        self._update_stats()
    
    def _clean_divergence(self):
        """
        Project E field to be divergence-free except for charges.
        
        Enforces: ∇·E = ρ/ε₀
        """
        # Current divergence
        Ex_k = np.fft.fft2(self.Ex)
        Ey_k = np.fft.fft2(self.Ey)
        div_E_k = 1j * self.KX * Ex_k + 1j * self.KY * Ey_k
        
        # Target divergence from charge
        rho_k = np.fft.fft2(self.rho)
        target_div_k = rho_k / self.config.epsilon_0
        
        # Correction potential
        correction_k = (target_div_k - div_E_k) / (self.K2 + 1e-10)
        correction_k[0, 0] = 0
        
        # Apply correction
        self.Ex += np.fft.ifft2(1j * self.KX * correction_k).real * 0.1
        self.Ey += np.fft.ifft2(1j * self.KY * correction_k).real * 0.1
    
    def _update_stats(self):
        """Update field statistics."""
        E2 = self.Ex**2 + self.Ey**2
        B2 = self.Bz**2
        
        # EM energy density: u = ε₀E²/2 + B²/(2μ₀)
        self.total_energy = 0.5 * np.sum(
            self.config.epsilon_0 * E2 + B2 / self.config.mu_0
        ) * self.dx**2
        
        self.max_E = np.sqrt(np.max(E2))
        self.max_B = np.max(np.abs(self.Bz))
    
    def sense(self, pos: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Sense the EM field at a position.
        
        Returns:
            E: Electric field vector [Ex, Ey]
            B: Magnetic field (scalar in 2D)
            grad_E: Gradient of |E| (direction of increasing field strength)
        """
        if self.config.mode == EMFieldMode.OFF:
            return np.zeros(2), 0.0, np.zeros(2)
            
        if np.any(np.isnan(pos)):
            return np.zeros(2), 0.0, np.zeros(2)
        
        i, j = self._world_to_grid(pos)
        
        # Field values at position
        E = np.array([self.Ex[j, i], self.Ey[j, i]])
        B = self.Bz[j, i]
        
        # Gradient of |E|
        E_mag = np.sqrt(self.Ex**2 + self.Ey**2)
        ip = (i + 1) % self.N
        im = (i - 1) % self.N
        jp = (j + 1) % self.N
        jm = (j - 1) % self.N
        
        grad_E = np.array([
            (E_mag[j, ip] - E_mag[j, im]) / (2 * self.dx),
            (E_mag[jp, i] - E_mag[jm, i]) / (2 * self.dx)
        ])
        
        return E, B, grad_E
    
    def get_force_on_charge(self, pos: np.ndarray, charge: float,
                            velocity: np.ndarray) -> np.ndarray:
        """
        Lorentz force on a moving charge.
        
        F = q(E + v × B)
        
        In 2D with B perpendicular:
        F = q[Ex + vy*Bz, Ey - vx*Bz]
        """
        if self.config.mode == EMFieldMode.OFF:
            return np.zeros(2)
            
        E, B, _ = self.sense(pos)
        
        # Lorentz force
        Fx = charge * (E[0] + velocity[1] * B)
        Fy = charge * (E[1] - velocity[0] * B)
        
        return np.array([Fx, Fy]) * self.config.coupling_strength
    
    def get_energy_density_at(self, pos: np.ndarray) -> float:
        """Get EM energy density at a position."""
        if self.config.mode == EMFieldMode.OFF:
            return 0.0
            
        i, j = self._world_to_grid(pos)
        E2 = self.Ex[j, i]**2 + self.Ey[j, i]**2
        B2 = self.Bz[j, i]**2
        
        return 0.5 * (self.config.epsilon_0 * E2 + B2 / self.config.mu_0)
    
    def get_poynting_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Poynting vector S = E × B / μ₀ (energy flow direction).
        
        In 2D: S = (Ey*Bz, -Ex*Bz) / μ₀
        """
        Sx = self.Ey * self.Bz / self.config.mu_0
        Sy = -self.Ex * self.Bz / self.config.mu_0
        return Sx, Sy
    
    def get_stats(self) -> dict:
        """Get field statistics."""
        return {
            'total_energy': self.total_energy,
            'max_E': self.max_E,
            'max_B': self.max_B,
            'total_charge': np.sum(self.rho) * self.dx**2,
            'step': self.step_count,
            'mode': self.config.mode.name
        }
    
    def to_dict(self) -> dict:
        """Serialize for saving."""
        return {
            'Ex': self.Ex.tolist(),
            'Ey': self.Ey.tolist(),
            'Bz': self.Bz.tolist(),
            'step_count': self.step_count,
            'world_size': self.world_size,
            'resolution': self.N,
            'mode': self.config.mode.value
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ElectromagneticField':
        """Deserialize from saved state."""
        config = EMFieldConfig(
            resolution=d['resolution'],
            mode=EMFieldMode(d.get('mode', 2))
        )
        em = cls(d['world_size'], config)
        em.Ex = np.array(d['Ex'])
        em.Ey = np.array(d['Ey'])
        em.Bz = np.array(d['Bz'])
        em.step_count = d['step_count']
        return em


class CreatureEMCoupler:
    """
    Couples creatures to the EM field.
    
    Creatures:
    - SOURCE the field through charge/current from wavefunctions
    - FEEL the field through Lorentz force on their body field
    - SENSE the field through afferent channels
    
    The coupling creates:
    - Long-range awareness (E field from other creatures)
    - Motion-dependent interaction (magnetic effects)
    - Radiation when accelerating (changing currents)
    """
    
    def __init__(self, em_field: ElectromagneticField):
        """
        Args:
            em_field: The electromagnetic field to couple to
        """
        self.em = em_field
    
    def creature_to_charge(self, creature) -> Tuple[float, np.ndarray]:
        """
        Extract charge and current from creature state.
        
        Charge ~ coherence of body field (concentrated = more charge)
        Current ~ momentum of body field (moving density = current)
        """
        charge = 0.0
        velocity = np.zeros(2)
        
        if not hasattr(creature, 'alive') or not creature.alive:
            return charge, velocity
        
        # Base charge from torus coherence
        if hasattr(creature, 'torus') and creature.torus is not None:
            coherence = creature.torus.coherence if hasattr(creature.torus, 'coherence') else 1.0
            charge = 0.5 + 0.5 * min(coherence / 5.0, 1.0)
        
        # Body field contribution
        if hasattr(creature, 'body_field') and creature.body_field is not None:
            psi = creature.body_field.psi
            density = np.abs(psi)**2
            # More concentrated = more charge
            concentration = np.sum(density**2) / (np.sum(density)**2 + 1e-10) * psi.size
            charge *= (0.5 + 0.5 * min(concentration, 2.0))
        
        # Velocity for current
        if hasattr(creature, 'vel'):
            velocity = np.array(creature.vel) if not np.any(np.isnan(creature.vel)) else np.zeros(2)
        
        # Species modifier
        if hasattr(creature, 'species'):
            if creature.species.name == 'Herbie':
                charge *= 1.2  # Herbies are more "charged" (social)
            elif creature.species.diet == 'carnivore':
                charge *= 0.6  # Predators are EM-quiet (stealth)
        
        return charge, velocity
    
    def apply_em_to_creature(self, creature, E: np.ndarray, B: float, 
                             grad_E: np.ndarray):
        """
        Apply EM field effects to creature.
        
        Effects:
        1. Lorentz force on body field momentum
        2. E field biases torus phase
        3. B field affects circulation tendency
        4. Afferent signaling from field intensity
        """
        if not hasattr(creature, 'alive') or not creature.alive:
            return
        
        # Get creature's effective charge
        charge, _ = self.creature_to_charge(creature)
        
        # 1. LORENTZ FORCE on motion
        # F = q(E + v×B)
        if hasattr(creature, 'vel') and charge > 0.1:
            vel = creature.vel
            Fx = charge * (E[0] + vel[1] * B) * self.em.config.coupling_strength
            Fy = charge * (E[1] - vel[0] * B) * self.em.config.coupling_strength
            
            # Apply very gently to avoid instability
            creature.vel[0] += np.clip(Fx * 0.005, -0.2, 0.2)
            creature.vel[1] += np.clip(Fy * 0.005, -0.2, 0.2)
        
        # 2. TORUS PHASE BIAS from E field direction
        if hasattr(creature, 'torus') and creature.torus is not None:
            E_mag = np.linalg.norm(E)
            if E_mag > 0.05:  # Higher threshold
                E_angle = np.arctan2(E[1], E[0])
                # Bias torus toward E field direction - very gentle
                creature.torus.inject_directional_bias(
                    E / (E_mag + 1e-6),
                    E_mag * self.em.config.coupling_strength * 0.02  # Reduced from 0.1
                )
        
        # 3. MAGNETIC FIELD affects circulation
        if hasattr(creature, 'torus') and creature.torus is not None and abs(B) > 0.05:
            # B field creates rotational tendency in torus - very gentle
            creature.torus.circulation += B * self.em.config.coupling_strength * 0.002  # Reduced
        
        # 4. AFFERENT SIGNALING from field intensity
        if hasattr(creature, 'afferent'):
            E_mag = np.linalg.norm(E)
            
            # Strong E field → environmental awareness (reduced sensitivity)
            if E_mag > 0.1 and 'proprioception' in creature.afferent:
                creature.afferent['proprioception'].nucleate(
                    E_mag * 0.1, 0.5  # Reduced from 0.3
                )
            
            # Gradient gives directional information
            grad_mag = np.linalg.norm(grad_E)
            if grad_mag > 0.05 and 'env_reward' in creature.afferent:
                # Strong gradient = something interesting nearby
                creature.afferent['env_reward'].nucleate(
                    grad_mag * 0.05, 0.3  # Reduced from 0.2
                )


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def integrate_em_field(manager, em_field: ElectromagneticField, 
                       coupler: CreatureEMCoupler):
    """
    Integrate EM field into the simulation step.
    
    Call this once per timestep after creature updates.
    
    Args:
        manager: CreatureManager with creatures list
        em_field: The EM field
        coupler: Creature-field coupler
    """
    if em_field.config.mode == EMFieldMode.OFF:
        return
    
    # Clear sources
    em_field.clear_sources()
    
    # Add creature charges
    for creature in manager.creatures:
        if not creature.alive:
            continue
        charge, velocity = coupler.creature_to_charge(creature)
        em_field.add_creature_charge(creature.pos, charge, velocity)
    
    # Add ghost charges
    if hasattr(manager, 'ghost_field'):
        for ghost in manager.ghost_field.ghosts:
            if ghost.alive:
                coherence = ghost._compute_coherence()
                circulation = getattr(ghost, 'circulation', 0)
                em_field.add_ghost_charge(ghost.pos, coherence, circulation)
    
    # Evolve field
    em_field.evolve()
    
    # Apply field to creatures
    for creature in manager.creatures:
        if not creature.alive:
            continue
        E, B, grad_E = em_field.sense(creature.pos)
        coupler.apply_em_to_creature(creature, E, B, grad_E)
