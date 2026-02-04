"""
Global Resonance Field - Schumann-like cavity resonance for creature coupling.

This implements a physically-motivated global field that:
1. Mediates all creature-creature interactions (no direct coupling)
2. Supports standing wave resonances (Schumann-like modes)
3. Enables safe parallel computation (all reads from field, writes deferred)
4. Creates emergent global coherence patterns

Physics basis:
- Earth's Schumann resonance: EM standing waves in Earth-ionosphere cavity
- Fundamental mode ~7.83 Hz, harmonics at 14.3, 20.8, 27.3, 33.8 Hz
- We scale this to simulation timesteps

The field equation:
    ∂²ψ/∂t² = c²∇²ψ - γ∂ψ/∂t + Σ source_i(x,t)

Where:
- c = wave propagation speed
- γ = damping (cavity Q factor)
- source_i = emission from creature i

Creatures:
- EMIT: inject amplitude at their position based on torus state
- SENSE: read local field value + gradient
- COUPLE: field affects torus evolution, creating feedback loops

This replaces direct creature-creature sensing with field-mediated interaction,
which is both more physical AND enables safe parallel stepping.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from scipy.ndimage import laplace, gaussian_filter


@dataclass
class SchumannMode:
    """A resonant mode of the global cavity."""
    frequency: float      # In simulation units (cycles per step)
    wavelength: float     # Spatial wavelength
    amplitude: float      # Current amplitude
    phase: float          # Current phase
    q_factor: float       # Quality factor (higher = less damping)
    
    def energy(self) -> float:
        """Mode energy ~ amplitude²."""
        return self.amplitude ** 2


class GlobalResonanceField:
    """
    Global EM-like field with Schumann resonance modes.
    
    The field provides:
    1. Instantaneous (within-timestep) coupling between all creatures
    2. Standing wave patterns that create spatial coherence
    3. Temporal coherence through resonant modes
    4. A physical medium for "sensing at a distance"
    
    Usage:
        field = GlobalResonanceField(world_size=100, resolution=64)
        
        # Each timestep:
        for creature in creatures:
            field.emit(creature.pos, creature.get_emission())
        
        field.evolve()
        
        for creature in creatures:
            local_value, gradient = field.sense(creature.pos)
            creature.receive_field(local_value, gradient)
    """
    
    # Schumann-like frequencies (scaled to simulation)
    # Real Schumann: 7.83, 14.3, 20.8, 27.3, 33.8 Hz
    # We use ratios: 1.0, 1.83, 2.66, 3.49, 4.32
    SCHUMANN_RATIOS = [1.0, 1.83, 2.66, 3.49, 4.32]
    
    def __init__(self, world_size: float, resolution: int = 64,
                 base_frequency: float = 0.02,  # Cycles per timestep
                 wave_speed: float = 5.0,       # Grid cells per timestep
                 damping: float = 0.02,         # Energy loss per step
                 q_factor: float = 50.0):       # Cavity quality
        """
        Initialize the global resonance field.
        
        Args:
            world_size: Physical size of world
            resolution: Grid resolution for field
            base_frequency: Fundamental Schumann frequency
            wave_speed: Wave propagation speed
            damping: Global damping coefficient
            q_factor: Resonance quality factor
        """
        self.world_size = world_size
        self.resolution = resolution
        self.dx = world_size / resolution
        self.base_frequency = base_frequency
        self.wave_speed = wave_speed
        self.damping = damping
        self.q_factor = q_factor
        
        # Field state: complex wavefunction for amplitude and phase
        self.psi = np.zeros((resolution, resolution), dtype=np.complex128)
        self.psi_prev = np.zeros((resolution, resolution), dtype=np.complex128)
        
        # Velocity field (for wave equation)
        self.dpsi_dt = np.zeros((resolution, resolution), dtype=np.complex128)
        
        # Emission accumulator (creatures write here during parallel phase)
        self.emission_buffer = np.zeros((resolution, resolution), dtype=np.complex128)
        
        # Initialize Schumann modes
        self.modes = self._init_schumann_modes()
        
        # Mode amplitudes for visualization
        self.mode_energies = np.zeros(len(self.SCHUMANN_RATIOS))
        
        # Statistics
        self.total_energy = 0.0
        self.coherence = 0.0  # Global phase coherence
        self.step_count = 0
        
    def _init_schumann_modes(self) -> List[SchumannMode]:
        """Initialize Schumann resonance modes."""
        modes = []
        for i, ratio in enumerate(self.SCHUMANN_RATIOS):
            freq = self.base_frequency * ratio
            # Wavelength decreases with frequency
            wavelength = self.wave_speed / freq
            modes.append(SchumannMode(
                frequency=freq,
                wavelength=wavelength,
                amplitude=0.01 * (1.0 / (i + 1)),  # Higher modes start weaker
                phase=np.random.uniform(0, 2 * np.pi),
                q_factor=self.q_factor / (i + 1)  # Higher modes decay faster
            ))
        return modes
    
    def _world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world position to grid indices."""
        # Wrap around (toroidal world)
        x = pos[0] % self.world_size
        y = pos[1] % self.world_size
        
        i = int(x / self.dx) % self.resolution
        j = int(y / self.dx) % self.resolution
        return i, j
    
    def emit(self, pos: np.ndarray, amplitude: complex, spread: float = 2.0):
        """
        Emit into the field at a position.
        
        Called by creatures during their step. Safe to call in parallel
        because we accumulate into a buffer.
        
        Args:
            pos: World position
            amplitude: Complex emission (magnitude and phase)
            spread: Spatial spread in grid cells
        """
        i, j = self._world_to_grid(pos)
        
        # Gaussian spread around emission point
        # Create a small patch to add
        patch_size = int(spread * 3) + 1
        half = patch_size // 2
        
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                ii = (i + di) % self.resolution
                jj = (j + dj) % self.resolution
                dist = np.sqrt(di*di + dj*dj)
                weight = np.exp(-dist*dist / (2 * spread * spread))
                self.emission_buffer[ii, jj] += amplitude * weight
    
    def sense(self, pos: np.ndarray) -> Tuple[complex, np.ndarray]:
        """
        Sense the field at a position.
        
        Returns:
            value: Complex field value (amplitude and phase)
            gradient: 2D gradient vector (direction of increasing amplitude)
        """
        # Guard against NaN positions
        if np.any(np.isnan(pos)):
            return 0.0 + 0j, np.array([0.0, 0.0])
        
        i, j = self._world_to_grid(pos)
        
        # Local value
        value = self.psi[i, j]
        
        # Guard against NaN field values
        if np.isnan(value):
            value = 0.0 + 0j
        
        # Gradient (central differences with wrapping)
        ip = (i + 1) % self.resolution
        im = (i - 1) % self.resolution
        jp = (j + 1) % self.resolution
        jm = (j - 1) % self.resolution
        
        grad_x = (np.abs(self.psi[ip, j]) - np.abs(self.psi[im, j])) / (2 * self.dx)
        grad_y = (np.abs(self.psi[i, jp]) - np.abs(self.psi[i, jm])) / (2 * self.dx)
        
        gradient = np.array([grad_x, grad_y])
        
        # Guard against NaN gradient
        if np.any(np.isnan(gradient)):
            gradient = np.array([0.0, 0.0])
        
        return value, gradient
    
    def sense_schumann(self, pos: np.ndarray) -> Dict[str, float]:
        """
        Sense Schumann mode amplitudes at a position.
        
        Returns dict with mode strengths at this location.
        """
        i, j = self._world_to_grid(pos)
        
        result = {}
        for k, mode in enumerate(self.modes):
            # Mode has spatial pattern based on wavelength
            kx = 2 * np.pi / mode.wavelength
            spatial_pattern = np.cos(kx * i * self.dx) * np.cos(kx * j * self.dx)
            result[f'mode_{k}'] = mode.amplitude * spatial_pattern
        
        return result
    
    def evolve(self, dt: float = 1.0):
        """
        Evolve the field by one timestep.
        
        Uses wave equation with damping and source terms:
        ∂²ψ/∂t² = c²∇²ψ - γ∂ψ/∂t + source
        """
        self.step_count += 1
        
        # Compute Laplacian (periodic boundaries)
        # Using finite differences on complex field
        lap_real = laplace(self.psi.real, mode='wrap')
        lap_imag = laplace(self.psi.imag, mode='wrap')
        laplacian = lap_real + 1j * lap_imag
        
        # Wave equation acceleration
        c2 = self.wave_speed ** 2
        acceleration = c2 * laplacian / (self.dx ** 2)
        
        # Damping
        acceleration -= self.damping * self.dpsi_dt
        
        # Add emissions (source term) - guard against NaN
        emission_safe = np.nan_to_num(self.emission_buffer, nan=0.0)
        acceleration += emission_safe * 0.1
        
        # Velocity Verlet integration
        self.psi += self.dpsi_dt * dt + 0.5 * acceleration * dt * dt
        self.dpsi_dt += acceleration * dt
        
        # NaN cleanup - replace any NaN with zero
        self.psi = np.nan_to_num(self.psi, nan=0.0)
        self.dpsi_dt = np.nan_to_num(self.dpsi_dt, nan=0.0)
        
        # Stability clamp - prevent runaway amplitudes
        max_amp = 100.0
        amp = np.abs(self.psi)
        mask = amp > max_amp
        if np.any(mask):
            self.psi[mask] *= max_amp / amp[mask]
        
        # Evolve Schumann modes (they modulate the background)
        self._evolve_modes(dt)
        
        # Add mode contributions to field
        self._apply_modes()
        
        # Clear emission buffer for next frame
        self.emission_buffer *= 0.0
        
        # Update statistics
        self._update_statistics()
    
    def _evolve_modes(self, dt: float):
        """Evolve the resonant Schumann modes."""
        for mode in self.modes:
            # Phase advances at mode frequency
            mode.phase += 2 * np.pi * mode.frequency * dt
            mode.phase = mode.phase % (2 * np.pi)
            
            # Amplitude decays based on Q factor
            decay = np.exp(-mode.frequency / mode.q_factor * dt)
            mode.amplitude *= decay
            
            # Modes can be excited by field energy at their frequency
            # (simplified - full version would do Fourier analysis)
            mode.amplitude += 0.001 * np.sqrt(self.total_energy + 0.001)
    
    def _apply_modes(self):
        """Apply Schumann mode patterns to the field."""
        x = np.arange(self.resolution) * self.dx
        y = np.arange(self.resolution) * self.dx
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for mode in self.modes:
            if mode.amplitude < 0.001:
                continue
                
            k = 2 * np.pi / mode.wavelength
            # Standing wave pattern (cavity mode)
            pattern = np.cos(k * X) * np.cos(k * Y)
            
            # Add with current phase
            contribution = mode.amplitude * pattern * np.exp(1j * mode.phase)
            self.psi += contribution * 0.01
    
    def _update_statistics(self):
        """Update field statistics."""
        # Total energy
        self.total_energy = np.sum(np.abs(self.psi) ** 2)
        
        # Mode energies
        for i, mode in enumerate(self.modes):
            self.mode_energies[i] = mode.energy()
        
        # Global coherence (how aligned are the phases?)
        if self.total_energy > 0.01:
            mean_psi = np.mean(self.psi)
            self.coherence = np.abs(mean_psi) / np.sqrt(self.total_energy / self.psi.size)
        else:
            self.coherence = 0.0
    
    def get_amplitude_field(self) -> np.ndarray:
        """Get amplitude for visualization."""
        return np.abs(self.psi)
    
    def get_phase_field(self) -> np.ndarray:
        """Get phase for visualization."""
        return np.angle(self.psi)
    
    def get_mode_spectrum(self) -> np.ndarray:
        """Get Schumann mode spectrum for visualization."""
        return self.mode_energies.copy()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get field statistics."""
        return {
            'total_energy': self.total_energy,
            'coherence': self.coherence,
            'mode_0_energy': self.mode_energies[0] if len(self.mode_energies) > 0 else 0,
            'mode_1_energy': self.mode_energies[1] if len(self.mode_energies) > 1 else 0,
            'step': self.step_count,
        }
    
    def to_dict(self) -> dict:
        """Serialize for saving."""
        return {
            'psi_real': self.psi.real.tolist(),
            'psi_imag': self.psi.imag.tolist(),
            'dpsi_dt_real': self.dpsi_dt.real.tolist(),
            'dpsi_dt_imag': self.dpsi_dt.imag.tolist(),
            'modes': [
                {
                    'frequency': m.frequency,
                    'wavelength': m.wavelength,
                    'amplitude': m.amplitude,
                    'phase': m.phase,
                    'q_factor': m.q_factor,
                }
                for m in self.modes
            ],
            'step_count': self.step_count,
            'world_size': self.world_size,
            'resolution': self.resolution,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'GlobalResonanceField':
        """Deserialize from saved state."""
        field = cls(
            world_size=d['world_size'],
            resolution=d['resolution']
        )
        field.psi = np.array(d['psi_real']) + 1j * np.array(d['psi_imag'])
        field.dpsi_dt = np.array(d['dpsi_dt_real']) + 1j * np.array(d['dpsi_dt_imag'])
        field.step_count = d['step_count']
        
        for i, mode_data in enumerate(d.get('modes', [])):
            if i < len(field.modes):
                field.modes[i].frequency = mode_data['frequency']
                field.modes[i].wavelength = mode_data['wavelength']
                field.modes[i].amplitude = mode_data['amplitude']
                field.modes[i].phase = mode_data['phase']
                field.modes[i].q_factor = mode_data['q_factor']
        
        return field


# =============================================================================
# CREATURE FIELD COUPLING
# =============================================================================

class CreatureFieldCoupler:
    """
    Handles coupling between creatures and the global resonance field.
    
    This replaces direct creature-creature interactions with field-mediated ones:
    
    OLD (direct, not parallelizable):
        for other in creatures:
            if close(self, other):
                interact(self, other)  # Writes to other!
    
    NEW (field-mediated, parallelizable):
        # Emission phase (parallel-safe: writes to field buffer)
        emission = self.torus.get_emission()
        field.emit(self.pos, emission)
        
        # Evolution phase (single-threaded)
        field.evolve()
        
        # Sensing phase (parallel-safe: reads from field)
        value, gradient = field.sense(self.pos)
        self.torus.receive_field(value, gradient)
    """
    
    def __init__(self, emission_strength: float = 0.1,
                 coupling_strength: float = 0.05):
        """
        Args:
            emission_strength: How strongly creatures emit to field
            coupling_strength: How strongly field affects creatures
        """
        self.emission_strength = emission_strength
        self.coupling_strength = coupling_strength
    
    def get_creature_emission(self, creature) -> complex:
        """
        Get emission from a creature to the field.
        
        Emission is based on:
        - Torus brain state (phase and amplitude)
        - Efferent channel activity (motor commands emit!)
        - Arousal/activity level
        - Species-specific factors
        """
        if not hasattr(creature, 'torus') or creature.torus is None:
            return 0.0 + 0j
        
        # Get dominant phase and amplitude from torus
        psi = creature.torus.psi
        max_idx = np.argmax(np.abs(psi))
        dominant_psi = psi.flat[max_idx]
        
        # Scale by arousal
        arousal = np.std(np.abs(psi) ** 2)
        
        # Base emission from torus
        emission = dominant_psi * arousal * self.emission_strength
        
        # === EFFERENT CHANNEL CONTRIBUTION ===
        # Active motor commands emit strongly (action radiates!)
        if hasattr(creature, 'efferent'):
            for name, channel in creature.efferent.items():
                # Get channel energy
                channel_energy = np.sum(np.abs(channel.u) ** 2) if hasattr(channel, 'u') else 0
                if channel_energy > 0.01:
                    # Efferent activity adds to emission with channel's phase
                    channel_phase = np.angle(np.sum(channel.u)) if hasattr(channel, 'u') else 0
                    emission += channel_energy * 0.5 * np.exp(1j * channel_phase)
        
        # Species modifier
        if hasattr(creature, 'species'):
            if creature.species.name == 'Herbie':
                emission *= 1.5  # Herbies are more "social" in the field
            elif creature.species.diet == 'carnivore':
                emission *= 0.7  # Predators emit less (stealth)
        
        return emission
    
    def apply_field_to_creature(self, creature, field_value: complex, 
                                 gradient: np.ndarray, mode_strengths: Dict[str, float]):
        """
        Apply field influence to a creature.
        
        The field affects:
        - Torus phase (entrainment to local field)
        - Movement tendency (gradient following)
        - Arousal level (field intensity)
        - Afferent channels (field signal as environmental input)
        - Skeleton resonance (piezoelectric coupling)
        """
        if not hasattr(creature, 'torus') or creature.torus is None:
            return
        
        # Phase entrainment: torus tends to align with local field phase
        field_phase = np.angle(field_value)
        field_amp = np.abs(field_value)
        
        if field_amp > 0.01:
            # Inject a small signal at field phase
            creature.torus.inject_at_phase(field_phase, 
                                           field_amp * self.coupling_strength)
        
        # === AFFERENT CHANNEL COUPLING ===
        # Field carries information that arrives via environmental sensing
        if hasattr(creature, 'afferent') and field_amp > 0.02:
            # env_reward channel: high field = good location (social)
            if 'env_reward' in creature.afferent:
                creature.afferent['env_reward'].nucleate(
                    field_amp * self.coupling_strength * 2.0,
                    field_phase / (2 * np.pi)  # Phase as position on channel
                )
            
            # proprioception: field gradient as "body sense" of environment
            if 'proprioception' in creature.afferent:
                grad_mag = np.linalg.norm(gradient)
                if grad_mag > 0.001:
                    creature.afferent['proprioception'].nucleate(
                        grad_mag * self.coupling_strength,
                        0.5
                    )
        
        # === SKELETON PIEZOELECTRIC COUPLING ===
        # Field oscillations create mechanical stress via piezo effect
        if hasattr(creature, 'skeleton') and field_amp > 0.03:
            # Fundamental Schumann mode creates coherent vibration
            mode_0_amp = mode_strengths.get('mode_0', 0)
            if mode_0_amp > 0.01:
                # Inject gentle oscillation into skeleton
                creature.skeleton.inject_vibration(
                    frequency=self.coupling_strength * mode_0_amp,
                    amplitude=mode_0_amp * 0.1
                )
        
        # Gradient influence on movement
        if hasattr(creature, 'vel') and np.linalg.norm(gradient) > 0.001:
            # Guard against NaN
            if not np.any(np.isnan(gradient)):
                # Creatures tend to move toward higher field regions (social attraction)
                # But predators might avoid (stealth)
                if hasattr(creature, 'species') and creature.species.diet == 'carnivore':
                    creature.vel -= gradient * 0.1 * self.coupling_strength
                else:
                    creature.vel += gradient * 0.1 * self.coupling_strength
        
        # Schumann mode entrainment
        # Fundamental mode (mode_0) promotes calm/coherent states
        # Higher modes promote activity
        if 'mode_0' in mode_strengths and mode_strengths['mode_0'] > 0.01:
            # Fundamental mode calms the torus
            creature.torus.g *= (1.0 - 0.01 * mode_strengths['mode_0'])
        
        if 'mode_2' in mode_strengths and mode_strengths['mode_2'] > 0.01:
            # Higher mode increases activity
            if hasattr(creature, 'metabolism'):
                creature.metabolism.hunger += 0.001 * mode_strengths['mode_2']
