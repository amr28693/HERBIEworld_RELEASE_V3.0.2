#!/usr/bin/env python3
"""
Cosmological Launcher - Watch the universe emerge from the void

Flow:
1. Audio starts immediately - modulates the soliton in real-time
2. User watches the "pre-cosmological soup" - soliton + audio interference
3. Press SPACE = collapse at current state (the "observation")
4. Big Bang animation
5. HERBIE World launches with those exact parameters

Usage:
    python cosmological_launcher.py [seed]
    python cosmological_launcher.py 42
    python cosmological_launcher.py        # Random seed from system time
"""

import numpy as np
import sys
import os
import time
import json
from dataclasses import dataclass
from typing import Optional, Dict

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib required: pip install matplotlib")

try:
    import sounddevice as sd
    HAS_AUDIO = True
except (ImportError, OSError):
    HAS_AUDIO = False
    sd = None
    print("sounddevice not available - will use synthetic audio")


# === COSMIC COLORMAP ===
def create_cosmic_cmap():
    colors = [
        (0.0, 0.0, 0.05),
        (0.1, 0.0, 0.3),
        (0.2, 0.0, 0.5),
        (0.1, 0.2, 0.7),
        (0.0, 0.5, 0.9),
        (0.3, 0.8, 1.0),
        (1.0, 1.0, 1.0),
    ]
    return LinearSegmentedColormap.from_list('cosmic', colors)

COSMIC_CMAP = create_cosmic_cmap()


def blackbody_color(T: float) -> tuple:
    """
    Convert blackbody temperature to RGB color.
    
    Uses Planck's law approximation for visible spectrum.
    T in Kelvin, returns (R, G, B) normalized to [0, 1].
    
    Based on algorithm by Tanner Helland.
    """
    # Clamp temperature
    T = max(1000, min(T, 40000))
    T = T / 100.0
    
    # Red
    if T <= 66:
        r = 255
    else:
        r = T - 60
        r = 329.698727446 * (r ** -0.1332047592)
        r = max(0, min(255, r))
    
    # Green
    if T <= 66:
        g = T
        g = 99.4708025861 * np.log(g) - 161.1195681661
        g = max(0, min(255, g))
    else:
        g = T - 60
        g = 288.1221695283 * (g ** -0.0755148492)
        g = max(0, min(255, g))
    
    # Blue
    if T >= 66:
        b = 255
    elif T <= 19:
        b = 0
    else:
        b = T - 10
        b = 138.5177312231 * np.log(b) - 305.0447927307
        b = max(0, min(255, b))
    
    return (r / 255.0, g / 255.0, b / 255.0)


# =============================================================================
# 3D NLSE QUANTUM FOAM - Real wave mechanics for the pre-cosmological void
# =============================================================================

class QuantumFoam3D:
    """
    3D Nonlinear Schrodinger Equation evolution for the cosmological intro.
    
    Physics:
    - Initialize with KdV soliton expanded via Green's function (spherical shell)
    - Evolve under NLSE: i∂ψ/∂t = -∇²ψ + g|ψ|²ψ
    - Audio couples to the field as perturbations
    
    Uses split-step Fourier method for efficient evolution.
    """
    
    def __init__(self, N: int = 24, L: float = 10.0, g: float = 1.0):
        """
        Initialize 3D quantum foam.
        
        Args:
            N: Grid points per dimension (N³ total)
            L: Box size (physical units)
            g: NLSE nonlinearity strength
        """
        self.N = N
        self.L = L
        self.g = g
        
        # Spatial grid
        self.dx = L / N
        x = np.linspace(-L/2, L/2, N, endpoint=False)
        self.x, self.y, self.z = np.meshgrid(x, x, x, indexing='ij')
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2) + 1e-6
        
        # Momentum grid for FFT
        k = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.kx, self.ky, self.kz = np.meshgrid(k, k, k, indexing='ij')
        self.k_squared = self.kx**2 + self.ky**2 + self.kz**2
        
        # Wavefunction - initialize with spherical shell (Green's function expanded soliton)
        self.psi = self._init_soliton_shell()
        
        # Time tracking
        self.t = 0.0
        self.dt = 0.02
        
        # Precompute kinetic propagator (for split-step)
        self.kinetic_prop = np.exp(-0.5j * self.k_squared * self.dt)
        
    def _init_soliton_shell(self, r0: float = 3.0, width: float = 1.0, amp: float = 2.0):
        """
        Initialize with KdV soliton expanded to 3D via Green's function.
        
        A 1D soliton u(x) = sech²((x-x0)/w) expanded spherically becomes
        a shell: ψ(r) = sech²((r-r0)/w) / r
        
        Args:
            r0: Shell radius
            width: Soliton width
            amp: Amplitude
        """
        # Soliton profile (sech² is the KdV soliton shape)
        profile = amp / (np.cosh((self.r - r0) / width)**2 + 0.1)
        
        # Divide by r for Green's function expansion (with softening)
        shell = profile / (self.r + 0.5)
        
        # Add random phase for quantum character
        phase = np.random.random((self.N, self.N, self.N)) * 2 * np.pi * 0.3
        
        # Complex wavefunction
        psi = shell * np.exp(1j * phase)
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx**3)
        return psi / (norm + 1e-6) * 5.0
    
    def step(self, audio_amp: float = 0.0, audio_phase: float = 0.0):
        """
        Evolve one timestep using split-step Fourier method.
        
        Split-step: exp(-iHdt) ≈ exp(-iVdt/2) exp(-iTdt) exp(-iVdt/2)
        where T = kinetic, V = potential (including nonlinearity)
        
        Args:
            audio_amp: Current audio amplitude (0-1)
            audio_phase: Phase from audio for spatial modulation
        """
        # === NONLINEAR HALF-STEP ===
        # V = g|ψ|² (self-interaction)
        density = np.abs(self.psi)**2
        
        # Audio creates localized perturbations
        if audio_amp > 0.01:
            # Audio injects energy at random locations
            audio_kick = audio_amp * 0.5 * np.sin(
                self.kx * 0.3 * np.cos(audio_phase) + 
                self.ky * 0.3 * np.sin(audio_phase) +
                self.t * 2
            )
            # Transform back to real space for the kick
            audio_potential = np.real(np.fft.ifftn(audio_kick * np.fft.fftn(density)))
        else:
            audio_potential = 0.0
        
        nonlinear_phase = -0.5j * self.dt * (self.g * density + audio_potential * 0.3)
        self.psi = self.psi * np.exp(nonlinear_phase)
        
        # === KINETIC STEP (in Fourier space) ===
        psi_k = np.fft.fftn(self.psi)
        psi_k = psi_k * self.kinetic_prop
        self.psi = np.fft.ifftn(psi_k)
        
        # === NONLINEAR HALF-STEP ===
        density = np.abs(self.psi)**2
        nonlinear_phase = -0.5j * self.dt * self.g * density
        self.psi = self.psi * np.exp(nonlinear_phase)
        
        # Gentle damping at boundaries to prevent reflections
        boundary_damp = 1.0 - 0.1 * np.exp(-((self.r - self.L/2 + 1)**2) / 0.5)
        boundary_damp = np.clip(boundary_damp, 0.9, 1.0)
        self.psi *= boundary_damp
        
        # Add small noise to maintain quantum character
        noise = 0.001 * (np.random.randn(self.N, self.N, self.N) + 
                         1j * np.random.randn(self.N, self.N, self.N))
        self.psi += noise
        
        self.t += self.dt
        
    def inject_soliton(self, direction: np.ndarray, amplitude: float = 0.5):
        """
        Inject a new soliton pulse (from audio burst).
        
        Args:
            direction: 3D direction vector
            amplitude: Pulse strength
        """
        # Normalize direction
        d = direction / (np.linalg.norm(direction) + 1e-6)
        
        # Create directional pulse
        proj = self.x * d[0] + self.y * d[1] + self.z * d[2]
        pulse = amplitude / (np.cosh((proj - 2.0) / 0.8)**2 + 0.1)
        pulse = pulse / (self.r + 0.5)
        
        # Add with random phase
        self.psi += pulse * np.exp(1j * self.t * 3)
        
    def get_render_points(self, threshold: float = 0.05, max_points: int = 2000):
        """
        Get points for 3D visualization.
        
        Returns points where |ψ|² > threshold, with colors from phase.
        
        Args:
            threshold: Density threshold for visibility
            max_points: Maximum points to return (for performance)
            
        Returns:
            (positions, colors, sizes) for scatter plot
        """
        density = np.abs(self.psi)**2
        phase = np.angle(self.psi)
        
        # Find points above threshold
        mask = density > threshold * np.max(density)
        
        # Get coordinates and values
        xs = self.x[mask]
        ys = self.y[mask]
        zs = self.z[mask]
        ds = density[mask]
        ps = phase[mask]
        
        # Subsample if too many points
        n_points = len(xs)
        if n_points > max_points:
            indices = np.random.choice(n_points, max_points, replace=False)
            xs, ys, zs = xs[indices], ys[indices], zs[indices]
            ds, ps = ds[indices], ps[indices]
        
        # Positions
        positions = np.column_stack([xs, ys, zs])
        
        # Colors from phase (cyan-white-magenta cycle)
        # Map phase [-π, π] to color
        hue = (ps + np.pi) / (2 * np.pi)  # 0 to 1
        
        # HSV-like mapping: cyan (0.5) -> white (0.75) -> magenta (1.0) -> cyan
        colors = np.zeros((len(ps), 4))
        colors[:, 0] = 0.3 + 0.7 * np.abs(np.sin(hue * np.pi))  # R
        colors[:, 1] = 0.5 + 0.5 * np.cos(hue * 2 * np.pi)      # G  
        colors[:, 2] = 0.8 + 0.2 * np.sin(hue * np.pi + 0.5)    # B
        colors[:, 3] = np.clip(ds / np.max(ds) * 0.8 + 0.2, 0.3, 1.0)  # Alpha
        
        # Sizes from density
        sizes = 10 + 40 * (ds / (np.max(ds) + 1e-6))
        
        return positions, colors, sizes
    
    def get_shell_radius(self) -> float:
        """Get current effective shell radius (for expansion tracking)."""
        density = np.abs(self.psi)**2
        # Weighted average radius
        total = np.sum(density)
        if total < 1e-6:
            return 0.0
        return float(np.sum(self.r * density) / total)
    
    def get_total_energy(self) -> float:
        """Get total field energy."""
        return float(np.sum(np.abs(self.psi)**2) * self.dx**3)


# Global foam instance (created when needed)
_quantum_foam: Optional[QuantumFoam3D] = None

def get_quantum_foam() -> QuantumFoam3D:
    """Get or create the 3D quantum foam instance."""
    global _quantum_foam
    if _quantum_foam is None:
        print("[Cosmos] Initializing 3D NLSE quantum foam (24³ grid)...")
        _quantum_foam = QuantumFoam3D(N=24, L=12.0, g=0.8)
        print(f"[Cosmos] Quantum foam ready. Initial energy: {_quantum_foam.get_total_energy():.2f}")
    return _quantum_foam

def reset_quantum_foam():
    """Reset the foam for a new run."""
    global _quantum_foam
    _quantum_foam = None


# =============================================================================
# STATE MACHINE
# =============================================================================
@dataclass
class CosmosState:
    seed: int
    phase: str = "evolving"  # evolving -> collapsing -> collapsed -> launching
    tau: float = 0.0
    psi: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    collapsed_params: Optional[Dict] = None
    audio_level: float = 0.0
    audio_loudness: float = 0.0
    message: str = "Pre-cosmological void..."


# === SOLITON MATH ===
def generate_soliton(x, seed):
    """Generate multi-soliton from seed."""
    np.random.seed(seed)
    n_solitons = 1 + (seed % 3)
    
    psi = np.zeros_like(x, dtype=complex)
    for i in range(n_solitons):
        sub_seed = seed * (i + 1) + i * 12345
        np.random.seed(sub_seed)
        
        amp = 0.5 + np.random.random() * 1.5
        width = 0.3 + np.random.random() * 1.5
        pos = np.random.uniform(-5, 5)
        phase = np.random.random() * 2 * np.pi
        vel = np.random.randn() * 0.3
        
        envelope = amp / np.cosh((x - pos) / width)
        psi += envelope * np.exp(1j * (vel * x + phase))
    
    psi += (np.random.randn(len(x)) + 1j * np.random.randn(len(x))) * 0.03
    return psi


def evolve_step(psi, x, dt=0.02, g=-1.0):
    """Single imaginary-time evolution step."""
    dx = x[1] - x[0]
    N = len(x)
    
    k = np.fft.fftfreq(N, dx / (2 * np.pi))
    psi_k = np.fft.fft(psi)
    psi_k *= np.exp(-k**2 * dt)
    psi = np.fft.ifft(psi_k)
    psi *= np.exp(-g * np.abs(psi)**2 * dt)
    
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    if norm > 1e-10:
        psi *= np.sqrt(N * dx) / norm
    
    return psi


def collapse_to_params(psi, x, audio_loudness: float) -> Dict:
    """Extract world parameters from collapsed soliton + audio."""
    
    intensity = np.abs(psi)**2
    phase = np.angle(psi)
    
    # Soliton properties
    soliton_amp = np.max(np.abs(psi))
    soliton_amp_norm = soliton_amp / (1.0 + soliton_amp)
    
    psi_fft = np.fft.fft(psi)
    fft_mags = np.abs(psi_fft[:6])
    fft_mags = fft_mags / (np.sum(fft_mags) + 1e-10)
    
    # Peak phase for terrain seed
    peak_idx = np.argmax(intensity)
    peak_phase = phase[peak_idx]
    
    # Width estimate
    total = np.sum(intensity)
    if total > 1e-10:
        mean_x = np.sum(np.arange(len(psi)) * intensity) / total
        var_x = np.sum((np.arange(len(psi)) - mean_x)**2 * intensity) / total
        width = np.sqrt(var_x) if var_x > 0 else 1.0
    else:
        width = 1.0
    
    # === MAP TO WORLD PARAMETERS ===
    
    # World size: 80-140 based on soliton + loudness
    world_size = 80.0 + 20.0 * soliton_amp_norm + 40.0 * audio_loudness
    
    # Populations
    base_pop = 3 + int(5 * np.clip(width / 50, 0, 1))
    loudness_boost = 1.0 + 0.8 * audio_loudness
    
    n_herbies = max(4, int(base_pop * loudness_boost * 1.5))
    n_blobs = max(2, int(base_pop * loudness_boost * 0.8))
    n_bipeds = max(2, int(base_pop * loudness_boost * 1.0))
    n_apex = max(1, int(1 + 4 * audio_loudness))
    n_scavengers = max(1, int(base_pop * 0.5))
    
    # Terrain seed from phase
    terrain_seed = int(abs(peak_phase * 1e7)) % (2**31)
    
    # Ecology
    nutrient_density = 0.4 + 0.3 * soliton_amp_norm + 0.3 * audio_loudness
    
    # Day/night
    day_length = int(400 + 300 * soliton_amp_norm + 300 * audio_loudness)
    
    # Creature stats
    base_metabolism = 0.7 + 0.2 * soliton_amp_norm + 0.2 * audio_loudness
    base_speed = 0.6 + 0.2 * soliton_amp_norm + 0.3 * audio_loudness
    
    # Genetics from FFT
    allele_freqs = {
        'chromosome_0': 0.3 + 0.4 * fft_mags[0] + 0.1 * audio_loudness,
        'chromosome_1': 0.3 + 0.4 * fft_mags[1] + 0.1 * audio_loudness,
        'chromosome_2': 0.3 + 0.4 * fft_mags[2] + 0.1 * (1 - audio_loudness),
        'chromosome_3': 0.3 + 0.4 * fft_mags[3] + 0.1 * audio_loudness,
    }
    
    return {
        'world_size': world_size,
        'n_herbies': n_herbies,
        'n_blobs': n_blobs,
        'n_bipeds': n_bipeds,
        'n_apex': n_apex,
        'n_scavengers': n_scavengers,
        'terrain_seed': terrain_seed,
        'nutrient_density': np.clip(nutrient_density, 0.2, 1.0),
        'day_length': day_length,
        'base_metabolism': np.clip(base_metabolism, 0.5, 1.5),
        'base_speed': np.clip(base_speed, 0.5, 1.5),
        'initial_allele_frequencies': allele_freqs,
        'audio_loudness': audio_loudness,
        'soliton_amplitude': float(soliton_amp),
    }


# === AUDIO MONITORING (starts immediately) ===
class AudioMonitor:
    """
    Continuous audio level monitoring - runs from the start.
    No triggering logic - just measures current audio level.
    """
    
    def __init__(self, gain: float = 5.0):
        """
        Args:
            gain: Amplification factor for audio sensitivity
        """
        self.gain = gain
        self.current_level = 0.0
        self.current_loudness = 0.0
        self.running = False
        self.stream = None
        self.history = []
        self.max_history = 100
        
    def start(self):
        if not HAS_AUDIO:
            print("[Audio] No audio device - using synthetic modulation")
            return
        
        self.running = True
        
        def callback(indata, frames, time_info, status):
            if not self.running:
                return
            
            # RMS level with gain
            rms = np.sqrt(np.mean(indata**2)) * self.gain
            self.current_level = min(rms, 1.0)  # Clamp to 1.0
            
            # Convert to loudness (dB normalized, very sensitive)
            db = 20 * np.log10(rms + 1e-10)
            # Very sensitive: -50dB = 0, -10dB = 1 (normal speech ~0.3-0.6)
            self.current_loudness = np.clip((db + 50) / 40, 0, 1)
            
            # Store history
            self.history.append(self.current_loudness)
            if len(self.history) > self.max_history:
                self.history.pop(0)
        
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=44100,
                blocksize=1024,  # Smaller for faster response
                callback=callback
            )
            self.stream.start()
            print("[Audio] Monitoring active (gain={:.1f}x)".format(self.gain))
        except Exception as e:
            print(f"[Audio] Could not start: {e}")
            self.running = False
    
    def stop(self):
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
    
    def get_synthetic_level(self, t):
        """Synthetic audio for when no mic available."""
        # Gentle pulsing with some randomness
        return 0.1 + 0.05 * np.sin(t * 2) + 0.05 * np.random.random()


# === MAIN VISUALIZATION + LAUNCH ===
def run_cosmological_launcher(seed: int = None):
    """
    Main entry point for cosmological launcher.
    
    Flow:
    1. Audio starts immediately - modulates the soliton in real-time
    2. User watches the "pre-cosmological soup" - soliton + audio interference
    3. Press SPACE = collapse at current state (the "observation")
    4. Big Bang animation
    5. Launch HERBIE World
    """
    if not HAS_MPL:
        print("ERROR: matplotlib required")
        return
    
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    
    print("=" * 60)
    print("   C O S M O L O G I C A L   L A U N C H E R")
    print("=" * 60)
    print(f"   Seed: {seed}")
    print()
    print("   You are observing the pre-cosmological void...")
    print("   Sound waves perturb the quantum foam.")
    print()
    print("   [SPACE] - Collapse the wavefunction (create universe)")
    print("   [Q]     - Quit")
    print("=" * 60)
    
    # Initialize
    N = 256
    x = np.linspace(-10, 10, N)
    psi = generate_soliton(x, seed)
    psi_base = psi.copy()  # Store base soliton for audio modulation
    
    # Reset 3D quantum foam for fresh run
    reset_quantum_foam()
    
    state = CosmosState(
        seed=seed,
        x=x,
        psi=psi,
    )
    
    # Start audio monitoring IMMEDIATELY
    audio_monitor = AudioMonitor(gain=10.0)  # Very high gain for sensitivity
    audio_monitor.start()
    
    # Setup figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8), facecolor='black')
    
    # Layout: main soliton view + info panels
    ax_main = fig.add_axes([0.05, 0.25, 0.55, 0.65], facecolor='black')
    ax_audio = fig.add_axes([0.65, 0.55, 0.30, 0.35], facecolor='black')
    ax_info = fig.add_axes([0.65, 0.08, 0.30, 0.40], facecolor='black')
    ax_phase = fig.add_axes([0.05, 0.05, 0.55, 0.15], facecolor='black')
    
    # Status text
    status_text = fig.text(0.5, 0.95, '', ha='center', va='top', 
                           fontsize=14, color='cyan', fontweight='bold')
    
    # State tracking for animation
    collapse_frame = [0]
    should_stop = [False]
    frame_count = [0]
    
    # Key handler - SPACE triggers INSTANT collapse
    def on_key(event):
        if event.key == ' ':
            if state.phase == "evolving":
                # INSTANT COLLAPSE - use current audio state
                state.phase = "collapsing"
                state.audio_loudness = audio_monitor.current_loudness
                collapse_frame[0] = frame_count[0]
                print(f"\n[Cosmos] ** COLLAPSE! **")
                print(f"[Cosmos] Loudness at collapse: {state.audio_loudness:.3f}")
        elif event.key == 'q':
            audio_monitor.stop()
            should_stop[0] = True
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Animation holder for cleanup from within animate()
    anim_holder = [None]
    
    def animate(frame):
        nonlocal psi, psi_base
        
        frame_count[0] = frame
        
        # Check if we should stop
        if should_stop[0]:
            return []
        
        # Get current audio state
        if audio_monitor.running:
            current_audio = audio_monitor.current_loudness
            current_level = audio_monitor.current_level
        else:
            # Synthetic audio when no mic
            current_audio = audio_monitor.get_synthetic_level(frame * 0.05)
            current_level = current_audio * 0.1
        
        # === PHASE: EVOLVING (pre-cosmological soup) ===
        if state.phase == "evolving":
            # DON'T collapse the soliton - keep it dispersed and breathing
            # Only do gentle phase evolution, not imaginary time collapse
            state.tau += 0.02
            
            # === DISPERSED WAVEFORM WITH AUDIO COUPLING ===
            # The base soliton breathes and shifts but stays spread out
            breath = 1.0 + 0.2 * np.sin(state.tau * 0.5)  # Gentle breathing
            shift = 0.5 * np.sin(state.tau * 0.3)  # Slow drift
            
            # Recreate dispersed soliton each frame (doesn't collapse)
            psi = generate_soliton(x, seed)  # Base shape from seed
            
            # Apply breathing and drift
            psi = psi * breath * np.exp(1j * state.tau * 2)
            
            # === AUDIO INTERFERENCE ===
            # Sound creates visible ripples in the quantum foam
            audio_boost = 1.0 + current_audio * 4.0  # Amplitude boost
            
            # Multiple frequency interference based on audio level
            freq1 = 1 + current_audio * 20  # Low freq ripple
            freq2 = 3 + current_audio * 30  # Mid freq
            freq3 = 7 + current_audio * 50  # High freq
            
            phase1 = state.tau * 3 + current_level * 300
            phase2 = state.tau * 5 - current_level * 200
            phase3 = state.tau * 8 + current_level * 500
            
            # Audio creates visible interference patterns
            interference = (
                0.5 * current_audio * np.sin(freq1 * x + phase1) +
                0.3 * current_audio * np.cos(freq2 * x + phase2) +
                0.2 * current_audio * np.sin(freq3 * x + phase3)
            )
            
            # Combine: dispersed soliton + audio ripples
            psi = psi * audio_boost + interference * np.exp(1j * phase1) * 2
            
            # Store state
            state.psi = psi
            state.audio_level = current_level
            state.audio_loudness = current_audio
            
            state.message = f"Quantum foam  tau = {state.tau:.2f}  |  Audio: {current_audio:.2f}\nSound ripples through the void...\n[SPACE to collapse into existence]"
        
        # === PHASE: COLLAPSING ===
        elif state.phase == "collapsing":
            frames_since = frame - collapse_frame[0]
            
            # Quick initial flash (frames 0-15), then fade during Wick rotation
            if frames_since < 15:
                # Bright flash at collapse moment
                pulse = np.sin(frames_since * 0.8) * 0.5 + 0.5
                intensity_flash = max(0, 1.0 - frames_since / 15) * pulse
                fig.patch.set_facecolor((intensity_flash, intensity_flash * 0.9, intensity_flash * 0.7))
                state.message = "#### C O L L A P S E ####"
            elif frames_since < 20:
                fig.patch.set_facecolor('black')
                state.message = "[*] Collapsing to delta [*]"
            elif frames_since < 70:
                state.message = "[o] Inverse Wick Rotation [o]"
            elif frames_since < 110:
                state.message = "[+] Density Perturbations [+]"
            else:
                state.message = "[*] Structure Formation [*]"
            
            if frames_since == 110:
                fig.patch.set_facecolor('black')
                state.collapsed_params = collapse_to_params(psi, x, state.audio_loudness)
                state.phase = "collapsed"
                state.message = "Universe collapsed. Watching cosmic evolution..."
                print("\n[Cosmos] Parameters extracted!")
                for k, v in state.collapsed_params.items():
                    if k != 'initial_allele_frequencies':
                        print(f"   {k}: {v}")
                print("\n[Cosmos] Watching cosmic evolution... (wait for timelapse to complete)")
                print("[Cosmos] DO NOT CLOSE WINDOW - HERBIE World will launch automatically")
        
        # === PHASE: COLLAPSED ===
        elif state.phase == "collapsed":
            frames_since = frame - collapse_frame[0]
            
            if frames_since > 460:  # Extended for all stages
                if not should_stop[0]:  # Only do this once
                    print("[Cosmos] Timelapse complete!")
                    state.phase = "launching"
                    audio_monitor.stop()
                    should_stop[0] = True
                    
                    # Stop animation and close figure
                    import threading
                    def stop_and_close():
                        import time
                        time.sleep(0.1)
                        # Stop the animation event source first
                        try:
                            if anim_holder[0] and anim_holder[0].event_source:
                                anim_holder[0].event_source.stop()
                        except:
                            pass
                        time.sleep(0.1)
                        # Now close
                        try:
                            plt.close('all')
                        except:
                            pass
                    threading.Thread(target=stop_and_close, daemon=True).start()
                return []
        
        # === RENDER ===
        
        ax_main.clear()
        ax_main.set_facecolor('black')
        intensity = np.abs(psi)**2
        
        if state.phase in ("collapsing", "collapsed"):
            # === PHYSICALLY MEANINGFUL COLLAPSE VISUALIZATION ===
            #
            # CE Theory cosmogenesis sequence:
            #
            # 0. INFORMATION VORTEX (frames 0-30):
            #    - In IMAGINARY TIME (tau): information geometry
            #    - Probability amplitude spirals INWARD toward t0
            #    - Like a funnel converging to a point
            #    - The pre-physical realm of entropic degrees of freedom
            #
            # 1. DIRAC DELTA COLLAPSE (frames 30-50):
            #    - At the t0 HYPERPLANE: wavefunction collapses
            #    - Probability distribution spikes to delta
            #    - The "observation" that selects universe parameters
            #    - Information crystallizes to definite values
            #
            # 2. INVERSE WICK ROTATION (frames 50-90):
            #    - Collapsed state rotates from Im axis -> Re axis
            #    - tau (imaginary time) -> t (real time)  
            #    - Information geometry -> Physical dynamics
            #    - Accompanied by metric expansion from the point
            #
            # 3. DENSITY PERTURBATIONS (frames 90-130):
            #    - FFT modes seed primordial fluctuations
            #    - delta_rho/rho grows as a(t)
            #
            # 4+ STRUCTURE, TIMELAPSE, WORMHOLE...
            
            frames_since = frame - collapse_frame[0]
            
            # === SCALE FACTOR a(t) ===
            # Only meaningful after Wick rotation begins
            if frames_since < 50:
                a_t = 0.001  # Pre-physical, essentially zero
            elif frames_since < 90:
                # During Wick rotation: rapid initial expansion
                t_norm = (frames_since - 50) / 40.0
                a_t = 0.001 + t_norm ** 2 * 0.5  # Accelerating from point
            elif frames_since < 150:
                t_norm = (frames_since - 90) / 60.0
                a_t = 0.5 + t_norm ** (2.0/3.0) * 0.5  # Matter-dominated Friedmann
            else:
                a_1 = 1.0
                dt = (frames_since - 150) / 100.0
                H = 0.3 * (1 + state.audio_loudness)
                a_t = a_1 * np.exp(H * dt)
            
            a_t = max(a_t, 0.001)
            
            theta = np.linspace(0, 2*np.pi, N)
            
            # === STAGE 0: INFORMATION VORTEX IN IMAGINARY TIME (frames 0-30) ===
            if frames_since < 30:
                # We are in IMAGINARY TIME (tau) - the information geometry sector
                # Probability amplitude forms a VORTEX spiraling INWARD
                # toward the t0 hyperplane where collapse will occur
                
                vortex_progress = frames_since / 30.0  # 0 to 1
                
                ax_main.set_facecolor('#050510')
                
                # === THE VORTEX: Information funneling toward t0 ===
                # Multiple spiral arms converging to center
                n_arms = 5
                n_points_per_arm = 80
                
                for arm in range(n_arms):
                    arm_phase = arm * 2 * np.pi / n_arms
                    
                    # Spiral parameters: r decreases as angle increases (inward spiral)
                    # The spiral ACCELERATES inward as it approaches center
                    t_spiral = np.linspace(0, 1, n_points_per_arm)
                    
                    # Logarithmic spiral: r = a * e^(-b*theta) - converges to center
                    spiral_angle = arm_phase + t_spiral * (4 * np.pi + vortex_progress * 2 * np.pi)
                    
                    # Radius shrinks toward center (the t0 point)
                    # Outer parts move faster (angular momentum conservation)
                    spiral_r = 8 * np.exp(-t_spiral * (2 + vortex_progress * 2))
                    
                    # Apply rotation - vortex spins faster as it converges
                    rotation_speed = vortex_progress * 3
                    spiral_angle += rotation_speed
                    
                    spiral_x = spiral_r * np.cos(spiral_angle)
                    spiral_y = spiral_r * np.sin(spiral_angle)
                    
                    # Color: blue/cyan for imaginary time (information/entropic)
                    # Intensity increases toward center
                    for i in range(len(spiral_x) - 1):
                        seg_progress = t_spiral[i]
                        # Bluer at edges, whiter at center
                        r_col = 0.3 + seg_progress * 0.5
                        g_col = 0.5 + seg_progress * 0.4
                        b_col = 0.9
                        alpha = 0.3 + seg_progress * 0.5
                        ax_main.plot(spiral_x[i:i+2], spiral_y[i:i+2],
                                    color=(r_col, g_col, b_col), alpha=alpha,
                                    linewidth=1 + seg_progress * 2)
                
                # === Probability density particles flowing inward ===
                n_particles = int(50 + vortex_progress * 100)
                np.random.seed(seed + frames_since)
                
                # Particles distributed in spiral pattern, moving inward
                particle_t = np.random.random(n_particles)
                particle_arm = np.random.randint(0, n_arms, n_particles)
                particle_angle = particle_arm * 2 * np.pi / n_arms + particle_t * 4 * np.pi
                
                # Radius decreases over time (flowing inward)
                base_r = 7 * np.exp(-particle_t * 2)
                # Add inward velocity based on vortex progress
                particle_r = base_r * (1 - vortex_progress * 0.5 * particle_t)
                
                particle_x = particle_r * np.cos(particle_angle + vortex_progress * 2)
                particle_y = particle_r * np.sin(particle_angle + vortex_progress * 2)
                
                # Size increases toward center (probability concentrating)
                particle_sizes = 10 + (1 - particle_r / 8) * 40
                ax_main.scatter(particle_x, particle_y, s=particle_sizes,
                               c='cyan', alpha=0.5, marker='.')
                
                # === Central concentration point (where t0 will occur) ===
                # Gets brighter as vortex converges
                central_brightness = vortex_progress * 0.8
                central_size = 50 + vortex_progress * 150
                ax_main.scatter([0], [0], s=central_size, c='white', 
                               alpha=central_brightness, zorder=10)
                
                # Concentric rings showing probability concentration
                for ring_r in [6, 4, 2, 1]:
                    ring_alpha = 0.2 * (1 - ring_r / 8) * (1 + vortex_progress)
                    ring_theta = np.linspace(0, 2*np.pi, 60)
                    ax_main.plot(ring_r * np.cos(ring_theta), ring_r * np.sin(ring_theta),
                                color='cyan', alpha=ring_alpha, linewidth=0.5)
                
                # === Labels ===
                ax_main.text(0.02, 0.98, 
                            "IMAGINARY TIME (tau)\n"
                            "Information geometry\n"
                            "Probability -> t0",
                            transform=ax_main.transAxes, fontsize=9, color='cyan',
                            va='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                ax_main.text(0, -8.5, "Vortex converging to t0 hyperplane",
                            color='white', fontsize=10, ha='center', alpha=0.8)
                
                ax_main.set_xlim(-10, 10)
                ax_main.set_ylim(-10, 10)
                ax_main.set_aspect('equal')
                ax_main.set_title(f'INFORMATION VORTEX  tau-space  ({vortex_progress*100:.0f}%)',
                                 color='cyan', fontsize=14, fontweight='bold')
            
            # === STAGE 1: DIRAC DELTA COLLAPSE AT t0 (frames 30-50) ===
            elif frames_since < 50:
                # The vortex has converged - now at the t0 HYPERPLANE
                # Probability distribution SPIKES to a Dirac delta
                # This is the "observation" that crystallizes the universe's parameters
                
                collapse_progress = (frames_since - 30) / 20.0  # 0 to 1
                
                ax_main.set_facecolor('#050510')
                
                # === Residual vortex (fading) ===
                if collapse_progress < 0.5:
                    vortex_fade = 1 - collapse_progress * 2
                    n_arms = 5
                    for arm in range(n_arms):
                        arm_phase = arm * 2 * np.pi / n_arms
                        t_spiral = np.linspace(0, 1, 40)
                        spiral_angle = arm_phase + t_spiral * 4 * np.pi + 3
                        spiral_r = 6 * np.exp(-t_spiral * 3) * (1 - collapse_progress)
                        spiral_x = spiral_r * np.cos(spiral_angle)
                        spiral_y = spiral_r * np.sin(spiral_angle)
                        ax_main.plot(spiral_x, spiral_y, color='cyan', 
                                    alpha=vortex_fade * 0.3, linewidth=1)
                
                # === The probability spike forming ===
                # In 2D polar view: probability concentrates at origin
                
                # Width shrinks exponentially toward delta
                initial_width = 3.0
                final_width = 0.1
                current_width = initial_width * np.exp(-collapse_progress * 4) + final_width
                
                # Amplitude grows (conservation of probability)
                amplitude = 1.0 / (current_width + 0.1)
                amplitude = min(amplitude, 5.0)
                
                # Draw radial probability distribution
                r_values = np.linspace(0, 8, 100)
                prob_density = amplitude * np.exp(-(r_values / current_width) ** 2)
                
                # As concentric rings with intensity = probability
                n_rings = 20
                for i, r in enumerate(np.linspace(0.1, 6, n_rings)):
                    prob_at_r = amplitude * np.exp(-(r / current_width) ** 2)
                    ring_alpha = min(0.8, prob_at_r * 0.3)
                    ring_theta = np.linspace(0, 2*np.pi, 60)
                    
                    # Color: cyan -> white -> yellow as it collapses
                    if collapse_progress < 0.5:
                        color = (0.5 + prob_at_r * 0.3, 0.8, 1.0)
                    else:
                        t = (collapse_progress - 0.5) * 2
                        color = (0.8 + t * 0.2, 0.9 + t * 0.1, 1.0 - t * 0.3)
                    
                    ax_main.plot(r * np.cos(ring_theta), r * np.sin(ring_theta),
                                color=color, alpha=ring_alpha, linewidth=1 + prob_at_r)
                
                # === Central DELTA SPIKE ===
                # This is the key moment - the spike at t0
                spike_brightness = 0.3 + collapse_progress * 0.7
                spike_size = 100 + collapse_progress * 400
                
                # Multi-layer glow for the delta
                for glow_r in [3, 2, 1]:
                    glow_size = spike_size * glow_r
                    glow_alpha = spike_brightness * 0.3 / glow_r
                    ax_main.scatter([0], [0], s=glow_size, c='white', alpha=glow_alpha)
                
                # The spike itself
                ax_main.scatter([0], [0], s=spike_size, c='yellow', alpha=spike_brightness, zorder=10)
                
                # === FLASH at final collapse ===
                if collapse_progress > 0.8:
                    flash_intensity = (collapse_progress - 0.8) / 0.2
                    # Screen flash effect
                    flash_rect = plt.Rectangle((-10, -10), 20, 20, 
                                              color='white', alpha=flash_intensity * 0.4)
                    ax_main.add_patch(flash_rect)
                
                # === Labels ===
                ax_main.text(0.02, 0.98, 
                            f"t0 HYPERPLANE\n"
                            f"Width: {current_width:.2f}\n"
                            f"Audio: {state.audio_loudness:.2f}\n"
                            "Probability -> delta",
                            transform=ax_main.transAxes, fontsize=9, color='yellow',
                            va='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                ax_main.set_xlim(-8, 8)
                ax_main.set_ylim(-8, 8)
                ax_main.set_aspect('equal')
                ax_main.set_title(f'DIRAC DELTA COLLAPSE  t0  ({collapse_progress*100:.0f}%)',
                                 color='yellow', fontsize=14, fontweight='bold')
            
            # === STAGE 2: INVERSE WICK ROTATION + METRIC EXPANSION (frames 50-90) ===
            elif frames_since < 90:
                # The collapsed delta now rotates from imaginary to real axis
                # AND simultaneously the universe begins expanding from the point
                # tau -> t  :  information geometry -> physical dynamics
                
                wick_progress = (frames_since - 50) / 40.0  # 0 to 1
                
                # ACCELERATING rotation: slow start (in imaginary time), 
                # accelerating as we approach real time (matter "wakes up")
                accelerated_progress = wick_progress ** 1.5
                
                # INVERSE Wick rotation: start at 90 deg (imaginary), end at 0 deg (real)
                wick_angle = (np.pi / 2) * (1 - accelerated_progress)
                
                # Create the delta spike (narrow gaussian) that we're rotating
                delta_width = 0.15  # Narrow spike
                delta_amplitude = 5.0
                psi_delta = delta_amplitude * np.exp(-(x / delta_width) ** 2)
                
                # The spike rotates in complex plane
                psi_rotated = psi_delta * np.exp(1j * wick_angle)
                
                # Visualize as trajectory in complex plane
                real_part = np.real(psi_rotated)
                imag_part = np.imag(psi_rotated)
                
                # Scale for visibility
                scale = 1.5
                
                # Draw ghost trails showing rotation history
                n_trails = min(6, int(wick_progress * 12) + 1)
                for trail in range(n_trails):
                    trail_progress = max(0, wick_progress - trail * 0.08)
                    trail_angle = (np.pi / 2) * (1 - trail_progress)
                    psi_trail = psi_delta * np.exp(1j * trail_angle)
                    trail_alpha = 0.4 - trail * 0.06
                    ax_main.plot(np.real(psi_trail) * scale, 
                                np.imag(psi_trail) * scale,
                                color='cyan', linewidth=1.5, alpha=max(0.05, trail_alpha))
                
                # Draw the delta spike as a parametric curve in Re-Im plane
                ax_main.plot(real_part * scale, imag_part * scale, 
                            color='white', linewidth=3, alpha=0.95)
                ax_main.fill(real_part * scale, imag_part * scale,
                            color='yellow', alpha=0.5)
                
                # Mark the peak of the delta
                peak_idx = len(x) // 2
                peak_re = real_part[peak_idx] * scale
                peak_im = imag_part[peak_idx] * scale
                ax_main.plot([peak_re], [peak_im], 'o', color='yellow', markersize=12, zorder=10)
                ax_main.plot([0, peak_re], [0, peak_im], '--', color='yellow', linewidth=1.5, alpha=0.7)
                
                # Show the rotation axes
                ax_main.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
                ax_main.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
                
                # Draw the "time plane" axes
                plane_size = 7
                
                # Imaginary time axis (tau) - vertical (where we START)
                tau_x = plane_size * np.cos(np.pi/2)
                tau_y = plane_size * np.sin(np.pi/2)
                ax_main.plot([0, tau_x], [0, tau_y], 'c--', linewidth=2, alpha=0.8)
                ax_main.text(tau_x + 0.3, tau_y * 1.05, 'tau (imaginary time)\nSTART HERE', 
                            color='cyan', fontsize=9, ha='left')
                
                # Real time axis (t) - horizontal (where we END)
                t_x = plane_size * np.cos(0)
                t_y = plane_size * np.sin(0)
                ax_main.plot([0, t_x], [0, t_y], 'r--', linewidth=2, alpha=0.8)
                ax_main.text(t_x * 1.05, t_y + 0.5, 't (real time)\nEND HERE', 
                            color='red', fontsize=9, ha='left')
                
                # Current rotation arrow - shows where psi currently points
                arrow_r = 5.5
                arrow_x = arrow_r * np.cos(wick_angle)
                arrow_y = arrow_r * np.sin(wick_angle)
                ax_main.annotate('', xy=(arrow_x, arrow_y), xytext=(0, 0),
                                arrowprops=dict(arrowstyle='->', color='yellow', lw=2.5))
                
                # Show angle from real axis (decreasing from 90 deg to 0 deg)
                angle_deg = wick_angle * 180 / np.pi
                ax_main.text(arrow_x * 1.15, arrow_y * 1.15, 
                            f'psi(tau)->psi(t)\n{angle_deg:.0f} deg from real',
                            color='yellow', fontsize=11, ha='center', fontweight='bold')
                
                # Arc showing rotation path (from pi/2 down to current angle)
                arc_angles = np.linspace(wick_angle, np.pi/2, 30)
                arc_r = 4
                ax_main.plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles),
                            color='yellow', linewidth=1.5, alpha=0.6)
                # Arrow head on arc showing direction
                if wick_progress > 0.1:
                    ax_main.annotate('', xy=(arc_r * np.cos(wick_angle + 0.1), arc_r * np.sin(wick_angle + 0.1)),
                                    xytext=(arc_r * np.cos(wick_angle + 0.2), arc_r * np.sin(wick_angle + 0.2)),
                                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
                
                # Label axes
                ax_main.set_xlabel('Re(psi) -> physical/oscillatory', color='red', fontsize=10)
                ax_main.set_ylabel('Im(psi) -> information/diffusive', color='cyan', fontsize=10)
                
                # === EXPLANATORY TEXT BOX ===
                if wick_progress < 0.5:
                    explain_text = (
                        "THE DELTA SPIKE:\n"
                        "Collapsed at t0, now lives\n"
                        "on the imaginary axis.\n"
                        "Pure information geometry.\n"
                        "Diffusive, not oscillatory."
                    )
                else:
                    explain_text = (
                        "INVERSE WICK ROTATION:\n"
                        "delta-spike rotates to real axis.\n"
                        "Information -> physics.\n"
                        "Diffusion -> oscillation.\n"
                        "Universe crystallizes."
                    )
                ax_main.text(0.98, 0.02, explain_text, transform=ax_main.transAxes,
                            fontsize=8, color='white', va='bottom', ha='right',
                            fontfamily='monospace', alpha=0.9,
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                view_size = 9
                ax_main.set_xlim(-view_size, view_size)
                ax_main.set_ylim(-view_size, view_size)
                ax_main.set_aspect('equal')
                
                # === METRIC EXPANSION at real axis crossing ===
                # The Big Bang is NOT particles flying outward through space
                # It is SPACE ITSELF expanding - metric expansion
                # Visualize with expanding coordinate grid and cooling plasma
                #
                # CE THEORY: Expansion begins when tau->t rotation is nearly complete
                # A small "bleed" (~8%) represents quantum uncertainty at the transition
                
                if wick_progress > 0.92:
                    expansion_intensity = (wick_progress - 0.92) / 0.08  # 0 to 1 in final 8%
                    
                    # Scale factor increases (space expands)
                    a_expansion = 1.0 + expansion_intensity * 3.0  # 1 to 4
                    
                    # Temperature cools as T ~ 1/a (blackbody)
                    T_normalized = 1.0 / a_expansion  # 1 to 0.25
                    
                    # Blackbody color: hot white -> yellow -> orange -> red
                    if T_normalized > 0.7:
                        bb_r, bb_g, bb_b = 1.0, 1.0, 0.95  # White-hot
                    elif T_normalized > 0.5:
                        t = (T_normalized - 0.5) / 0.2
                        bb_r, bb_g, bb_b = 1.0, 0.85 + 0.15*t, 0.6 + 0.35*t
                    elif T_normalized > 0.3:
                        t = (T_normalized - 0.3) / 0.2
                        bb_r, bb_g, bb_b = 1.0, 0.6 + 0.25*t, 0.3 + 0.3*t
                    else:
                        bb_r, bb_g, bb_b = 1.0, 0.5, 0.2  # Orange-red
                    
                    # === EXPANDING COORDINATE GRID (spacetime stretching) ===
                    # This shows space itself expanding, not motion through space
                    grid_spacing = 1.5 * a_expansion
                    grid_alpha = 0.4 * (1 - expansion_intensity * 0.3)
                    
                    for g in np.arange(-12, 13, 1.5):
                        # Grid lines stretch with scale factor
                        scaled_g = g * a_expansion / 2
                        if abs(scaled_g) < view_size:
                            ax_main.axhline(scaled_g, color='cyan', alpha=grid_alpha * 0.5, linewidth=0.5)
                            ax_main.axvline(scaled_g, color='cyan', alpha=grid_alpha * 0.5, linewidth=0.5)
                    
                    # === HOMOGENEOUS PLASMA FIELD ===
                    # Early universe was uniformly hot - no discrete particles
                    # Create smooth radial glow representing the plasma
                    n_glow = 80
                    glow_r = np.linspace(0, 8 * a_expansion, n_glow)
                    for i, r in enumerate(glow_r):
                        if r > 0.1:
                            # Intensity falls off, but universe is uniform
                            # This represents our observable horizon expanding
                            glow_alpha = 0.6 * (1 - i/n_glow) * expansion_intensity
                            circle = plt.Circle((0, 0), r, fill=False, 
                                               color=(bb_r, bb_g, bb_b), 
                                               alpha=glow_alpha, linewidth=1.5)
                            ax_main.add_patch(circle)
                    
                    # Central bright region (primordial plasma)
                    central_glow = plt.Circle((0, 0), 2 * a_expansion, 
                                             color=(bb_r, bb_g, bb_b), 
                                             alpha=0.3 * expansion_intensity)
                    ax_main.add_patch(central_glow)
                    
                    # === DENSITY PERTURBATIONS ===
                    # The soliton modes seed primordial fluctuations
                    # These are tiny ripples in the otherwise uniform field
                    # delta_rho/rho ~ 10^-5 at recombination
                    psi_fft = np.fft.fft(psi)
                    n_modes = 8
                    for mode_i in range(1, n_modes):
                        mode_amp = np.abs(psi_fft[mode_i]) / (np.max(np.abs(psi_fft)) + 1e-10)
                        if mode_amp > 0.1:
                            # Perturbation as subtle density wave
                            k_mode = mode_i * 0.5
                            phase_mode = np.angle(psi_fft[mode_i])
                            wave_theta = np.linspace(0, 2*np.pi, 60)
                            wave_r = 3 * a_expansion * (1 + 0.05 * mode_amp * np.sin(k_mode * wave_theta + phase_mode))
                            ax_main.plot(wave_r * np.cos(wave_theta), wave_r * np.sin(wave_theta),
                                        color=(bb_r, bb_g, bb_b), alpha=0.4 * mode_amp, linewidth=1)
                    
                    # Scale factor annotation
                    ax_main.text(0.02, 0.98, f"a(t) = {a_expansion:.2f}\nT ~ {T_normalized:.2f} T_0\nSpace expanding",
                                transform=ax_main.transAxes, fontsize=9, color='cyan',
                                va='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                # Title with physics explanation
                if wick_progress < 0.3:
                    subtitle = "In the void (imaginary time)"
                elif wick_progress < 0.7:
                    subtitle = "Rotating into physical reality..."
                elif wick_progress < 0.9:
                    subtitle = "Emerging into real time"
                else:
                    subtitle = "** BIG BANG! **"
                
                ax_main.set_title(f'INVERSE WICK ROTATION  tau -> t  ({wick_progress*100:.0f}%)\n{subtitle}', 
                                 color='cyan' if wick_progress < 0.9 else 'yellow', 
                                 fontsize=13, fontweight='bold')
            
            # === STAGE 3: DENSITY PERTURBATIONS (frames 90-130) ===
            elif frames_since < 130:
                # FFT modes as primordial density perturbations
                # deltarho/rho grows as delta ~ a(t) in matter-dominated era
                # ANIMATED: modes propagate and interfere over time
                
                perturbation_progress = (frames_since - 90) / 40.0
                time_phase = (frames_since - 90) * 0.4  # FASTER animation
                
                # Compute power spectrum from soliton FFT
                psi_fft = np.fft.fft(psi)
                k_modes = np.fft.fftfreq(N, d=(x[1]-x[0]))
                power_spectrum = np.abs(psi_fft) ** 2
                
                # Only positive k modes
                k_pos = k_modes[:N//2]
                P_k = power_spectrum[:N//2]
                P_k_norm = P_k / (np.max(P_k) + 1e-10)
                
                # Growth factor: perturbations grow as delta ~ a(t)
                # Make growth more dramatic and visible
                growth = 0.3 + perturbation_progress * 3.0  # 0.3 to 3.3
                
                # Create 2D density field from superposition of modes
                nx, ny = 150, 150  # Higher resolution for smoother animation
                extent_size = 6 + perturbation_progress * 6  # Expand view more
                xx = np.linspace(-extent_size, extent_size, nx)
                yy = np.linspace(-extent_size, extent_size, ny)
                X, Y = np.meshgrid(xx, yy)
                
                density = np.ones_like(X)
                
                # Add perturbations from dominant modes - ANIMATED
                n_modes = min(30, N//4)
                for i in range(1, n_modes):
                    if P_k_norm[i] < 0.02:
                        continue
                    
                    k = abs(k_pos[i]) + 0.1
                    amplitude = P_k_norm[i] * growth * 0.5
                    phase_k = np.angle(psi_fft[i])
                    
                    # Seeded random direction for this mode
                    np.random.seed(seed + i)
                    angle = np.random.random() * 2 * np.pi
                    
                    # ANIMATE: phase evolves with time (modes propagate)
                    # Higher k modes oscillate faster (dispersion relation)
                    omega = np.sqrt(k + 0.5) * 0.8  # Faster oscillation
                    animated_phase = phase_k + time_phase * omega
                    
                    # Add this Fourier mode as a propagating plane wave
                    wave = amplitude * np.cos(k * (X * np.cos(angle) + Y * np.sin(angle)) + animated_phase)
                    density += wave
                
                # Normalize density with contrast enhancement
                density_centered = density - np.mean(density)
                density_norm = density_centered / (np.std(density_centered) * 2.5 + 1e-10)
                density_display = np.clip(0.5 + density_norm * 0.5, 0, 1)
                
                # Temperature T ~ 1/a(t)
                T_initial = 15000
                T_current = T_initial / (0.5 + perturbation_progress * 1.5)
                bb_color = blackbody_color(T_current)
                
                # Plot density field with better colormap
                from matplotlib.colors import LinearSegmentedColormap
                density_cmap = LinearSegmentedColormap.from_list('density', 
                    [(0, 0, 0.1), bb_color, (1, 1, 0.9)])
                
                ax_main.imshow(density_display, extent=[-extent_size, extent_size, -extent_size, extent_size], 
                              origin='lower', cmap=density_cmap, alpha=0.9)
                
                # Contours showing structure - animate threshold
                base_level = 0.5 + 0.1 * np.sin(time_phase * 0.5)
                contour_levels = [base_level - 0.15, base_level, base_level + 0.15, base_level + 0.3]
                ax_main.contour(X, Y, density_display, levels=contour_levels, 
                               colors='white', alpha=0.3, linewidths=0.5)
                
                # "Proto-galaxies" at density peaks
                if perturbation_progress > 0.3:
                    peak_mask = density_display > 0.75
                    if np.any(peak_mask):
                        py, px = np.where(peak_mask)
                        px_world = -extent_size + px * (2 * extent_size) / nx
                        py_world = -extent_size + py * (2 * extent_size) / ny
                        n_show = min(30, len(px_world))
                        idx = np.random.choice(len(px_world), n_show, replace=False) if len(px_world) > n_show else range(len(px_world))
                        sizes = 5 + density_display[peak_mask][list(idx)] * 15
                        ax_main.scatter(px_world[list(idx)], py_world[list(idx)], s=sizes, c='white', alpha=0.6)
                
                # Power spectrum inset
                ax_inset = ax_main.inset_axes([0.02, 0.02, 0.25, 0.2])
                ax_inset.semilogy(k_pos[1:25], P_k_norm[1:25], color='cyan', linewidth=1)
                ax_inset.axvline(k_pos[5], color='yellow', alpha=0.5, linestyle='--')
                ax_inset.set_facecolor('black')
                ax_inset.set_title('P(k)', color='cyan', fontsize=7)
                ax_inset.tick_params(colors='gray', labelsize=5)
                ax_inset.set_xlim(0, k_pos[25])
                
                # === EXPLANATORY TEXT ===
                explain_text = (
                    "FFT of soliton -> primordial\n"
                    "power spectrum P(k)\n"
                    f"Modes grow: deltarho/rho ~ a(t)\n"
                    f"Growth: {growth:.1f}x"
                )
                ax_main.text(0.98, 0.98, explain_text, transform=ax_main.transAxes,
                            fontsize=8, color='white', va='top', ha='right',
                            fontfamily='monospace', alpha=0.8,
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
                
                ax_main.set_xlim(-extent_size, extent_size)
                ax_main.set_ylim(-extent_size, extent_size)
                ax_main.set_title(f'DENSITY PERTURBATIONS  T={T_current:.0f}K  Growth={growth:.1f}x', 
                                 color='orange', fontsize=13, fontweight='bold')
            
            # === STAGE 4: STRUCTURE + REDSHIFT (frames 130-220) ===
            elif frames_since < 220:
                # Universe has formed structure, now cooling and redshifting
                # ANIMATED: space expands, structures drift apart
                
                cooling_progress = (frames_since - 130) / 90.0
                time_phase = (frames_since - 130) * 0.1  # Slower animation for majesty
                
                # Power spectrum (same as before)
                psi_fft = np.fft.fft(psi)
                k_modes = np.fft.fftfreq(N, d=(x[1]-x[0]))
                P_k = np.abs(psi_fft[:N//2]) ** 2
                P_k_norm = P_k / (np.max(P_k) + 1e-10)
                k_pos = k_modes[:N//2]
                
                # Growth factor (saturates as dark energy dominates)
                growth = a_t * 3
                
                # Temperature evolution: T ~ 1/a(t)
                T_initial = 15000
                T_current = max(T_initial / (a_t + 0.1), 2.7)  # Floor at CMB temp
                
                # Create density field with evolved perturbations
                # ANIMATE: field expands with scale factor
                nx, ny = 120, 120
                current_extent = 8 * (1 + cooling_progress * 0.5)  # Expanding
                xx = np.linspace(-current_extent, current_extent, nx)
                yy = np.linspace(-current_extent, current_extent, ny)
                X, Y = np.meshgrid(xx, yy)
                
                density = np.ones_like(X)
                n_modes = min(25, N//4)
                
                for i in range(1, n_modes):
                    if P_k_norm[i] < 0.03:
                        continue
                    
                    k = abs(k_pos[i]) + 0.1
                    # Non-linear growth: overdensities collapse
                    amplitude = min(P_k_norm[i] * growth * 0.5, 1.5)
                    phase_k = np.angle(psi_fft[i])
                    
                    np.random.seed(seed + i)
                    angle = np.random.random() * 2 * np.pi
                    
                    # Slow drift animation - structures moving apart (Hubble flow)
                    drift = time_phase * 0.02 * (i % 5 - 2)  # Different drift per mode
                    
                    density += amplitude * np.cos(k * (X * np.cos(angle + drift) + Y * np.sin(angle + drift)) / (a_t + 0.5) + phase_k)
                
                # Non-linear collapse: high densities get higher (gravitational instability)
                # Clip to positive values before power to avoid RuntimeWarning
                density = np.where(density > 1.15, np.abs(density) ** 1.4, density)
                density = (density - density.min()) / (density.max() - density.min() + 1e-10)
                
                # === REDSHIFT COLORMAP ===
                # Create colormap that transitions from hot to cold
                bb_color = blackbody_color(T_current)
                
                # Custom colormap: black -> blackbody color -> white (for dense regions)
                from matplotlib.colors import LinearSegmentedColormap
                colors_list = [
                    (0, 0, 0.03),      # Void: very dark
                    (bb_color[0]*0.3, bb_color[1]*0.3, bb_color[2]*0.5),  # Dim
                    bb_color,           # Medium: blackbody temp
                    (1, 1, 0.9),       # Dense: white/yellow (stars)
                ]
                redshift_cmap = LinearSegmentedColormap.from_list('redshift', colors_list)
                
                # Plot the evolved density field
                extent = [-current_extent, current_extent, -current_extent, current_extent]
                ax_main.imshow(density, extent=extent, origin='lower',
                              cmap=redshift_cmap, alpha=0.9)
                
                # "Galaxies" at density peaks - animated twinkling
                peak_threshold = 0.75
                galaxy_mask = density > peak_threshold
                if np.any(galaxy_mask):
                    gy, gx = np.where(galaxy_mask)
                    gx_world = xx[0] + gx * (xx[-1] - xx[0]) / nx
                    gy_world = yy[0] + gy * (yy[-1] - yy[0]) / ny
                    # Sample a subset
                    n_galaxies = min(60, len(gx_world))
                    np.random.seed(seed + frames_since)  # Different each frame for twinkle
                    idx = np.random.choice(len(gx_world), n_galaxies, replace=False) if len(gx_world) > n_galaxies else range(len(gx_world))
                    # Twinkle effect
                    twinkle = 0.7 + 0.3 * np.sin(time_phase * 3 + np.arange(len(list(idx))) * 0.5)
                    sizes = (8 + density[galaxy_mask][list(idx)] * 25) * twinkle
                    ax_main.scatter(gx_world[list(idx)], gy_world[list(idx)], 
                                   s=sizes, c='white', alpha=0.8, marker='.')
                
                # Show parameters that emerged
                if state.collapsed_params:
                    p = state.collapsed_params
                    info_text = f"H:{p['n_herbies']} A:{p['n_apex']} W:{p['world_size']:.0f}"
                    ax_main.text(0.02, 0.98, info_text, transform=ax_main.transAxes,
                                fontsize=10, color='white', va='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                # === EXPLANATORY TEXT ===
                explain_text = (
                    "STRUCTURE FORMATION:\n"
                    f"Scale factor a(t) = {a_t:.2f}\n"
                    f"Temp: {T_current:.0f}K -> redshift\n"
                    "Overdensities collapse\n"
                    "into proto-galaxies (delta^1.5)\n"
                    "White dots = density peaks"
                )
                ax_main.text(0.98, 0.02, explain_text, transform=ax_main.transAxes,
                            fontsize=8, color='white', va='bottom', ha='right',
                            fontfamily='monospace', alpha=0.9,
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                view_size = current_extent
                ax_main.set_xlim(-view_size, view_size)
                ax_main.set_ylim(-view_size, view_size)
                
                # Redshift z = a_0/a - 1 (if a_0 = 1 at present)
                z = max(0, 1/a_t - 1) if a_t < 1 else 0
                ax_main.set_title(f'STRUCTURE FORMATION  T={T_current:.0f}K  z={z:.1f}', 
                                 color=bb_color, fontsize=14, fontweight='bold')
            
            # === STAGE 5: COSMIC TIMELAPSE (frames 220-300) ===
            elif frames_since < 300:
                # Fast-forward through cosmic history: Big Bang -> Modern day
                # Epochs: Inflation -> Nucleosynthesis -> Recombination -> 
                #         Dark Ages -> First Stars -> Galaxy Formation -> Modern
                
                timelapse_progress = (frames_since - 220) / 80.0
                
                # Cosmic time labels (logarithmic scale)
                epochs = [
                    (0.00, "INFLATION", "10^-36 s", (1.0, 1.0, 0.5)),
                    (0.08, "NUCLEOSYNTHESIS", "3 min", (1.0, 0.8, 0.3)),
                    (0.20, "RECOMBINATION", "380,000 yr", (1.0, 0.6, 0.4)),
                    (0.35, "DARK AGES", "10 Myr", (0.3, 0.2, 0.4)),
                    (0.50, "FIRST STARS", "100 Myr", (0.6, 0.7, 1.0)),
                    (0.65, "GALAXY FORMATION", "1 Byr", (0.8, 0.6, 1.0)),
                    (0.80, "SOLAR SYSTEM", "9 Byr", (1.0, 0.9, 0.3)),
                    (0.95, "HERBIE WORLD", "NOW", (0.3, 1.0, 0.5)),
                ]
                
                # Background: zooming through space
                ax_main.set_facecolor('black')
                
                # Star field that gets denser over time
                np.random.seed(seed)
                n_stars = int(50 + timelapse_progress * 250)
                star_x = np.random.uniform(-10, 10, n_stars)
                star_y = np.random.uniform(-10, 10, n_stars)
                star_sizes = np.random.exponential(2, n_stars) * (1 + timelapse_progress)
                star_alpha = 0.3 + timelapse_progress * 0.5
                
                # Stars colored by epoch - physically accurate stellar populations
                if timelapse_progress < 0.35:
                    # Dark ages / early: no stars yet, just faint background
                    star_colors = [(0.3, 0.3, 0.4, star_alpha * 0.3)] * n_stars
                elif timelapse_progress < 0.5:
                    # First stars (Pop III): extremely hot, blue-white, massive
                    star_colors = [(0.7, 0.8, 1.0, star_alpha)] * n_stars
                elif timelapse_progress < 0.7:
                    # Galaxy formation: mix of blue (young) and some older
                    star_colors = [(0.8, 0.85, 0.95, star_alpha)] * n_stars
                else:
                    # Modern: diverse stellar populations (G/K/M type more common)
                    star_colors = [(1.0, 0.95, 0.85, star_alpha)] * n_stars
                
                ax_main.scatter(star_x, star_y, s=star_sizes, c=star_colors, marker='.')
                
                # === DENSITY COLLAPSE -> GALAXIES ===
                # Galaxies form from gravitational collapse of overdense regions
                # NOT by spirals appearing - they emerge from density perturbations
                if timelapse_progress > 0.5:
                    collapse_progress = (timelapse_progress - 0.5) * 2  # 0 to 1
                    
                    # Use soliton FFT modes to seed galaxy positions (information -> matter)
                    np.random.seed(seed + 42)
                    n_proto = int(3 + collapse_progress * 8)  # More galaxies form over time
                    
                    for i in range(n_proto):
                        # Galaxy positions seeded by modes
                        gal_x = np.random.uniform(-7, 7)
                        gal_y = np.random.uniform(-7, 7)
                        
                        # Galaxy size grows as it collapses and accretes
                        gal_size = 0.5 + collapse_progress * 1.5
                        
                        # Draw as collapsed density region, not spiral line
                        # Inner bright core (bulge)
                        core = plt.Circle((gal_x, gal_y), gal_size * 0.3,
                                         color=(1.0, 0.95, 0.8), alpha=0.6 * collapse_progress)
                        ax_main.add_patch(core)
                        
                        # Disk/halo (if late enough for disk to form)
                        if collapse_progress > 0.4:
                            # Spiral structure emerges from disk rotation
                            disk_alpha = (collapse_progress - 0.4) * 1.5
                            n_arm_points = 50
                            for arm in range(2):  # Two spiral arms
                                theta_arm = np.linspace(0, 3*np.pi, n_arm_points) + arm * np.pi
                                r_arm = 0.2 + theta_arm * 0.12 * gal_size
                                # Add rotation based on collapse progress
                                rotation = collapse_progress * 2 + i * 0.5
                                arm_x = gal_x + r_arm * np.cos(theta_arm + rotation) * 0.7
                                arm_y = gal_y + r_arm * np.sin(theta_arm + rotation)
                                # Fade toward edges
                                for j in range(len(arm_x) - 1):
                                    seg_alpha = disk_alpha * (1 - j/n_arm_points) * 0.5
                                    ax_main.plot(arm_x[j:j+2], arm_y[j:j+2], 
                                               color=(0.8, 0.85, 1.0), 
                                               linewidth=1.5 * (1 - j/n_arm_points),
                                               alpha=seg_alpha)
                
                # Current epoch indicator
                current_epoch = None
                for i, (threshold, name, time_str, color) in enumerate(epochs):
                    if timelapse_progress >= threshold:
                        current_epoch = (name, time_str, color)
                
                if current_epoch:
                    name, time_str, color = current_epoch
                    # Pulsing effect
                    pulse = 0.8 + 0.2 * np.sin(timelapse_progress * 20)
                    ax_main.text(0, 0, name, fontsize=20, color=color, 
                                ha='center', va='center', fontweight='bold',
                                alpha=pulse)
                    ax_main.text(0, -1.5, time_str, fontsize=12, color='white',
                                ha='center', va='center', alpha=0.8)
                
                # Timeline bar at bottom
                ax_main.axhline(-8, color='gray', linewidth=2, alpha=0.3)
                ax_main.plot([-9 + timelapse_progress * 18], [-8], 'o', 
                            color='cyan', markersize=10)
                ax_main.text(-9, -9, "Big Bang", color='gray', fontsize=8, ha='left')
                ax_main.text(9, -9, "Now", color='gray', fontsize=8, ha='right')
                
                ax_main.set_xlim(-10, 10)
                ax_main.set_ylim(-10, 10)
                ax_main.set_title('COSMIC TIMELAPSE', color='white', fontsize=14, fontweight='bold')
            
            # === STAGE 6: HERBIE WORLD TITLE CARD (frames 300-360) ===
            elif frames_since < 360:
                # Video game style title card
                title_progress = (frames_since - 300) / 60.0
                
                ax_main.set_facecolor('#0a0a1a')
                
                # Starfield background
                np.random.seed(seed + 1)
                n_bg_stars = 150
                bg_x = np.random.uniform(-10, 10, n_bg_stars)
                bg_y = np.random.uniform(-10, 10, n_bg_stars)
                bg_sizes = np.random.exponential(1.5, n_bg_stars)
                ax_main.scatter(bg_x, bg_y, s=bg_sizes, c='white', alpha=0.4, marker='.')
                
                # Title fade in
                title_alpha = min(1.0, title_progress * 2.5)
                
                # Main title with glow effect
                if title_progress > 0.1:
                    # Glow
                    for offset in [3, 2, 1]:
                        glow_alpha = title_alpha * 0.15 / offset
                        ax_main.text(0, 2, "HERBIE WORLD", fontsize=28 + offset,
                                    color=(0.3, 0.8, 0.5), ha='center', va='center',
                                    fontweight='bold', alpha=glow_alpha)
                    # Main text
                    ax_main.text(0, 2, "HERBIE WORLD", fontsize=28,
                                color=(0.4, 1.0, 0.6), ha='center', va='center',
                                fontweight='bold', alpha=title_alpha)
                
                # Subtitle
                if title_progress > 0.25:
                    sub_alpha = min(1.0, (title_progress - 0.25) * 4)
                    ax_main.text(0, 0, "Emergent Artificial Life", fontsize=14,
                                color='white', ha='center', va='center',
                                style='italic', alpha=sub_alpha * 0.9)
                
                # Version and seed info
                if title_progress > 0.4:
                    info_alpha = min(1.0, (title_progress - 0.4) * 3)
                    ax_main.text(0, -2, f"Universe Seed: {seed}", fontsize=10,
                                color='cyan', ha='center', va='center',
                                alpha=info_alpha * 0.8)
                    ax_main.text(0, -3, f"Audio Energy: {state.audio_loudness:.2f}", fontsize=9,
                                color='orange', ha='center', va='center',
                                alpha=info_alpha * 0.7)
                
                # "Initializing..." prompt
                if title_progress > 0.6:
                    blink = 0.5 + 0.5 * np.sin(title_progress * 15)
                    ax_main.text(0, -5.5, "Initializing wormhole...", fontsize=11,
                                color='cyan', ha='center', va='center',
                                alpha=blink * 0.8)
                
                # Decorative border
                border_alpha = min(0.6, title_progress)
                rect = plt.Rectangle((-9, -8), 18, 16, fill=False, 
                                    edgecolor=(0.3, 0.8, 0.5), linewidth=2,
                                    alpha=border_alpha)
                ax_main.add_patch(rect)
                
                # Corner decorations
                for cx, cy in [(-8.5, 7.5), (8.5, 7.5), (-8.5, -7.5), (8.5, -7.5)]:
                    ax_main.plot([cx], [cy], '*', color=(0.4, 1.0, 0.6), 
                                markersize=12, alpha=border_alpha)
                
                ax_main.set_xlim(-10, 10)
                ax_main.set_ylim(-10, 10)
                ax_main.set_title('', color='white')
            
            # === STAGE 7: WORMHOLE - TIME BRIDGE (frames 360-460) ===
            # Fast-forward from primordial soliton through cosmic time to HERBIE
            # The user's audio shaped the soliton -> which determines ALL future conditions
            # This visualizes: Past(soliton + audio) ===> Future(HERBIE initial state)
            else:
                wormhole_progress = (frames_since - 360) / 100.0
                
                ax_main.set_facecolor('#050510')
                
                # Calculate elapsed cosmic time for display
                if wormhole_progress < 0.4:
                    cosmic_years = 10 ** (wormhole_progress * 25)  # up to 10^10 years
                else:
                    cosmic_years = 13.8e9  # 13.8 billion years
                
                # === PHASE 1: ENTERING THE TIME BRIDGE (0-40%) ===
                if wormhole_progress < 0.4:
                    throat_progress = wormhole_progress / 0.4
                    
                    # === TIME TUNNEL VISUALIZATION ===
                    # Concentric rings flowing toward us = traveling through time
                    n_rings = 35
                    for i in range(n_rings):
                        ring_depth = i / n_rings
                        
                        # Tunnel expands as we travel forward in time
                        # Hyperbolic profile: narrow past -> wide future
                        z = (ring_depth - 0.3) * 5
                        ring_r = 2.0 / (np.cosh(z) + 0.3) + ring_depth * 3
                        
                        # Animate: rings flow toward viewer (time passing)
                        flow = (throat_progress * 4 + i * 0.08) % 1.0
                        ring_r *= (0.4 + flow * 1.2)
                        
                        # Color gradient: PAST (blue/cyan) -> FUTURE (green)
                        # Blue = information geometry, Green = physical matter
                        past_color = np.array([0.2, 0.5, 0.9])  # Cyan-blue
                        future_color = np.array([0.3, 0.9, 0.5])  # Green
                        ring_color = past_color * (1 - ring_depth) + future_color * ring_depth
                        
                        ring_alpha = 0.5 * (1 - abs(ring_depth - 0.5) * 1.2) * (0.5 + throat_progress)
                        theta = np.linspace(0, 2*np.pi, 60)
                        ax_main.plot(ring_r * np.cos(theta), ring_r * np.sin(theta),
                                    color=ring_color, alpha=ring_alpha, linewidth=1.5)
                    
                    # === SOLITON MODES TRAVELING THROUGH TIME ===
                    # The FFT modes from the original soliton propagate forward
                    psi_fft = np.fft.fft(psi)
                    P_k = np.abs(psi_fft[:N//4]) ** 2
                    P_k_norm = P_k / (np.max(P_k) + 1e-10)
                    
                    # Modes spiral inward = information encoding future
                    for mode_i in range(1, min(15, len(P_k_norm))):
                        if P_k_norm[mode_i] > 0.03:
                            # Spiral toward center as we travel through time
                            mode_angle = mode_i * np.pi / 6 + throat_progress * 6
                            mode_r = 7 * (1 - throat_progress * 0.8) * (1 - mode_i * 0.03)
                            mx = mode_r * np.cos(mode_angle)
                            my = mode_r * np.sin(mode_angle)
                            
                            # Size = spectral power = how much this mode matters
                            mode_size = 20 + P_k_norm[mode_i] * 120
                            # Color fades from cyan to green
                            mode_color = (0.3 + throat_progress * 0.1, 
                                         0.5 + throat_progress * 0.4, 
                                         0.9 - throat_progress * 0.4,
                                         0.7)
                            ax_main.scatter([mx], [my], s=mode_size, c=[mode_color], marker='o')
                    
                    # Time markers
                    ax_main.text(0, 8.5, f"PAST: Primordial Soliton", fontsize=10,
                                color='cyan', ha='center', fontweight='bold')
                    ax_main.text(0, 7.5, f"(shaped by YOUR audio: {state.audio_loudness:.2f})", 
                                fontsize=9, color='cyan', ha='center', alpha=0.7)
                    
                    future_alpha = throat_progress
                    ax_main.text(0, -7.5, f"FUTURE: HERBIE World", fontsize=10,
                                color=(0.4, 1.0, 0.6), ha='center', fontweight='bold',
                                alpha=future_alpha)
                    ax_main.text(0, -8.5, f"~{cosmic_years:.1e} years ahead", fontsize=9,
                                color='white', ha='center', alpha=future_alpha * 0.7)
                    
                    # Central time indicator
                    ax_main.scatter([0], [0], s=100 + throat_progress * 200, 
                                   c='white', alpha=0.6 * throat_progress)
                    
                    ax_main.set_title('TIME BRIDGE: Soliton -> HERBIE World', 
                                     color='cyan', fontsize=13, fontweight='bold')
                
                # === PHASE 2: EMERGENCE (40-70%) ===
                elif wormhole_progress < 0.7:
                    emerge_progress = (wormhole_progress - 0.4) / 0.3
                    
                    # Wormhole exit: information crystallizes into creatures
                    # Central emergence point
                    ax_main.scatter([0], [0], s=200 * (1 - emerge_progress), 
                                   c='white', alpha=0.8 * (1 - emerge_progress))
                    
                    # Creature positions emerge from soliton modes
                    # Use power spectrum to seed positions (information -> configuration)
                    psi_fft = np.fft.fft(psi)
                    n_creatures = state.collapsed_params.get('n_herbies', 8) if state.collapsed_params else 8
                    n_apex = state.collapsed_params.get('n_apex', 2) if state.collapsed_params else 2
                    
                    np.random.seed(seed)
                    
                    # Herbies emerge (green dots)
                    for i in range(n_creatures):
                        # Position seeded by FFT mode
                        mode_phase = np.angle(psi_fft[i + 1]) if i + 1 < len(psi_fft) else 0
                        mode_amp = np.abs(psi_fft[i + 1]) if i + 1 < len(psi_fft) else 0.5
                        
                        # Emerge from center outward
                        final_r = 3 + mode_amp * 4
                        final_angle = mode_phase + i * 0.7
                        current_r = final_r * emerge_progress
                        
                        hx = current_r * np.cos(final_angle)
                        hy = current_r * np.sin(final_angle)
                        
                        # Materialize effect
                        mat_alpha = min(1.0, emerge_progress * 2)
                        ax_main.scatter([hx], [hy], s=80, c=[(0.3, 0.9, 0.4, mat_alpha)],
                                       marker='o', edgecolors='white', linewidths=0.5)
                        
                        # Trail showing emergence path
                        if emerge_progress > 0.2:
                            trail_r = np.linspace(0, current_r, 10)
                            trail_x = trail_r * np.cos(final_angle)
                            trail_y = trail_r * np.sin(final_angle)
                            ax_main.plot(trail_x, trail_y, color=(0.3, 0.9, 0.4),
                                        alpha=0.2, linewidth=1)
                    
                    # Apex predators emerge (red dots)
                    for i in range(n_apex):
                        apex_angle = np.pi + i * np.pi / n_apex
                        apex_r = 5 * emerge_progress
                        ax_main.scatter([apex_r * np.cos(apex_angle)], 
                                       [apex_r * np.sin(apex_angle)],
                                       s=120, c=[(0.9, 0.2, 0.2, emerge_progress)],
                                       marker='^', edgecolors='white', linewidths=0.5)
                    
                    # World boundary forming
                    world_size = state.collapsed_params.get('world_size', 100) if state.collapsed_params else 100
                    boundary_r = 8 * emerge_progress
                    boundary_theta = np.linspace(0, 2*np.pi, 100)
                    ax_main.plot(boundary_r * np.cos(boundary_theta),
                                boundary_r * np.sin(boundary_theta),
                                color=(0.4, 1.0, 0.6), alpha=0.5 * emerge_progress,
                                linewidth=2, linestyle='--')
                    
                    ax_main.set_title(f'ARRIVING: 13.8 Gyr later | {n_creatures} Herbies, {n_apex} Apex',
                                     color=(0.4, 1.0, 0.6), fontsize=12, fontweight='bold')
                
                # === PHASE 3: WORLD INITIALIZATION (70-100%) ===
                else:
                    init_progress = (wormhole_progress - 0.7) / 0.3
                    
                    # Final world setup visualization
                    ax_main.set_facecolor('#0a1510')
                    
                    # Terrain grid forming
                    grid_alpha = init_progress * 0.3
                    for g in np.linspace(-8, 8, 17):
                        ax_main.axhline(g, color=(0.3, 0.5, 0.4), alpha=grid_alpha, linewidth=0.5)
                        ax_main.axvline(g, color=(0.3, 0.5, 0.4), alpha=grid_alpha, linewidth=0.5)
                    
                    # Creatures in final positions
                    n_creatures = state.collapsed_params.get('n_herbies', 8) if state.collapsed_params else 8
                    n_apex = state.collapsed_params.get('n_apex', 2) if state.collapsed_params else 2
                    n_blobs = state.collapsed_params.get('n_blobs', 4) if state.collapsed_params else 4
                    
                    psi_fft = np.fft.fft(psi)
                    np.random.seed(seed)
                    
                    # Draw all creatures
                    for i in range(n_creatures):
                        mode_phase = np.angle(psi_fft[i + 1]) if i + 1 < len(psi_fft) else i * 0.5
                        mode_amp = np.abs(psi_fft[i + 1]) / (np.max(np.abs(psi_fft)) + 1e-10) if i + 1 < len(psi_fft) else 0.5
                        r = 3 + mode_amp * 4
                        angle = mode_phase + i * 0.7
                        ax_main.scatter([r * np.cos(angle)], [r * np.sin(angle)],
                                       s=100, c=[(0.3, 0.9, 0.4, 1.0)], marker='o',
                                       edgecolors='white', linewidths=1)
                    
                    for i in range(n_apex):
                        apex_angle = np.pi + i * np.pi / max(1, n_apex)
                        ax_main.scatter([5 * np.cos(apex_angle)], [5 * np.sin(apex_angle)],
                                       s=150, c=[(0.9, 0.2, 0.2, 1.0)], marker='^',
                                       edgecolors='white', linewidths=1)
                    
                    for i in range(n_blobs):
                        blob_angle = i * 2 * np.pi / max(1, n_blobs) + 0.5
                        blob_r = 4
                        ax_main.scatter([blob_r * np.cos(blob_angle)], [blob_r * np.sin(blob_angle)],
                                       s=70, c=[(0.6, 0.4, 0.8, 1.0)], marker='s',
                                       edgecolors='white', linewidths=0.5)
                    
                    # World info
                    info_text = (
                        f"World Size: {state.collapsed_params.get('world_size', 100):.0f}\n"
                        f"Herbies: {n_creatures}\n"
                        f"Apex: {n_apex}\n"
                        f"Blobs: {n_blobs}\n"
                        f"Seed: {seed}"
                    ) if state.collapsed_params else "Initializing..."
                    
                    ax_main.text(0.02, 0.98, info_text, transform=ax_main.transAxes,
                                fontsize=10, color='white', va='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    
                    # Show the causal chain
                    ax_main.text(0.98, 0.98, 
                                "CAUSAL CHAIN:\n"
                                "Audio -> Soliton\n"
                                "  -> FFT modes\n"
                                "  -> Structure\n"
                                "  -> Creatures",
                                transform=ax_main.transAxes, fontsize=8, color='cyan',
                                va='top', ha='right', fontfamily='monospace', alpha=0.7)
                    
                    # "Launching" countdown
                    countdown = max(0, int((1 - init_progress) * 3) + 1)
                    if init_progress < 0.9:
                        ax_main.text(0, -7, f"Launching in {countdown}...", fontsize=14,
                                    color='white', ha='center', fontweight='bold',
                                    alpha=0.5 + 0.5 * np.sin(init_progress * 20))
                    else:
                        ax_main.text(0, -7, "LAUNCH!", fontsize=18,
                                    color=(0.4, 1.0, 0.6), ha='center', fontweight='bold')
                    
                    ax_main.set_title('PRESENT DAY: Conditions set by primordial soliton', 
                                     color=(0.4, 1.0, 0.6), fontsize=12, fontweight='bold')
                
                ax_main.set_xlim(-10, 10)
                ax_main.set_ylim(-10, 10)
            
            ax_main.set_aspect('equal')
            
        else:
            # === PRE-COSMOLOGICAL VISUALIZATION ===
            # Swimming in the high-dimensional quantum foam
            
            # Color shifts with audio: deep blue -> cyan -> orange -> white
            if current_audio < 0.3:
                # Quiet: deep blue/purple
                t = current_audio / 0.3
                color = np.array([0.1 + t * 0.1, 0.1 + t * 0.5, 0.8 + t * 0.2])
            elif current_audio < 0.6:
                # Medium: cyan
                t = (current_audio - 0.3) / 0.3
                color = np.array([0.2 + t * 0.6, 0.6 + t * 0.2, 1.0 - t * 0.3])
            else:
                # Loud: orange/white
                t = (current_audio - 0.6) / 0.4
                color = np.array([0.8 + t * 0.2, 0.8 - t * 0.3, 0.7 - t * 0.5])
            
            # === 3D NLSE QUANTUM FOAM ===
            # Get the foam and evolve it
            foam = get_quantum_foam()
            
            # Evolve the 3D NLSE with audio coupling
            foam.step(audio_amp=current_audio, audio_phase=state.tau * 2)
            
            # Inject soliton pulses on loud audio bursts
            if current_audio > 0.5 and np.random.random() < 0.1:
                direction = np.array([
                    np.cos(state.tau * 3),
                    np.sin(state.tau * 3),
                    np.sin(state.tau * 2) * 0.5
                ])
                foam.inject_soliton(direction, amplitude=current_audio * 0.8)
            
            # Get render points
            positions, point_colors, sizes = foam.get_render_points(
                threshold=0.03 + current_audio * 0.02,
                max_points=1500 + int(current_audio * 500)
            )
            
            if len(positions) > 0:
                # Rotate view over time for 3D effect
                rotation_angle = state.tau * 0.3
                tilt_angle = 0.3 + 0.2 * np.sin(state.tau * 0.5)
                
                # Rotation matrix for view
                cos_r, sin_r = np.cos(rotation_angle), np.sin(rotation_angle)
                cos_t, sin_t = np.cos(tilt_angle), np.sin(tilt_angle)
                
                # Apply rotation around Y axis (horizontal spin)
                x_rot = positions[:, 0] * cos_r - positions[:, 2] * sin_r
                z_rot = positions[:, 0] * sin_r + positions[:, 2] * cos_r
                
                # Apply tilt around X axis (vertical tilt)
                y_proj = positions[:, 1] * cos_t - z_rot * sin_t
                
                # Project to 2D (x_rot, y_proj), use z for depth
                depth = z_rot * cos_t + positions[:, 1] * sin_t
                
                # Depth affects size and alpha
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
                sizes_adjusted = sizes * (0.5 + depth_norm * 1.0)
                
                # Render as scatter
                ax_main.scatter(x_rot, y_proj, s=sizes_adjusted, c=point_colors, 
                               alpha=0.7, edgecolors='none')
                
                # Add glow effect at center
                if current_audio > 0.2:
                    center_mask = (np.abs(x_rot) < 2) & (np.abs(y_proj) < 2)
                    if np.any(center_mask):
                        ax_main.scatter(x_rot[center_mask], y_proj[center_mask],
                                       s=sizes_adjusted[center_mask] * 2,
                                       c='white', alpha=current_audio * 0.3)
            
            # Background stars (distant structure)
            n_bg = 50 + int(current_audio * 100)
            np.random.seed(seed)
            bg_x = np.random.uniform(-10, 10, n_bg)
            bg_y = np.random.uniform(-8, 8, n_bg)
            bg_sizes = np.random.uniform(0.5, 3, n_bg)
            ax_main.scatter(bg_x, bg_y, s=bg_sizes, c='white', alpha=0.2 + current_audio * 0.1)
            
            ax_main.set_xlim(-10, 10)
            ax_main.set_ylim(-8, 8)
            ax_main.set_xlabel('', color='gray')
            ax_main.set_ylabel('', color='gray')
            ax_main.set_xticks([])
            ax_main.set_yticks([])
            
            # Title pulses with audio
            title_intensity = 0.6 + current_audio * 0.4
            shell_r = foam.get_shell_radius()
            ax_main.set_title(f'~ 3D NLSE Quantum Foam ~  r={shell_r:.1f}  [Seed: {seed}]', 
                             color=(title_intensity, title_intensity * 0.9, title_intensity), 
                             fontsize=12 + int(current_audio * 4), fontweight='bold')
        
        # Phase display
        ax_phase.clear()
        ax_phase.set_facecolor('black')
        phase = np.angle(psi)
        colors = (phase + np.pi) / (2 * np.pi)
        jitter_y = np.random.randn(len(x)) * current_audio * 0.2 if state.phase == "evolving" else np.zeros(len(x))
        ax_phase.scatter(x, jitter_y, c=colors, cmap='hsv', s=20 + current_audio * 30, alpha=0.8)
        ax_phase.set_xlim(-10, 10)
        ax_phase.set_ylim(-0.5, 0.5)
        ax_phase.set_yticks([])
        ax_phase.set_title('Phase (sound-modulated)', color='magenta', fontsize=10)
        
        # Audio display
        ax_audio.clear()
        ax_audio.set_facecolor('black')
        
        # Waveform history - copy to avoid race condition
        history_copy = list(audio_monitor.history)
        if len(history_copy) > 1:
            hist_x = np.linspace(0, 1, len(history_copy))
            ax_audio.fill_between(hist_x, 0, history_copy, color='cyan', alpha=0.3)
            ax_audio.plot(hist_x, history_copy, color='cyan', linewidth=1)
        
        # Current level bar
        bar_color = 'orange' if state.phase != "evolving" else ('lime' if current_audio > 0.2 else 'cyan')
        ax_audio.barh([0.8], [current_audio], color=bar_color, height=0.15)
        ax_audio.set_xlim(0, 1)
        ax_audio.set_ylim(0, 1)
        ax_audio.set_title(f'Audio: {current_audio:.3f}', color=bar_color, fontsize=10)
        ax_audio.set_yticks([])
        
        # Info panel - PHYSICS READOUT
        ax_info.clear()
        ax_info.set_facecolor('black')
        ax_info.axis('off')
        
        if state.phase in ("collapsing", "collapsed"):
            frames_since = frame - collapse_frame[0]
            
            # === STAGE-SPECIFIC PHYSICS READOUT ===
            if frames_since < 30:
                # VORTEX stage
                info = """CE THEORY: IMAGINARY TIME
========================
Domain: Im(t) = tau
Dynamics: DIFFUSIVE

The wave equation in imaginary
time is the HEAT EQUATION:
  d psi/d tau = D * nabla^2 psi

Key insight: Diffusion runs
BACKWARDS - contracts where
you'd expect expansion.

Information geometry seeks
GROUND STATE - probability
amplitude converges toward
the t0 observation point.

Audio energy shapes the
soliton's power spectrum.
--------------------------
[Im] Entropy converging
[Re] Not yet physical"""
            
            elif frames_since < 50:
                # DELTA COLLAPSE stage
                info = f"""CE THEORY: t0 HYPERPLANE
========================
Domain: Im(t) = 0
Event: WAVEFUNCTION COLLAPSE

The Dirac delta spike:
  psi -> delta(x - x0)

This IS the "observation"
that creates the universe.
Probability distribution
crystallizes to definite
values.

Audio at collapse: {state.audio_loudness:.3f}
  -> Seeds ALL parameters

Universe params FIXED here:
  - World topology
  - Creature populations  
  - Resource density
--------------------------
[Im] Information selected
[Re] Parameters frozen"""
            
            elif frames_since < 90:
                # WICK ROTATION stage
                wick_prog = (frames_since - 50) / 40.0
                info = f"""CE THEORY: WICK ROTATION
========================
Transform: tau -> i*t
Progress: {wick_prog*100:.0f}%

INVERSE Wick rotation:
  e^(-H*tau) -> e^(-i*H*t)

Diffusive -> Oscillatory
Heat eq  -> Wave equation
Entropy  -> Energy

The collapsed delta rotates
from imaginary axis to real
axis in complex plane.

Metric expansion begins:
  a(t) ~ t^(2/3)
  Space ITSELF stretches
  (not motion through space)

T ~ 1/a(t) : cooling
--------------------------
[Im] Releasing to physics
[Re] Dynamics awakening"""
            
            elif frames_since < 130:
                # PERTURBATIONS stage
                info = f"""PRIMORDIAL PERTURBATIONS
========================
Era: Radiation-dominated
a(t) = {a_t:.3f}

Soliton FFT -> P(k) spectrum
Each mode k seeds structure:
  delta_rho/rho ~ 10^-5

Growth: delta ~ a(t)
Harrison-Zeldovich spectrum
(nearly scale-invariant)

The AUDIO at collapse shaped
the soliton, which shaped
the power spectrum, which
seeds ALL cosmic structure.

Your sound -> Galaxies
--------------------------
[Im] Information encoded
[Re] Structure seeding"""
            
            elif frames_since < 220:
                # STRUCTURE stage
                info = f"""STRUCTURE FORMATION
========================
Era: Matter-dominated
a(t) = {a_t:.3f}
T ~ {max(2.7, 15000/a_t):.0f} K

Non-linear collapse:
  Overdensities -> Halos
  Halos -> Galaxies
  
Jeans instability:
  Gravity wins over pressure
  above critical scale.

Dark energy begins to
dominate at late times:
  a(t) ~ e^(H*t)

Cosmic web emerges from
initial perturbations.
--------------------------
[Im] Pattern crystallized
[Re] Matter clustering"""
            
            elif frames_since < 300:
                # TIMELAPSE stage
                tl_prog = (frames_since - 220) / 80.0
                cosmic_time = 10 ** (tl_prog * 10) if tl_prog > 0 else 1  # years
                info = f"""COSMIC TIMELAPSE
========================
Progress: {tl_prog*100:.0f}%
~{cosmic_time:.0e} years

Epochs traversed:
{'[x]' if tl_prog > 0.00 else '[ ]'} Inflation (10^-36 s)
{'[x]' if tl_prog > 0.08 else '[ ]'} Nucleosynthesis (3 min)
{'[x]' if tl_prog > 0.20 else '[ ]'} Recombination (380 kyr)
{'[x]' if tl_prog > 0.35 else '[ ]'} Dark Ages (10 Myr)
{'[x]' if tl_prog > 0.50 else '[ ]'} First Stars (100 Myr)
{'[x]' if tl_prog > 0.65 else '[ ]'} Galaxy Formation (1 Gyr)
{'[x]' if tl_prog > 0.80 else '[ ]'} Solar System (9 Gyr)
{'[x]' if tl_prog > 0.95 else '[ ]'} HERBIE World (Now)

All from YOUR audio input.
--------------------------
[Soliton] -> [Universe]"""
            
            elif frames_since < 360:
                # TITLE CARD stage
                info = f"""HERBIE WORLD
========================
Seed: {seed}
Audio Energy: {state.audio_loudness:.3f}

The primordial soliton has
evolved through 13.8 billion
years of cosmic history.

Initial conditions for
artificial life simulation
are EMERGENT from:
  1. Soliton wavefunction
  2. Your audio at collapse
  3. FFT power spectrum
  4. Structure formation

Information -> Matter
Entropy -> Complexity
--------------------------
Preparing launch..."""
            
            else:
                # WORMHOLE stage
                wh_prog = (frames_since - 360) / 100.0
                if state.collapsed_params:
                    p = state.collapsed_params
                    info = f"""WORMHOLE: TIME BRIDGE
========================
Fast-forward: {wh_prog*100:.0f}%

Traversing from cosmic
structure to HERBIE World
initial conditions.

The wormhole connects:
  PAST: Primordial soliton
        shaped by your audio
  
  FUTURE: Simulation start
          with these params:

World: {p['world_size']:.0f}x{p['world_size']:.0f}
Herbies: {p['n_herbies']}
Apex: {p['n_apex']}
Resources: {p['nutrient_density']:.2f}

Causality preserved:
  Audio -> Soliton -> FFT
  -> Structure -> Creatures
--------------------------
[Past] ====> [Future]"""
                else:
                    info = """WORMHOLE: TIME BRIDGE
========================
Connecting primordial
soliton to simulation
initial conditions...

Your audio shaped the
beginning of this universe.
Now we fast-forward to
see what emerged.
--------------------------
[Past] ====> [Future]"""
        
        elif state.collapsed_params:
            p = state.collapsed_params
            info = f"""EMERGENT UNIVERSE
*******************
World Size: {p['world_size']:.1f}
Herbies: {p['n_herbies']}
Bipeds: {p['n_bipeds']}
Apex: {p['n_apex']}
Resources: {p['nutrient_density']:.2f}
Day Length: {p['day_length']}

Audio at collapse: {p['audio_loudness']:.3f}

Launching HERBIE World..."""
        else:
            # Audio bar visualization
            audio_bar = "#" * int(current_audio * 20) + "-" * (20 - int(current_audio * 20))
            
            info = f"""CE THEORY: PRE-PHYSICAL
========================
Domain: Im(t) = tau
tau = {state.tau:.2f}

You exist in IMAGINARY TIME
before the universe begins.

The soliton evolves via
diffusive dynamics (heat eq).
Your audio modulates its
amplitude and spectrum.

Audio: [{audio_bar}]
Level: {current_audio:.2f}

LOUDER = higher energy modes
       = more structure
       = richer universe

Press [SPACE] to collapse
the wavefunction at t0!
--------------------------
[Im] Pre-physical void
[Re] Awaiting collapse"""
        
        ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                    fontsize=9, color='white', fontfamily='monospace',
                    verticalalignment='top')
        
        # Status text color pulses with audio
        status_text.set_text(state.message)
        if state.phase == "evolving":
            r = 0.0 + current_audio * 0.8
            g = 0.8 - current_audio * 0.3
            b = 1.0 - current_audio * 0.5
            status_text.set_color((max(0,r), max(0,g), max(0,b)))
        elif state.phase == "collapsing":
            status_text.set_color('orange')
        else:
            status_text.set_color('yellow')
        
        return []
    
    # Run animation - wrap in robust exception handling
    # frames=800 gives plenty of time (collapse at ~100-200, timelapse ~460 more)
    anim = animation.FuncAnimation(fig, animate, frames=800, interval=50, blit=False, 
                                   cache_frame_data=False, repeat=False)
    anim_holder[0] = anim  # Store for cleanup from within animate()
    
    # Suppress macOS-specific timer errors
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    try:
        plt.show(block=True)
    except:
        pass  # Ignore ALL matplotlib errors
    
    # Cleanup matplotlib
    try:
        plt.close('all')
    except:
        pass
    
    # Short delay
    import time as _time
    _time.sleep(0.3)
    
    # Launch if we completed
    if state.phase == "launching" and state.collapsed_params:
        print("\n" + "=" * 60)
        print("   LAUNCHING HERBIE WORLD")
        print("=" * 60)
        launch_herbie_world(state.collapsed_params, seed)


def launch_herbie_world(params: Dict, seed: int):
    """Launch HERBIE World with collapsed cosmological parameters."""
    print("\n[Cosmos -> HERBIE] Initializing universe with collapsed parameters...")
    
    # Set environment variables
    os.environ['HERBIE_START_POP'] = str(params['n_herbies'])
    os.environ['HERBIE_START_APEX'] = str(params['n_apex'])
    os.environ['HERBIE_WORLD_SIZE'] = str(int(params['world_size']))
    os.environ['HERBIE_DAY_LENGTH'] = str(params['day_length'])
    os.environ['HERBIE_FOOD_MULT'] = str(params['nutrient_density'])
    
    # Save params
    params_file = os.path.join(os.getcwd(), 'data', 'cosmological_params.json')
    os.makedirs(os.path.dirname(params_file), exist_ok=True)
    
    with open(params_file, 'w') as f:
        json_params = {}
        for k, v in params.items():
            if isinstance(v, (np.floating, np.integer)):
                json_params[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, dict):
                json_params[k] = {kk: float(vv) for kk, vv in v.items()}
            else:
                json_params[k] = v
        json_params['seed'] = seed
        json.dump(json_params, f, indent=2)
    
    print(f"[Cosmos -> HERBIE] Parameters saved to {params_file}")
    print(f"[Cosmos -> HERBIE] World size: {params['world_size']:.1f}")
    print(f"[Cosmos -> HERBIE] Populations: H={params['n_herbies']}, B={params['n_blobs']}, Bi={params['n_bipeds']}, A={params['n_apex']}")
    print(f"[Cosmos -> HERBIE] Audio loudness at collapse: {params['audio_loudness']:.3f}")
    print()
    
    try:
        from herbie_world.main import main_visual
        main_visual()
    except ImportError as e:
        print(f"[Cosmos] Could not import HERBIE World: {e}")
        import subprocess
        herbie_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run([sys.executable, '-m', 'herbie_world'], cwd=os.path.dirname(herbie_dir), env=os.environ)


# === ENTRY POINT ===
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_cosmological_launcher(seed)
