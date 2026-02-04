"""
Cosmological Seed System - V6_ALT

Implements CE Theory-inspired world initialization where:
1. Initial conditions are encoded in a soliton waveform (information space)
2. Soliton evolves in "imaginary time" (pre-cosmological )
3. First audio input triggers Wick rotation / collapse ( -> it)
4. Dirac delta-like transduction extracts world parameters from soliton structure

The "Big Bang" of HERBIE World occurs when the observer (microphone)
collapses the superposition of possible initial conditions into a
determined universe.

This replaces arbitrary random seeds with physically meaningful
wave structures - the same PDEs that govern creature cognition
also govern universe creation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable
from enum import Enum


# =============================================================================
# SOLITON TYPES
# =============================================================================

class SolitonType(Enum):
    """Types of soliton solutions that can encode initial conditions."""
    KDV = 'kdv'           # KdV soliton: amplitude determines velocity
    NLSE_BRIGHT = 'nlse_bright'  # Bright soliton: bound state
    NLSE_DARK = 'nlse_dark'      # Dark soliton: density notch
    BREATHER = 'breather'        # Oscillating bound state
    MULTI_SOLITON = 'multi'      # Superposition of solitons


@dataclass
class SolitonProfile:
    """
    A soliton waveform encoding potential initial conditions.
    
    In CE theory terms: this is the information-space representation
    before collapse onto the real hyperplane.
    """
    soliton_type: SolitonType = SolitonType.NLSE_BRIGHT
    
    # Soliton parameters (information encoded here)
    amplitude: float = 1.0        # Peak amplitude -> energy scale
    velocity: float = 0.0         # Propagation speed -> time scale  
    width: float = 1.0            # Spatial extent -> world granularity
    phase: float = 0.0            # Global phase -> terrain seed
    position: float = 0.0         # Center position -> spatial offset
    
    # For multi-soliton configurations
    n_solitons: int = 1
    soliton_params: list = field(default_factory=list)  # List of (amp, vel, width, phase)
    
    # Pre-collapse evolution parameters
    tau_duration: float = 10.0    # How long to evolve in imaginary time
    tau_steps: int = 100          # Resolution of pre-collapse evolution
    
    def __post_init__(self):
        if self.n_solitons > 1 and not self.soliton_params:
            # Generate multi-soliton parameters
            self.soliton_params = [
                (self.amplitude * (1.0 - 0.2 * i), 
                 self.velocity + 0.1 * i,
                 self.width * (1.0 + 0.1 * i),
                 self.phase + np.pi/4 * i)
                for i in range(self.n_solitons)
            ]


# =============================================================================
# SOLITON WAVEFORM GENERATION
# =============================================================================

def generate_kdv_soliton(x: np.ndarray, amplitude: float, velocity: float, 
                         position: float = 0.0) -> np.ndarray:
    """
    Generate KdV soliton: u(x) = A * sech^2(sqrt(A/12) * (x - x0))
    
    The KdV soliton is remarkable: amplitude directly determines velocity.
    This is a natural encoding where "how much" determines "how fast".
    """
    # KdV: velocity c = A/3 for soliton of amplitude A
    k = np.sqrt(amplitude / 12.0) if amplitude > 0 else 0.1
    return amplitude * np.cosh(k * (x - position)) ** (-2)


def generate_nlse_bright_soliton(x: np.ndarray, amplitude: float, width: float,
                                  velocity: float, phase: float,
                                  position: float = 0.0) -> np.ndarray:
    """
    Generate NLSE bright soliton: (x) = A * sech((x-x0)/w) * exp(i*v*x + i*)
    
    Complex-valued, carrying both amplitude AND phase information.
    This is richer than KdV - phase encodes additional degrees of freedom.
    """
    envelope = amplitude * np.cosh((x - position) / width) ** (-1)
    phase_factor = np.exp(1j * (velocity * x + phase))
    return envelope * phase_factor


def generate_nlse_dark_soliton(x: np.ndarray, amplitude: float, depth: float,
                                velocity: float, position: float = 0.0) -> np.ndarray:
    """
    Generate NLSE dark soliton: density notch on background.
    
    (x) = A * (i*v + sqrt(1-v^2) * tanh(sqrt(1-v^2) * (x-x0)))
    
    Dark solitons represent "absence" - could encode voids/barriers.
    """
    v_norm = np.clip(velocity, -0.99, 0.99)
    cos_theta = np.sqrt(1 - v_norm**2)
    sin_theta = v_norm
    
    psi = amplitude * (1j * sin_theta + cos_theta * np.tanh(cos_theta * depth * (x - position)))
    return psi


def generate_breather(x: np.ndarray, t: float, amplitude: float, 
                      omega: float, position: float = 0.0) -> np.ndarray:
    """
    Generate Akhmediev breather or Kuznetsov-Ma breather.
    
    Breathers are "breathing" bound states - oscillatory solitons.
    They naturally encode periodic/cyclical information.
    """
    # Simplified breather form
    envelope = amplitude * np.cosh(0.5 * (x - position)) ** (-1)
    oscillation = np.cos(omega * t) + 0.5j * np.sin(omega * t)
    return envelope * oscillation


def generate_multi_soliton(x: np.ndarray, soliton_params: list, 
                           soliton_type: SolitonType) -> np.ndarray:
    """
    Generate superposition of multiple solitons.
    
    Multi-soliton solutions encode multiple independent degrees of freedom.
    In CE theory: multiple information streams collapsing simultaneously.
    """
    psi = np.zeros_like(x, dtype=complex)
    
    for amp, vel, width, phase in soliton_params:
        if soliton_type == SolitonType.KDV:
            # KdV is real-valued
            psi += generate_kdv_soliton(x, amp, vel, phase).astype(complex)
        else:
            psi += generate_nlse_bright_soliton(x, amp, width, vel, phase)
    
    return psi


# =============================================================================
# PRE-COSMOLOGICAL EVOLUTION (Imaginary Time)
# =============================================================================

def evolve_in_imaginary_time(psi_0: np.ndarray, tau_duration: float, 
                              tau_steps: int, g: float = -1.0) -> np.ndarray:
    """
    Evolve soliton in imaginary time (Wick rotated).
    
    Real time: i/t = -^2/x^2 + g||^2  (NLSE)
    Imaginary time: / = ^2/x^2 - g||^2  (diffusion-like)
    
    Imaginary time evolution is like "annealing" - the system
    relaxes toward its ground state / most stable configuration.
    
    This represents the "before times" where possible universes
    explore configuration space before collapse.
    """
    N = len(psi_0)
    dx = 2 * np.pi / N
    dtau = tau_duration / tau_steps
    
    psi = psi_0.copy()
    
    # Pre-compute Laplacian via FFT (periodic boundary)
    k = np.fft.fftfreq(N, dx / (2 * np.pi))
    k2 = k ** 2
    
    for _ in range(tau_steps):
        # Split-step in imaginary time
        # Diffusion step (linear part)
        psi_k = np.fft.fft(psi)
        psi_k *= np.exp(-k2 * dtau)  # Note: no 'i', this is real diffusion
        psi = np.fft.ifft(psi_k)
        
        # Nonlinear step
        psi *= np.exp(-g * np.abs(psi)**2 * dtau)
        
        # Renormalize to prevent decay to zero
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        if norm > 1e-10:
            psi *= np.sqrt(N * dx) / norm
    
    return psi


# =============================================================================
# COLLAPSE / TRANSDUCTION
# =============================================================================

@dataclass
class CollapseEvent:
    """
    The moment of collapse - Dirac delta transduction from information
    space to physical configuration.
    """
    audio_trigger: np.ndarray      # The audio sample(s) that triggered collapse
    trigger_energy: float          # Energy of triggering audio
    trigger_phase: float           # Phase information from audio
    collapse_time: float           # When collapse occurred
    
    # Extracted world parameters
    world_params: Dict = field(default_factory=dict)


def extract_collapse_trigger(audio_buffer: np.ndarray, 
                             sample_rate: int = 44100) -> Tuple[float, float, np.ndarray]:
    """
    Extract collapse parameters from audio input.
    
    The first sound is the "observer" that collapses the wavefunction.
    Different sounds -> different collapsed configurations.
    """
    # Energy of audio burst
    energy = np.sum(audio_buffer ** 2) / len(audio_buffer)
    
    # Phase from FFT of audio (first significant frequency)
    fft = np.fft.rfft(audio_buffer)
    magnitudes = np.abs(fft)
    
    # Find dominant frequency
    dominant_idx = np.argmax(magnitudes[1:]) + 1  # Skip DC
    phase = np.angle(fft[dominant_idx])
    
    # Spectral centroid (brightness)
    freqs = np.fft.rfftfreq(len(audio_buffer), 1/sample_rate)
    if np.sum(magnitudes) > 1e-10:
        centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes)
    else:
        centroid = 1000.0  # Default
    
    return energy, phase, np.array([energy, phase, centroid])


def collapse_soliton(psi_evolved: np.ndarray, 
                     audio_trigger: np.ndarray,
                     sample_rate: int = 44100) -> CollapseEvent:
    """
    Perform the collapse: transduce soliton + audio into world parameters.
    
    This is the Dirac delta moment where continuous potential
    becomes discrete actuality.
    """
    # Extract audio influence
    trigger_energy, trigger_phase, trigger_features = extract_collapse_trigger(
        audio_trigger, sample_rate
    )
    
    # === NORMALIZE AUDIO ENERGY TO PERCEPTUAL LOUDNESS ===
    # Raw RMS is typically 0.001-0.5. Convert to log-scale "loudness" 0-1
    # Reference: -60dB (silence) to 0dB (loud)
    db = 20 * np.log10(trigger_energy + 1e-10)
    db_normalized = np.clip((db + 60) / 60, 0, 1)  # -60dB->0, 0dB->1
    loudness = db_normalized  # This is now 0-1 perceptual loudness
    
    # Soliton features
    soliton_amplitude = np.max(np.abs(psi_evolved))
    soliton_energy = np.sum(np.abs(psi_evolved)**2)
    soliton_phase = np.angle(psi_evolved[np.argmax(np.abs(psi_evolved))])
    soliton_width = estimate_soliton_width(psi_evolved)
    soliton_centroid = compute_centroid(psi_evolved)
    
    # Soliton amplitude normalized
    soliton_amp_norm = soliton_amplitude / (1.0 + soliton_amplitude)  # 0-1
    
    # === TRANSDUCTION: Map soliton -- audio -> world parameters ===
    # Now loudness (0-1) meaningfully affects everything
    
    # World size: base from soliton, scaled by loudness
    # Quiet sound -> smaller world (80-100), loud -> larger (100-140)
    world_size = 80.0 + 20.0 * soliton_amp_norm + 40.0 * loudness
    
    # Time scale: soliton amplitude (louder -> slightly faster)
    dt = 0.01 + 0.015 * soliton_amp_norm + 0.005 * loudness
    
    # Population counts: soliton width + loudness boost
    base_pop = 3 + int(5 * np.clip(soliton_width / 5.0, 0, 1))
    loudness_boost = 1.0 + 0.5 * loudness  # 1.0x to 1.5x multiplier
    
    n_herbies = int(base_pop * loudness_boost)
    n_apex = int(1 + 3 * loudness)  # Quiet->1, loud->4
    n_blobs = int(base_pop * 0.8 * loudness_boost)
    n_bipeds = int(base_pop * 1.0 * loudness_boost)
    
    # Terrain seed: combined phase (soliton + audio)
    terrain_seed = int(abs((soliton_phase + trigger_phase) * 1e6)) % (2**31)
    
    # Resource abundance: soliton centroid + loudness
    # Loud sounds -> more resources (energetic universe)
    nutrient_density = 0.3 + 0.3 * (soliton_centroid / len(psi_evolved) + 0.5) + 0.2 * loudness
    
    # Day/night cycle: spectral centroid (bright sound -> longer days)
    spectral_brightness = np.clip(trigger_features[2] / 4000, 0, 1)  # Normalize to 0-1
    day_length = int(400 + 400 * spectral_brightness + 200 * loudness)
    
    # Creature parameters from soliton + audio texture
    base_metabolism = 0.7 + 0.3 * soliton_amp_norm + 0.2 * loudness
    base_speed = 0.6 + 0.4 * np.clip(abs(soliton_phase) / np.pi, 0, 1) + 0.2 * loudness
    
    # Mendelian genetics initial allele frequencies from soliton FFT
    psi_fft = np.fft.fft(psi_evolved)
    fft_mags = np.abs(psi_fft[:6])  # First 6 modes
    fft_mags = fft_mags / (np.sum(fft_mags) + 1e-10)
    
    # Map to dominant allele frequencies - loudness adds variation
    allele_freqs = {
        'chromosome_0': 0.3 + 0.4 * fft_mags[0] + 0.1 * loudness,  # Metabolism genes
        'chromosome_1': 0.3 + 0.4 * fft_mags[1] + 0.1 * loudness,  # Locomotion genes
        'chromosome_2': 0.3 + 0.4 * fft_mags[2] + 0.1 * (1-loudness),  # Sensory genes (inverse)
        'chromosome_3': 0.3 + 0.4 * fft_mags[3] + 0.1 * loudness,  # Behavioral genes
    }
    
    world_params = {
        'world_size': world_size,
        'dt': dt,
        'n_herbies': max(4, n_herbies),
        'n_apex': max(1, n_apex),
        'n_blobs': max(2, n_blobs),
        'n_bipeds': max(2, n_bipeds),
        'terrain_seed': terrain_seed,
        'nutrient_density': np.clip(nutrient_density, 0.2, 1.0),
        'day_length': day_length,
        'base_metabolism': np.clip(base_metabolism, 0.5, 1.5),
        'base_speed': np.clip(base_speed, 0.5, 1.5),
        'initial_allele_frequencies': allele_freqs,
        
        # Metadata
        'soliton_amplitude': float(soliton_amplitude),
        'soliton_energy': float(soliton_energy),
        'soliton_phase': float(soliton_phase),
        'audio_energy': float(trigger_energy),
        'audio_loudness': float(loudness),  # The normalized 0-1 value
        'audio_db': float(db),
        'audio_phase': float(trigger_phase),
        'spectral_brightness': float(spectral_brightness),
    }
    
    return CollapseEvent(
        audio_trigger=audio_trigger,
        trigger_energy=trigger_energy,
        trigger_phase=trigger_phase,
        collapse_time=0.0,
        world_params=world_params
    )


def estimate_soliton_width(psi: np.ndarray) -> float:
    """Estimate width of soliton from intensity profile."""
    intensity = np.abs(psi) ** 2
    total = np.sum(intensity)
    if total < 1e-10:
        return 1.0
    
    # Width as second moment
    x = np.arange(len(psi))
    mean_x = np.sum(x * intensity) / total
    var_x = np.sum((x - mean_x)**2 * intensity) / total
    return np.sqrt(var_x) if var_x > 0 else 1.0


def compute_centroid(psi: np.ndarray) -> float:
    """Compute center of mass of soliton."""
    intensity = np.abs(psi) ** 2
    total = np.sum(intensity)
    if total < 1e-10:
        return len(psi) / 2
    x = np.arange(len(psi))
    return np.sum(x * intensity) / total


# =============================================================================
# COSMOLOGICAL INITIALIZATION
# =============================================================================

class CosmologicalSeed:
    """
    Complete cosmological initialization system.
    
    Usage:
        seed = CosmologicalSeed()
        seed.prepare_soliton(...)  # Create pre-cosmological state
        seed.evolve_pre_cosmos()   # Imaginary time evolution
        
        # When ready to start simulation:
        audio = get_microphone_input()
        params = seed.big_bang(audio)  # Collapse and extract parameters
    """
    
    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        self.x = np.linspace(-10, 10, resolution)
        
        self.profile: Optional[SolitonProfile] = None
        self.psi_initial: Optional[np.ndarray] = None
        self.psi_evolved: Optional[np.ndarray] = None
        self.collapse_event: Optional[CollapseEvent] = None
        
    def prepare_soliton(self, profile: SolitonProfile = None, 
                        seed: int = None) -> np.ndarray:
        """
        Prepare the pre-cosmological soliton.
        
        If seed is provided, use it to generate soliton parameters.
        Otherwise use profile or random.
        """
        if seed is not None:
            np.random.seed(seed)
            profile = SolitonProfile(
                amplitude=0.5 + np.random.random(),
                velocity=np.random.randn() * 0.3,
                width=0.5 + np.random.random() * 2,
                phase=np.random.random() * 2 * np.pi,
                n_solitons=np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            )
        
        if profile is None:
            profile = SolitonProfile()
        
        self.profile = profile
        
        # Generate initial soliton waveform
        if profile.n_solitons > 1:
            self.psi_initial = generate_multi_soliton(
                self.x, profile.soliton_params, profile.soliton_type
            )
        elif profile.soliton_type == SolitonType.KDV:
            self.psi_initial = generate_kdv_soliton(
                self.x, profile.amplitude, profile.velocity, profile.position
            ).astype(complex)
        elif profile.soliton_type == SolitonType.NLSE_DARK:
            self.psi_initial = generate_nlse_dark_soliton(
                self.x, profile.amplitude, 1.0, profile.velocity, profile.position
            )
        else:  # Default: bright soliton
            self.psi_initial = generate_nlse_bright_soliton(
                self.x, profile.amplitude, profile.width,
                profile.velocity, profile.phase, profile.position
            )
        
        return self.psi_initial
    
    def evolve_pre_cosmos(self, g: float = -1.0) -> np.ndarray:
        """
        Evolve soliton in imaginary time (the "before times").
        
        This is the pre-cosmological phase where the potential
        universe explores configuration space.
        """
        if self.psi_initial is None:
            self.prepare_soliton()
        
        self.psi_evolved = evolve_in_imaginary_time(
            self.psi_initial,
            self.profile.tau_duration,
            self.profile.tau_steps,
            g=g
        )
        
        return self.psi_evolved
    
    def big_bang(self, audio_trigger: np.ndarray, 
                 sample_rate: int = 44100) -> Dict:
        """
        THE BIG BANG: Collapse soliton into world parameters.
        
        The audio input is the observer that collapses the
        superposition of possible universes into one determined
        configuration.
        """
        if self.psi_evolved is None:
            self.evolve_pre_cosmos()
        
        self.collapse_event = collapse_soliton(
            self.psi_evolved, audio_trigger, sample_rate
        )
        
        p = self.collapse_event.world_params
        
        print("\n" + "="*60)
        print("         C O S M O L O G I C A L   C O L L A P S E")
        print("="*60)
        print(f"  SOLITON STATE:")
        print(f"    Amplitude:     {p['soliton_amplitude']:.4f}")
        print(f"    Energy:        {p['soliton_energy']:.4f}")
        print(f"    Phase:         {p['soliton_phase']:.4f} rad")
        print(f"  AUDIO TRIGGER:")
        print(f"    Raw energy:    {p['audio_energy']:.6f}")
        print(f"    Level (dB):    {p['audio_db']:.1f} dB")
        print(f"    Loudness:      {p['audio_loudness']:.3f} (0=silent, 1=loud)")
        print(f"    Brightness:    {p['spectral_brightness']:.3f}")
        print("-"*60)
        print("  EMERGENT UNIVERSE:")
        print(f"    World size:      {p['world_size']:.1f}")
        print(f"    Time scale (dt): {p['dt']:.4f}")
        print(f"    Day length:      {p['day_length']} steps")
        print(f"  POPULATIONS:")
        print(f"    Herbies:         {p['n_herbies']}")
        print(f"    Blobs:           {p['n_blobs']}")
        print(f"    Bipeds:          {p['n_bipeds']}")
        print(f"    Apex:            {p['n_apex']}")
        print(f"  ECOLOGY:")
        print(f"    Resources:       {p['nutrient_density']:.2f}")
        print(f"    Metabolism:      {p['base_metabolism']:.2f}")
        print(f"    Speed:           {p['base_speed']:.2f}")
        print(f"  TERRAIN:")
        print(f"    Seed:            {p['terrain_seed']}")
        print("="*60 + "\n")
        
        return self.collapse_event.world_params
    
    def get_visualization_data(self) -> Dict:
        """Get data for visualizing the soliton and collapse."""
        return {
            'x': self.x,
            'psi_initial': self.psi_initial,
            'psi_evolved': self.psi_evolved,
            'profile': self.profile,
            'collapse': self.collapse_event
        }


# =============================================================================
# AUDIO CAPTURE FOR COLLAPSE
# =============================================================================

def capture_collapse_audio(duration: float = 0.5, 
                           sample_rate: int = 44100,
                           threshold: float = 0.005,
                           warmup_seconds: float = 0.2,
                           window_ms: float = 30.0) -> np.ndarray:
    """
    Capture audio from microphone to trigger collapse.
    
    Waits for *sustained* sound above threshold (not just single-sample spikes).
    This prevents mic initialization artifacts from triggering premature collapse.
    
    Args:
        duration: Duration of audio to capture after trigger (seconds)
        sample_rate: Audio sample rate
        threshold: RMS energy threshold to trigger (0.0-1.0, typical speech ~0.05-0.2)
        warmup_seconds: Ignore this much audio at start (mic initialization)
        window_ms: Window size for RMS calculation (milliseconds)
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("[Cosmological] sounddevice not available, using synthetic trigger")
        return np.random.randn(int(duration * sample_rate)) * 0.1
    
    buffer_size = int(duration * sample_rate)
    warmup_samples = int(warmup_seconds * sample_rate)
    window_samples = int(window_ms * sample_rate / 1000)
    
    print("[Cosmological] Waiting for audio trigger...")
    print(f"              (Make a sound to collapse the wavefunction)")
    print(f"              [threshold={threshold:.3f}, window={window_ms:.0f}ms]")
    
    # Record with a longer buffer, find the trigger point
    max_wait = 15.0  # Maximum wait time
    recording = sd.rec(int(max_wait * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1, 
                       dtype='float32')
    sd.wait()
    recording = recording.flatten()
    
    # Skip warmup period (mic initialization artifacts)
    search_start = warmup_samples
    
    # Calculate RMS in sliding windows
    trigger_idx = None
    for i in range(search_start, len(recording) - window_samples):
        window = recording[i:i + window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        
        if rms > threshold:
            trigger_idx = i
            print(f"[Cosmological] Trigger detected! RMS={rms:.4f} at sample {trigger_idx}")
            print(f"              (time = {trigger_idx / sample_rate:.2f}s)")
            break
    
    if trigger_idx is not None:
        # Extract buffer starting from trigger
        end_idx = min(trigger_idx + buffer_size, len(recording))
        audio = recording[trigger_idx:end_idx]
        
        # Pad if necessary
        if len(audio) < buffer_size:
            audio = np.pad(audio, (0, buffer_size - len(audio)))
        
        # Report the energy level
        final_rms = np.sqrt(np.mean(audio ** 2))
        print(f"[Cosmological] Captured {len(audio)} samples, RMS={final_rms:.4f}")
        return audio
    else:
        # No trigger - report max RMS seen
        max_rms = 0.0
        for i in range(search_start, len(recording) - window_samples, window_samples):
            window = recording[i:i + window_samples]
            rms = np.sqrt(np.mean(window ** 2))
            max_rms = max(max_rms, rms)
        
        print(f"[Cosmological] No trigger detected (max RMS={max_rms:.4f}, threshold={threshold:.3f})")
        print(f"[Cosmological] Using ambient noise for collapse...")
        return recording[warmup_samples:warmup_samples + buffer_size]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def cosmological_init(seed: int = None, 
                      wait_for_audio: bool = True,
                      audio_duration: float = 0.5) -> Dict:
    """
    Complete cosmological initialization in one call.
    
    Args:
        seed: Optional seed for soliton generation
        wait_for_audio: If True, wait for microphone input to trigger collapse
        audio_duration: Duration of audio to capture for collapse
        
    Returns:
        Dictionary of world parameters extracted from collapse
    """
    cosmos = CosmologicalSeed()
    
    # Prepare and evolve soliton
    cosmos.prepare_soliton(seed=seed)
    cosmos.evolve_pre_cosmos()
    
    # Get collapse trigger
    if wait_for_audio:
        audio = capture_collapse_audio(duration=audio_duration)
    else:
        # Use synthetic noise (for testing)
        audio = np.random.randn(int(audio_duration * 44100)) * 0.1
    
    # BIG BANG
    return cosmos.big_bang(audio)


# =============================================================================
# TESTING / DEMO
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("    HERBIE WORLD - COSMOLOGICAL SEED SYSTEM")
    print("    V6_ALT: CE Theory Universe Initialization")
    print("="*60)
    
    # Create cosmological seed
    cosmos = CosmologicalSeed(resolution=256)
    
    # Prepare soliton (using command line seed if provided)
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"\nPreparing soliton (seed={seed})...")
    cosmos.prepare_soliton(seed=seed)
    
    print(f"  Type: {cosmos.profile.soliton_type.value}")
    print(f"  Amplitude: {cosmos.profile.amplitude:.3f}")
    print(f"  Width: {cosmos.profile.width:.3f}")
    print(f"  N solitons: {cosmos.profile.n_solitons}")
    
    # Evolve in imaginary time
    print("\nEvolving in imaginary time (pre-cosmological phase)...")
    cosmos.evolve_pre_cosmos()
    print("  Evolution complete.")
    
    # Trigger collapse
    print("\nReady for collapse.")
    
    if '--no-audio' in sys.argv:
        print("Using synthetic trigger...")
        audio = np.random.randn(22050) * 0.1
    else:
        audio = capture_collapse_audio()
    
    # BIG BANG
    params = cosmos.big_bang(audio)
    
    print("\nWorld parameters ready for simulation initialization.")
    print("These would be passed to CreatureManager and World setup.")
