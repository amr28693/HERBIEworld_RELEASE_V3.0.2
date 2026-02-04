"""
Gator - Amphibious apex predator with dual NLSE body dynamics.

Gators are large, slow predators that:
- Move faster in water than on land
- Have a separate tail field for swimming propulsion
- Hunt other creatures with high damage strikes
- Have armored bodies that reduce incoming damage
"""

from typing import List, TYPE_CHECKING
import numpy as np

from ..core.constants import (
    WORLD_L, Nx, Ny, X, Y, dt, G_SMOOTH, L_op_2d,
    AFFERENT_CHANNELS
)
from ..body.skeleton import PiezoSkeleton
from ..events.console_log import console_log
from .creature import Creature
from .species import SPECIES_GATOR

if TYPE_CHECKING:
    from ..world.multi_world import MultiWorld


class Gator(Creature):
    """
    Amphibious predator with dual body fields.
    
    The Gator has:
    - Main body field (from Creature base class)
    - Separate tail field for swimming propulsion
    - Two piezoelectric skeletons (body and tail)
    - Amphibious locomotion (faster in water)
    - High damage hunting capability
    - Armor that reduces incoming damage
    """
    
    def __init__(self, world: 'MultiWorld', pos: np.ndarray, generation: int = 0,
                 parent_id: str = "", lineage_depth: int = 0):
        """Initialize Gator with dual body dynamics."""
        # Initialize base creature
        super().__init__(SPECIES_GATOR, world, pos, generation, parent_id, lineage_depth)
        
        # === TAIL FIELD (separate from body) ===
        # Initialize as localized wavepacket offset from body center
        tail_offset = np.array([-2.0, 0.0])  # Tail behind body
        tail_center = np.array([Nx//2, Ny//2]) + tail_offset * 5
        
        # Create tail wavefunction (complex field like body.psi)
        self.tail_psi = np.zeros((Ny, Nx), dtype=np.complex128)
        r2 = (X - tail_center[0]/5)**2 + (Y - tail_center[1]/5)**2
        self.tail_psi = 0.5 * np.exp(-r2 / 2.0).astype(np.complex128)
        self.tail_g = SPECIES_GATOR.body_g_base * 0.8  # Slightly less nonlinear than body
        
        # === DUAL PIEZOELECTRIC SKELETONS ===
        # Body skeleton already created by parent
        self.body_bone = self.skeleton  # Alias for clarity
        self.tail_bone = PiezoSkeleton()  # Separate tail skeleton
        
        # === AMPHIBIOUS STATE ===
        self.in_water = False
        self.water_speed_mult = 2.5  # Much faster in water
        self.land_speed_mult = 0.7   # Slower on land
        
        # === TAIL OSCILLATION STATE ===
        self.tail_phase = 0.0
        self.tail_amplitude = 0.0
        self.tail_thrust = np.zeros(2)
        
        # === HUNTING/COMBAT ===
        self.armor_value = 0.8  # Reduces damage taken by 80%
        self.strike_cooldown = 0
        self.last_prey_direction = np.zeros(2)
        
        # Reduce torus nonlinearity for simpler brain dynamics
        self.torus.g *= 0.5
        
        console_log().log(f"[Gator] Spawned at ({pos[0]:.1f}, {pos[1]:.1f})", force=True)
    
    def step(self, all_creatures: List, audio_amp: float = 0.0,
             silence_frames: int = 100, audio_system=None, season=None,
             deferred_attacks: bool = False):
        """
        Full Gator step - follows Creature.step() pattern exactly.
        """
        if not self.alive:
            return 'dead'
        
        # === BOOKKEEPING (same as Creature.step) ===
        self.step_count += 1
        self.hunt_cooldown = max(0, self.hunt_cooldown - 1)
        self.damage_taken_this_step = 0.0
        self.lifetime_tracker.update(self, self.step_count)
        
        # === WATER STATE ===
        self._update_water_state()
        
        # === DREAM STATE (same as Creature.step) ===
        if silence_frames < 80:
            self.dream_depth = max(0.0, self.dream_depth - 0.018)
        else:
            self.dream_depth = 0.94 * self.dream_depth + 0.06 * np.clip((silence_frames - 80) / 180, 0, 1)
        self.is_dreaming = self.dream_depth > 0.15
        
        # === SENSE CREATURES (same as Creature.step) ===
        self.sense_nearby_creatures(all_creatures)
        
        # === HUNTING (predator behavior from Creature.step) ===
        hunt_energy = 0.0
        if self.strike_cooldown > 0:
            self.strike_cooldown -= 1
        
        if not self.is_dreaming and not self.is_digesting:
            prey = self.find_prey(all_creatures, hunt_radius=15.0)
            if prey is not None:
                self.current_prey_target = prey.creature_id
                direction = prey.pos - self.pos
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    direction = direction / dist
                    self.last_prey_direction = direction
                    # Pursuit - inject into torus
                    pursuit_strength = 0.6 + 0.6 * self.metabolism.hunger
                    self.vel += direction * pursuit_strength * self.species.speed_factor * 0.3
                    angle = np.arctan2(direction[1], direction[0])
                    self.torus.inject_reward(angle, 0.4)
                
                # Attack at close range
                if dist < 4.0 and self.strike_cooldown <= 0:
                    hunt_energy = self.attack_prey(prey, deferred=deferred_attacks)
                    self.strike_cooldown = 60  # Cooldown after strike
            else:
                self.current_prey_target = None
        
        # === AFFERENT CHANNELS (same as Creature.step) ===
        for ch in self.afferent.values():
            ch.evolve()
            for a in ch.get_arrivals():
                self.torus.receive_afferent(ch.name, [a])
        
        # === TORUS (same as Creature.step) ===
        g_target = self.species.torus_g_base
        g_target += 0.6 * audio_amp
        g_target -= 0.5 * self.metabolism.hunger
        g_target -= 1.2 * self.dream_depth
        self.torus.g = G_SMOOTH * self.torus.g + (1 - G_SMOOTH) * g_target
        self.torus.evolve(audio_amp, self.metabolism.hunger, self.dream_depth)
        
        # Extract torus state (same as Creature.step)
        torus_arousal = self.torus.get_arousal()
        torus_phase = float(np.angle(self.torus.psi[np.argmax(np.abs(self.torus.psi)**2)]))
        torus_bias = self.torus.get_directional_bias()
        
        # === EFFERENT FIRING (same as Creature.step) ===
        if self.torus.should_fire_efferent():
            patterns = self.torus.get_efferent_pattern()
            for i, (_, amp) in enumerate(patterns[:len(self.efferent)]):
                amp_scaled = amp * self.species.efferent_strength
                eff_keys = list(self.efferent.keys())
                if i < len(eff_keys):
                    self.efferent[eff_keys[i]].nucleate(amp_scaled, np.random.random())
                
                if i < len(self.limb_defs):
                    limb_name = list(self.limb_defs.keys())[i]
                    self.limbs[limb_name].inject_efferent(amp_scaled * 0.7)
                    self.morph.apply_efferent_torque(limb_name, amp_scaled * 0.6)
        
        for ch in self.efferent.values():
            ch.evolve()
        
        # === BODY SKELETON (same as Creature.step) ===
        body_I = self.get_body_I()
        self.body_bone.step(body_I)
        
        # === TAIL SKELETON (Gator-specific) ===
        tail_I = np.abs(self.tail_psi)**2
        self.tail_bone.step(tail_I)
        
        # === POTENTIAL (same as Creature.step) ===
        V_total = self._compute_potential()
        
        # === BODY FIELD (same as Creature.step) ===
        g_base = self.species.body_g_base
        g_target = g_base + 0.5 * torus_arousal + self.metabolism.get_g_modifier() + 0.4 * audio_amp - 0.4 * self.dream_depth
        self.body.g = G_SMOOTH * self.body.g + (1 - G_SMOOTH) * g_target
        self.body.evolve(V_total, torus_phase, torus_arousal, self.metabolism.get_g_modifier(),
                         self.dream_depth, audio_amp)
        
        # === TAIL FIELD EVOLUTION (Gator-specific, using same split-step as BodyField) ===
        tail_excitation = torus_arousal * 0.6 + self.tail_bone.rms * 0.3
        self.tail_phase += 0.15 * (1 + tail_excitation)
        self.tail_amplitude = 0.3 + 0.7 * tail_excitation
        
        # Split-step Fourier NLSE (same method as BodyField.evolve)
        # Kinetic step
        tail_k = np.fft.fft2(self.tail_psi) * L_op_2d
        self.tail_psi = np.fft.ifft2(tail_k)
        # Nonlinear step
        tail_I = np.abs(self.tail_psi)**2
        self.tail_psi *= np.exp(1j * self.tail_g * dt * tail_I)
        
        # Inject oscillation for swimming
        tail_signal = self.tail_amplitude * np.sin(self.tail_phase)
        if np.linalg.norm(torus_bias) > 0.01:
            kx = torus_bias[0] * tail_signal * 0.3
            ky = torus_bias[1] * tail_signal * 0.3
            self.tail_psi *= np.exp(1j * (kx * X + ky * Y) * 0.1)
        
        # Damping
        self.tail_psi *= 0.995
        
        # === LIMBS (same as Creature.step) ===
        limb_thrust = np.zeros(2)
        for limb_name, limb in self.limbs.items():
            origin_name = self.limb_defs[limb_name][0]
            phase, amp = self.body.get_region_state(origin_name)
            limb.inject_from_body(amp, phase + 0.15 * torus_phase, torus_arousal)
            
            limb.g = 0.9 * limb.g + 0.1 * (self.species.limb_g_base + 0.4 * torus_arousal)
            limb.evolve(torus_arousal, self.metabolism.hunger, self.dream_depth)
            
            if limb.pulse_position > 0.5 and limb.pulse_amplitude > 0.04:
                tip_phase, tip_amp = limb.get_tip_state()
                self.body.inject_at_region(origin_name, tip_amp * 0.15, tip_phase)
                angle = self.morph.limb_angles[limb_name]
                limb_thrust += np.array([-np.cos(angle), -np.sin(angle)]) * limb.pulse_amplitude * abs(self.morph.limb_velocities.get(limb_name, 0)) * 0.4
        
        if self.limb_defs:
            self.morph.update_limb_angles(self.limbs, torus_bias, self.body_bone.rms,
                                          self.metabolism.hunger, audio_amp)
        
        # === MOVEMENT (Gator-specific: amphibious) ===
        body_momentum = self.body.get_momentum()
        
        if np.linalg.norm(torus_bias) > 0.05:
            self.body.inject_momentum(torus_bias, torus_arousal * 0.5)
        if np.linalg.norm(limb_thrust) > 0.02:
            self.body.inject_momentum(limb_thrust, 0.3)
        
        # Calculate tail thrust (water only)
        if self.in_water:
            # Tail provides extra propulsion in water
            tail_I = np.abs(self.tail_psi)**2
            tail_total = np.sum(tail_I) + 1e-6
            
            # Tail momentum from phase gradient
            from ..core.constants import dx
            dpsi_x = np.gradient(self.tail_psi, dx, axis=1)
            dpsi_y = np.gradient(self.tail_psi, dx, axis=0)
            tail_px = np.sum(np.imag(np.conj(self.tail_psi) * dpsi_x)) / tail_total
            tail_py = np.sum(np.imag(np.conj(self.tail_psi) * dpsi_y)) / tail_total
            tail_momentum = np.array([tail_px, tail_py]) * 0.5
            
            # Lateral tail thrust
            tail_lateral = np.array([-torus_bias[1], torus_bias[0]]) if np.linalg.norm(torus_bias) > 0.1 else np.zeros(2)
            self.tail_thrust = tail_lateral * self.tail_amplitude * 0.3 + tail_momentum * 0.5
        else:
            self.tail_thrust = np.zeros(2)
        
        # Apply movement with amphibious speed modifier
        speed_mult = self.water_speed_mult if self.in_water else self.land_speed_mult
        speed_factor = self.species.speed_factor * speed_mult
        hunger_boost = 1.0 + self.metabolism.hunger * 0.5
        
        force = body_momentum * 5.0 * hunger_boost * speed_factor
        force += torus_bias * 1.5 * hunger_boost * speed_factor
        force += limb_thrust * 2.5 * speed_factor
        force += self.tail_thrust * 3.0 * speed_factor  # Tail thrust in water
        
        # NaN guard for force
        if np.any(np.isnan(force)):
            force = np.zeros(2)
        
        speed = np.linalg.norm(self.vel)
        if speed < 0.5 or self.metabolism.hunger > 0.5:
            force += np.random.randn(2) * (0.4 + 0.3 * self.metabolism.hunger)
        
        # Boundary forces
        boundary_margin = WORLD_L/2 - 5
        for i in range(2):
            if self.pos[i] < -boundary_margin:
                force[i] += 2.0 * (-boundary_margin - self.pos[i])
            elif self.pos[i] > boundary_margin:
                force[i] -= 2.0 * (self.pos[i] - boundary_margin)
        
        self.vel += force * dt * 20
        self.vel *= 0.96
        
        # NaN guard for velocity
        if np.any(np.isnan(self.vel)):
            self.vel = np.zeros(2)
        
        speed = np.linalg.norm(self.vel)
        max_speed = (2.5 + self.metabolism.hunger * 0.5) * speed_factor
        if speed > max_speed and speed > 0:
            self.vel = self.vel / speed * max_speed
        
        self.pos += self.vel * dt
        self.pos = np.clip(self.pos, -WORLD_L/2 + 2, WORLD_L/2 - 2)
        
        # Final NaN guard for position
        if np.any(np.isnan(self.pos)):
            self.pos = np.zeros(2)
        
        # === WORLD INTERACTION (same as Creature.step) ===
        body_I = self.get_body_I()
        self.total_reward, self.mass_extracted = self.world.process_creature_interactions(
            self, body_I, body_momentum
        )
        
        self.total_reward += hunt_energy
        self.mass_extracted += hunt_energy
        
        # === AFFERENT SENSING (same as Creature.step) ===
        self._do_afferent_sensing(V_total, audio_amp, audio_system)
        
        # === METABOLISM (same as Creature.step) ===
        reward_source = self.world.get_reward_source_for_creature(self)
        vel_mag = np.linalg.norm(self.vel)
        self.metabolism.update(self.total_reward, self.mass_extracted, vel_mag, reward_source)
        
        # Defecation
        if self.metabolism.defecation_pending:
            self.world.drop_nutrient(self.pos.copy(), self.metabolism.last_defecation_amount)
            self.lifetime_tracker.log_defecation(self.metabolism.last_defecation_amount, self.step_count)
        
        self.display_vel = 0.8 * self.display_vel + 0.2 * self.vel
        
        # === DEATH/REPRODUCTION (same as Creature.step) ===
        if self.metabolism.is_dead:
            self.alive = False
            console_log().log(f"[Gator] Died: {self.metabolism.cause_of_death}", force=True)
            return 'dead'
        
        if self.metabolism.ready_to_reproduce:
            self.metabolism.ready_to_reproduce = False
            self.metabolism.total_consumed *= 0.5
            self.metabolism.hunger = min(1.0, self.metabolism.hunger + 0.3)
            self.body.energy *= 0.6
            self.body.psi *= np.sqrt(0.6)
            self.offspring_count += 1
            self.lifetime_tracker.log_reproduction(self.step_count)
            console_log().log(f"[Gator] Reproduced!", force=True)
            return 'reproduce'
        
        return 'alive'
    
    def _update_water_state(self):
        """Check if Gator is in water based on terrain."""
        if hasattr(self.world, 'terrain') and self.world.terrain:
            terrain_type = self.world.terrain.get_terrain_at(self.pos)
            self.in_water = terrain_type.name == 'water'
        else:
            self.in_water = False
    
    def receive_damage(self, damage: float, source: str = "unknown") -> float:
        """
        Override damage reception with armor.
        Returns actual damage taken after armor reduction.
        """
        # Armor reduces damage
        actual_damage = damage * (1.0 - self.armor_value)
        
        # Apply to metabolism/body
        self.metabolism.hunger = min(1.0, self.metabolism.hunger + actual_damage * 0.3)
        self.body.energy *= (1.0 - actual_damage * 0.1)
        
        # Pain signal
        if 'env_pain' in self.afferent:
            self.afferent['env_pain'].nucleate(actual_damage * 2.0, 0.0)
        
        # Death check
        if self.metabolism.hunger >= 0.99 or self.body.energy < 5:
            self.alive = False
            self.metabolism.is_dead = True
            self.metabolism.cause_of_death = f"killed_by:{source}"
        
        self.damage_taken_this_step += actual_damage
        return actual_damage
    
    def get_meat_value(self) -> dict:
        """Gators provide high-quality meat when killed."""
        body_I = np.abs(self.body.psi)**2
        return {
            'chunks': 6,  # More meat than smaller creatures
            'energy_per_chunk': self.body.energy * 0.25,
            'meat_energy': self.body.energy * 1.5,
        }
    
    def get_state(self) -> dict:
        """Return Gator-specific state for visualization."""
        base_state = {
            'pos': self.pos.copy(),
            'vel': self.vel.copy(),
            'alive': self.alive,
            'in_water': self.in_water,
            'tail_amplitude': self.tail_amplitude,
            'tail_phase': self.tail_phase,
            'body_stress': self.body_bone.rms,
            'tail_stress': self.tail_bone.rms,
            'strike_cooldown': self.strike_cooldown,
            'armor': self.armor_value,
            'hunger': self.metabolism.hunger,
            'energy': self.body.energy,
        }
        return base_state
    
    def to_dict(self) -> dict:
        """Serialize Gator for persistence."""
        base = super().to_dict()
        base.update({
            # Tail field (the second NLSE field)
            'tail_psi_real': self.tail_psi.real.tolist(),
            'tail_psi_imag': self.tail_psi.imag.tolist(),
            'tail_g': self.tail_g,
            'tail_phase': self.tail_phase,
            'tail_amplitude': self.tail_amplitude,
            
            # Tail skeleton
            'tail_bone': self.tail_bone.to_dict() if hasattr(self.tail_bone, 'to_dict') else {},
            
            # Amphibious state
            'in_water': self.in_water,
            'water_speed_mult': self.water_speed_mult,
            'land_speed_mult': self.land_speed_mult,
            
            # Combat
            'armor_value': self.armor_value,
            'strike_cooldown': self.strike_cooldown,
            'last_prey_direction': self.last_prey_direction.tolist(),
        })
        return base
    
    @classmethod
    def from_dict(cls, data: dict, world: 'MultiWorld') -> 'Gator':
        """
        Deserialize Gator from saved state.
        
        Args:
            data: Dict from to_dict()
            world: World to place Gator in
            
        Returns:
            Restored Gator instance
        """
        # First restore base creature
        from .creature import Creature
        
        # Create gator using parent's from_dict logic
        gator = Creature.from_dict.__func__(cls, data, world)
        
        # === GATOR-SPECIFIC ===
        
        # Restore tail field
        if 'tail_psi_real' in data:
            gator.tail_psi = np.array(data['tail_psi_real']) + 1j * np.array(data['tail_psi_imag'])
        else:
            # Initialize fresh tail
            tail_center = np.array([Nx//2, Ny//2]) + np.array([-10, 0])
            r2 = (X - tail_center[0]/5)**2 + (Y - tail_center[1]/5)**2
            gator.tail_psi = 0.5 * np.exp(-r2 / 2.0).astype(np.complex128)
        
        gator.tail_g = data.get('tail_g', SPECIES_GATOR.body_g_base * 0.8)
        gator.tail_phase = data.get('tail_phase', 0.0)
        gator.tail_amplitude = data.get('tail_amplitude', 0.0)
        gator.tail_thrust = np.zeros(2)
        
        # Dual skeletons - body_bone is alias for skeleton
        gator.body_bone = gator.skeleton
        if data.get('tail_bone'):
            gator.tail_bone = PiezoSkeleton.from_dict(data['tail_bone'])
        else:
            gator.tail_bone = PiezoSkeleton()
        
        # Amphibious state
        gator.in_water = data.get('in_water', False)
        gator.water_speed_mult = data.get('water_speed_mult', 2.5)
        gator.land_speed_mult = data.get('land_speed_mult', 0.7)
        
        # Combat
        gator.armor_value = data.get('armor_value', 0.8)
        gator.strike_cooldown = data.get('strike_cooldown', 0)
        gator.last_prey_direction = np.array(data.get('last_prey_direction', [0.0, 0.0]))
        
        return gator


def spawn_gator(world: 'MultiWorld', pos: np.ndarray = None, generation: int = 0) -> Gator:
    """
    Spawn a new Gator at the specified position.
    Prefers water-adjacent locations.
    """
    if pos is None:
        # Try to spawn near water
        half = world.world_size / 2
        for _ in range(10):
            test_pos = np.array([
                np.random.uniform(-half + 5, half - 5),
                np.random.uniform(-half + 5, half - 5)
            ])
            if hasattr(world, 'terrain') and world.terrain:
                terrain = world.terrain.get_terrain_at(test_pos)
                if terrain.name == 'water':
                    pos = test_pos
                    break
        if pos is None:
            pos = np.array([
                np.random.uniform(-half + 5, half - 5),
                np.random.uniform(-half + 5, half - 5)
            ])
    
    return Gator(world, pos, generation)
