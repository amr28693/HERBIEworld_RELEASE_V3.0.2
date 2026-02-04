"""
Creature - Base creature class with NLSE body, torus brain, and behavior.

This is the core creature implementation used by all species.
Herbie-specific features (hands, mating) are in separate modules.
"""

import time
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.constants import (
    WORLD_L, BODY_L, Nx, Ny, X, Y, dt, G_SMOOTH,
    AFFERENT_CHANNELS
)
from ..brain.torus import TorusBrain
from ..brain.kdv_channels import KdVChannel
from ..body.body_field import BodyField
from ..body.limbs import LimbField
from ..body.skeleton import PiezoSkeleton
from ..body.metabolism import Metabolism
from ..events.logger import event_log
from ..statistics import LifetimeTracker
from .species import SpeciesParams, get_species_basins, get_species_limb_defs
from .traits import MutatedTraits, DIGESTION_SPEED_PENALTY

if TYPE_CHECKING:
    from ..world.objects import WorldObject
    from ..chemistry.elements import ElementType


class Creature:
    """
    A creature instance with species-specific parameters and behaviors.
    
    Each creature has:
    - Body field (NLSE): Physical form with energy dynamics
    - Torus brain: Decision-making via pattern formation
    - Limbs: KdV-driven appendages for locomotion
    - Metabolism: Energy management, hunger, aging
    - Afferent/efferent channels: Sensory input and motor output
    
    Behavior emerges from the interplay of these systems.
    """
    
    _global_id_counter = 0
    
    def __init__(
        self,
        species: SpeciesParams,
        world: 'MultiWorld',
        pos: np.ndarray = None,
        generation: int = 0,
        parent_id: str = "",
        lineage_depth: int = 0,
        inherited_knowledge: dict = None,
        inherited_traits: MutatedTraits = None
    ):
        """
        Initialize a new creature.
        
        Args:
            species: Species parameters defining morphology and behavior
            world: World the creature lives in
            pos: Initial position (random if None)
            generation: Generation number in lineage
            parent_id: ID of parent creature
            lineage_depth: Depth in family tree
            inherited_knowledge: Learned potentials from parent
            inherited_traits: Mutated traits from parent
        """
        Creature._global_id_counter += 1
        self.creature_id = f"{species.name[0]}{int(time.time()*1000) % 100000:05d}_{Creature._global_id_counter:03d}"
        
        self.species = species
        self.world = world
        self.generation = generation
        self.parent_id = parent_id
        self.lineage_depth = lineage_depth
        
        # Position in world
        if pos is None:
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(0, WORLD_L/4)
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        self.pos = pos.copy()
        self.vel = np.zeros(2)
        
        # Species-specific configuration
        self.basins = get_species_basins(species)
        self.limb_defs = get_species_limb_defs(species)
        
        # === TRAITS (mutations) ===
        self.traits = inherited_traits if inherited_traits else MutatedTraits()
        self.effective_speed = self.traits.get('speed_factor', species.speed_factor)
        self.effective_efficiency = self.traits.get('energy_efficiency', species.energy_efficiency)
        self.effective_afferent = self.traits.get('afferent_sensitivity', species.afferent_sensitivity)
        self.effective_efferent = self.traits.get('efferent_strength', species.efferent_strength)
        self.effective_metabolism = self.traits.get('metabolism_rate', species.metabolism_rate)
        
        # === COMPONENTS ===
        self.morph = self._create_morphology()
        self.torus = self._create_torus()
        self.body = self._create_body()
        self.skeleton = PiezoSkeleton()
        self.limbs = {name: self._create_limb(name) for name in self.limb_defs}
        self.metabolism = self._create_metabolism()
        
        # Neural channels
        self.afferent = {}
        for name in AFFERENT_CHANNELS:
            self.afferent[name] = KdVChannel(name, 'afferent')
        
        # Add threat channel
        if 'threat' not in self.afferent:
            self.afferent['threat'] = KdVChannel('threat', 'afferent')
        
        self.efferent = {}
        for i in range(min(species.num_limbs, 4)):
            name = f'motor_{i}'
            self.efferent[name] = KdVChannel(name, 'efferent')
        
        if not self.efferent:
            self.efferent['motor_body'] = KdVChannel('motor_body', 'efferent')
        
        # Knowledge
        if inherited_knowledge:
            self.learned_potential = inherited_knowledge.copy()
            for k in self.learned_potential:
                self.learned_potential[k] *= np.random.uniform(0.8, 1.2)
        else:
            self.learned_potential = {}
        
        # State
        self.dream_depth = 0.0
        self.is_dreaming = False
        self.step_count = 0
        self.total_reward = 0.0
        self.mass_extracted = 0.0
        self.offspring_count = 0
        self.alive = True
        self.display_vel = np.zeros(2)
        
        # Predation state
        self.current_prey_target = None
        self.hunt_cooldown = 0
        self.meat_consumed = 0.0
        
        # Hibernation
        self.is_hibernating = False
        self.hibernation_steps = 0
        
        # Defense
        self.is_defending = False
        self.defense_successes = 0
        self.defense_attempts = 0
        self.learned_defense_bonus = 0.0
        self.damage_taken_this_step = 0.0
        
        # Digestion
        self.is_digesting = False
        self.digestion_steps_remaining = 0
        self.digestion_energy_pending = 0.0
        
        # For meat chunk spawning on death
        self._pending_meat_chunks = None
        
        # Tracking
        self.lifetime_tracker = LifetimeTracker(
            self.creature_id, self.generation, self.parent_id, self.lineage_depth
        )
        
        print(f"[{species.name}] Spawned Gen{generation} {self.creature_id[:12]} at ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # =========================================================================
    # COMPONENT CREATION
    # =========================================================================
    
    def _create_morphology(self):
        """Create species-specific morphology."""
        creature = self
        
        class CreatureMorphology:
            def __init__(cm_self):
                cm_self.limb_angles = {name: info[1] for name, info in creature.limb_defs.items()}
                cm_self.limb_velocities = {name: 0.0 for name in creature.limb_defs}
                cm_self.limb_torques = {name: 0.0 for name in creature.limb_defs}
                cm_self.skin_stiffness = 7.0 * creature.species.body_scale
                cm_self.skin_thickness = 0.55 * creature.species.body_scale
                
            def get_limb_tip(cm_self, limb_name):
                origin_name, _ = creature.limb_defs[limb_name]
                origin = creature.basins[origin_name]['pos']
                angle = cm_self.limb_angles[limb_name]
                return (origin[0] + creature.species.limb_length * np.cos(angle),
                        origin[1] + creature.species.limb_length * np.sin(angle))
            
            def apply_efferent_torque(cm_self, limb_name, strength):
                if limb_name in cm_self.limb_torques:
                    cm_self.limb_torques[limb_name] += strength * 0.4
            
            def update_limb_angles(cm_self, limb_fields, torus_bias, skel_rms, hunger, audio_amp):
                for limb_name, limb in limb_fields.items():
                    if limb_name not in cm_self.limb_angles:
                        continue
                    
                    _, base_angle = creature.limb_defs[limb_name]
                    momentum = limb.get_momentum()
                    eff_torque = cm_self.limb_torques.get(limb_name, 0.0)
                    
                    angular_accel = momentum * 0.8 + eff_torque * 1.2 + skel_rms * 0.3 * (np.random.random() - 0.5)
                    cm_self.limb_velocities[limb_name] = 0.7 * cm_self.limb_velocities[limb_name] + 0.3 * angular_accel
                    
                    neutral_diff = base_angle - cm_self.limb_angles[limb_name]
                    neutral_diff = np.arctan2(np.sin(neutral_diff), np.cos(neutral_diff))
                    cm_self.limb_velocities[limb_name] += 0.01 * neutral_diff
                    cm_self.limb_velocities[limb_name] *= 0.95
                    cm_self.limb_angles[limb_name] += cm_self.limb_velocities[limb_name] * dt * 10
                
                for name in cm_self.limb_torques:
                    cm_self.limb_torques[name] *= 0.75
            
            def compute_skin(cm_self):
                min_dist = np.ones((Ny, Nx)) * 100.0
                
                for name, info in creature.basins.items():
                    bx, by = info['pos']
                    r = info['radius']
                    dist = np.sqrt((X - bx)**2 + (Y - by)**2) - r
                    min_dist = np.minimum(min_dist, dist)
                
                for limb_name in creature.limb_defs:
                    origin_name, _ = creature.limb_defs[limb_name]
                    p1 = creature.basins[origin_name]['pos']
                    p2 = cm_self.get_limb_tip(limb_name)
                    x1, y1 = p1
                    x2, y2 = p2
                    dx_c, dy_c = x2 - x1, y2 - y1
                    length = np.sqrt(dx_c**2 + dy_c**2)
                    if length > 0.01:
                        ux, uy = dx_c / length, dy_c / length
                        proj = np.clip((X - x1)*ux + (Y - y1)*uy, 0, length)
                        dist = np.sqrt((X - (x1 + proj*ux))**2 + (Y - (y1 + proj*uy))**2) - 0.45
                        min_dist = np.minimum(min_dist, dist)
                
                return cm_self.skin_stiffness * (1 - np.exp(-np.maximum(0, min_dist) / cm_self.skin_thickness))
        
        return CreatureMorphology()
    
    def _create_torus(self) -> TorusBrain:
        torus = TorusBrain()
        torus.g = self.species.torus_g_base
        return torus
    
    def _create_body(self) -> BodyField:
        body = BodyField(self.morph)
        body.g = self.species.body_g_base
        body.psi *= self.species.body_scale
        body.energy = float(np.sum(np.abs(body.psi)**2))
        body.E_target = body.energy
        return body
    
    def _create_limb(self, name: str) -> LimbField:
        limb = LimbField(name)
        limb.g = self.species.limb_g_base
        return limb
    
    def _create_metabolism(self) -> Metabolism:
        vigor = 1.0 + 0.1 * self.generation
        
        # Get species-specific refractory (biological recovery time)
        repro_refractory = getattr(self.species, 'reproduction_refractory', 1000)
        
        metab = Metabolism(
            inherited_vigor=min(vigor, 1.5),
            energy_efficiency=self.species.energy_efficiency,
            metabolism_rate=self.species.metabolism_rate,
            reproduction_refractory=repro_refractory
        )
        metab.max_age = int(self.species.max_age_base * np.random.uniform(0.8, 1.2))
        return metab
    
    def get_body_I(self) -> np.ndarray:
        """Get body field intensity."""
        return np.abs(self.body.psi)**2
    
    def get_body_center_world(self) -> np.ndarray:
        """Get body center in world coordinates."""
        return self.pos
    
    def get_effective_speed(self, season=None) -> float:
        """Speed with age, hibernation, digestion, season modifiers."""
        from ..world.seasons import get_age_speed_modifier
        
        base = self.effective_speed
        age_mod = get_age_speed_modifier(self.metabolism.age, self.metabolism.max_age)
        
        if self.is_hibernating:
            return 0.0
        
        digest_mod = DIGESTION_SPEED_PENALTY if self.is_digesting else 1.0
        season_mod = 1.0 / season.movement_cost if season else 1.0
        
        return base * age_mod * digest_mod * season_mod
    
    # =========================================================================
    # PREDATION
    # =========================================================================
    
    def can_hunt(self, other: 'Creature') -> bool:
        """Check if this creature can hunt another."""
        if self.species.diet == 'herbivore':
            return False
        if other.species.name not in self.species.prey_species:
            return False
        if not other.alive:
            return False
        return True
    
    def find_prey(self, creatures: List['Creature'], hunt_radius: float = 12.0) -> Optional['Creature']:
        """Find nearest valid prey within hunt radius."""
        if self.species.diet == 'herbivore':
            return None
        
        best_prey = None
        best_score = -1
        
        for other in creatures:
            if not self.can_hunt(other):
                continue
            
            dist = np.linalg.norm(other.pos - self.pos)
            if dist > hunt_radius or dist < 0.5:
                continue
            
            # Score prey by: proximity, weakness, smallness
            proximity_score = (hunt_radius - dist) / hunt_radius
            weakness_score = other.metabolism.hunger
            size_score = 1.0 - (other.species.body_scale / 1.5)
            
            if self.species.diet == 'omnivore':
                score = proximity_score * 0.3 + weakness_score * 0.5 + size_score * 0.2
                if other.species.name == 'Herbie' and other.metabolism.hunger < 0.5:
                    score *= 0.2
            else:
                score = proximity_score * 0.5 + weakness_score * 0.3 + size_score * 0.2
            
            if score > best_score:
                best_score = score
                best_prey = other
        
        return best_prey
    
    def attack_prey(self, prey: 'Creature', deferred: bool = False) -> float:
        """
        Attack prey, return energy gained.
        
        If deferred=True, record the attack for later application (multicore safe).
        If deferred=False, apply damage immediately (sequential mode).
        """
        if not prey.alive or self.hunt_cooldown > 0:
            return 0.0
        
        dist = np.linalg.norm(prey.pos - self.pos)
        if dist > 3.0:
            return 0.0
        
        # Calculate damage
        damage = self.species.hunt_damage
        
        # Prey resistance: body scale + hunger weakness
        base_resistance = prey.species.body_scale * (1.0 - prey.metabolism.hunger * 0.5)
        
        # Defense bonus from prey (family proximity, learned defense, tools, etc.)
        defense_bonus = getattr(prey, 'defense_bonus', 0.0)
        
        # Total resistance with defense
        resistance = base_resistance * (1.0 + defense_bonus)
        
        effective_damage = damage / max(0.1, resistance)  # Prevent division issues
        
        # If deferred mode, record and return (don't apply yet)
        if deferred:
            if not hasattr(self, '_pending_attacks'):
                self._pending_attacks = []
            self._pending_attacks.append({
                'prey_id': prey.creature_id,
                'damage': effective_damage,
                'attacker_name': self.species.name
            })
            # Still set cooldown and record intent
            self.hunt_cooldown = 15
            energy_gained = effective_damage * 15.0 * self.species.energy_efficiency
            self.meat_consumed += energy_gained
            return energy_gained
        
        # Immediate mode - apply damage now
        # First check for counter-attack from prey with weapons
        counter_damage = 0.0
        if hasattr(prey, 'hands'):
            held_left = prey.hands.left.held_object if hasattr(prey.hands.left, 'held_object') else None
            held_right = prey.hands.right.held_object if hasattr(prey.hands.right, 'held_object') else None
            
            has_weapon = (held_left is not None) or (held_right is not None)
            
            if has_weapon:
                grip = max(
                    getattr(prey.hands.left, 'grip_strength', 0.5),
                    getattr(prey.hands.right, 'grip_strength', 0.5)
                )
                tool_mass = 0.0
                if held_left and hasattr(held_left, 'size'):
                    tool_mass = max(tool_mass, held_left.size)
                if held_right and hasattr(held_right, 'size'):
                    tool_mass = max(tool_mass, held_right.size)
                
                desperation = 0.5 + prey.metabolism.hunger
                counter_damage = grip * (0.3 + tool_mass * 0.2) * desperation
                
                if counter_damage > 0.05:
                    self.metabolism.hunger = min(1.0, self.metabolism.hunger + counter_damage * 0.5)
                    self.body.energy *= (1.0 - counter_damage * 0.2)
                    
                    prey_name_counter = prey.mating_state.name if hasattr(prey, 'mating_state') and prey.mating_state else prey.species.name
                    from ..events.console_log import console_log
                    console_log().log(f"[DEFENSE] {prey_name_counter} struck back at {self.species.name}! (dmg={counter_damage:.2f})")
        
        prey.metabolism.hunger = min(1.0, prey.metabolism.hunger + effective_damage)
        prey.body.energy *= (1.0 - effective_damage * 0.3)
        
        if 'env_pain' in prey.afferent:
            prey.afferent['env_pain'].nucleate(effective_damage * 3.0, 0.0)
        
        flee_dir = prey.pos - self.pos
        flee_dir = flee_dir / (np.linalg.norm(flee_dir) + 0.1)
        prey.vel += flee_dir * effective_damage * 2.0
        
        energy_gained = effective_damage * 15.0 * self.species.energy_efficiency
        self.meat_consumed += energy_gained
        self.hunt_cooldown = 15
        
        # Log attack
        prey_name = prey.mating_state.name if hasattr(prey, 'mating_state') and prey.mating_state else prey.species.name
        from ..events.console_log import console_log
        console_log().log(f"[{self.species.name}]  Attacking {prey_name}! (dmg={effective_damage:.2f})")
        
        if prey.metabolism.hunger >= 0.99 or prey.body.energy < 5:
            prey.alive = False
            prey.metabolism.is_dead = True
            prey.metabolism.cause_of_death = f"predation:{self.species.name}"
            energy_gained += prey.body.energy * 0.5
            
            # Log kill
            console_log().log(f"[{self.species.name}]  KILLED {prey_name}!", force=True)
            
            event_log().log_predation(
                step=self.step_count,
                predator=self.species.name,
                prey=prey.species.name,
                prey_name=prey_name
            )
            
            # Record in world history
            from ..events.world_history import world_history
            world_history().record_predation(
                step=self.step_count,
                predator=self.species.name,
                prey_name=prey_name,
                prey_species=prey.species.name
            )
            
            # START DIGESTION - predator needs to rest after eating!
            from .traits import DIGESTION_DURATION
            if self.species.name in DIGESTION_DURATION:
                self.is_digesting = True
                self.digestion_steps_remaining = DIGESTION_DURATION[self.species.name]
                self.digestion_energy_pending = energy_gained * 0.5  # Half energy delivered slowly
                energy_gained *= 0.5  # Get half immediately
                console_log().log(f"[{self.species.name}]  Settling down to digest...")
        
        return energy_gained
    
    def get_pending_attacks(self) -> List[dict]:
        """Get and clear pending attacks (for multicore mode)."""
        attacks = getattr(self, '_pending_attacks', [])
        self._pending_attacks = []
        return attacks
    
    def receive_damage(self, damage: float, attacker_name: str, attacker: 'Creature' = None):
        """
        Receive damage from an attack (used in multicore mode).
        Called after parallel phase to apply deferred damage.
        
        Returns counter-attack damage dealt (if any).
        """
        counter_damage = 0.0
        
        # ACTIVE DEFENSE: If this creature has hands and is holding something, fight back!
        if hasattr(self, 'hands') and attacker is not None:
            # Check if holding a tool/weapon
            held_left = self.hands.left.held_object if hasattr(self.hands.left, 'held_object') else None
            held_right = self.hands.right.held_object if hasattr(self.hands.right, 'held_object') else None
            
            has_weapon = (held_left is not None) or (held_right is not None)
            
            if has_weapon:
                # Counter-attack! Damage based on grip strength and tool mass
                grip = max(
                    getattr(self.hands.left, 'grip_strength', 0.5),
                    getattr(self.hands.right, 'grip_strength', 0.5)
                )
                tool_mass = 0.0
                if held_left and hasattr(held_left, 'size'):
                    tool_mass = max(tool_mass, held_left.size)
                if held_right and hasattr(held_right, 'size'):
                    tool_mass = max(tool_mass, held_right.size)
                
                # Counter damage: grip strength * tool effectiveness * desperation
                desperation = 0.5 + self.metabolism.hunger  # Fight harder when desperate
                counter_damage = grip * (0.3 + tool_mass * 0.2) * desperation
                
                # Apply counter damage to attacker
                if attacker.alive and counter_damage > 0.05:
                    attacker.metabolism.hunger = min(1.0, attacker.metabolism.hunger + counter_damage * 0.5)
                    if hasattr(attacker, 'body'):
                        attacker.body.energy *= (1.0 - counter_damage * 0.2)
                    
                    my_name = self.mating_state.name if hasattr(self, 'mating_state') and self.mating_state else self.species.name
                    from ..events.console_log import console_log
                    console_log().log(f"[DEFENSE] {my_name} struck back at {attacker_name}! (dmg={counter_damage:.2f})")
        
        # Apply incoming damage
        self.metabolism.hunger = min(1.0, self.metabolism.hunger + damage)
        self.body.energy *= (1.0 - damage * 0.3)
        
        if 'env_pain' in self.afferent:
            self.afferent['env_pain'].nucleate(damage * 3.0, 0.0)
        
        # Check for death
        if self.metabolism.hunger >= 0.99 or self.body.energy < 5:
            self.alive = False
            self.metabolism.is_dead = True
            self.metabolism.cause_of_death = f"predation:{attacker_name}"
            
            # Log kill
            prey_name = self.mating_state.name if hasattr(self, 'mating_state') and self.mating_state else self.species.name
            from ..events.console_log import console_log
            console_log().log(f"[{attacker_name}]  KILLED {prey_name}!", force=True)
            
            event_log().log_predation(
                step=self.step_count,
                predator=attacker_name,
                prey=self.species.name,
                prey_name=prey_name
            )
        
        return counter_damage
    
    def sense_nearby_creatures(self, creatures: List['Creature'], sense_radius: float = 10.0):
        """Generate afferent signals from nearby creatures."""
        for other in creatures:
            if other.creature_id == self.creature_id or not other.alive:
                continue
            
            dist = np.linalg.norm(other.pos - self.pos)
            if dist > sense_radius or dist < 0.1:
                continue
            
            proximity = (sense_radius - dist) / sense_radius
            
            # Predator sensing prey
            if self.can_hunt(other) and proximity > 0.3:
                self.afferent['env_reward'].nucleate(proximity * 0.5, 0.3)
                angle = np.arctan2(other.pos[1] - self.pos[1], other.pos[0] - self.pos[0])
                self.torus.inject_reward(angle, proximity * 0.3)
            
            # Prey sensing predator
            elif other.can_hunt(self) and proximity > 0.2:
                fear_intensity = proximity * other.species.hunt_damage * 2.0
                self.afferent['env_pain'].nucleate(fear_intensity, 0.0)
                angle = np.arctan2(other.pos[1] - self.pos[1], other.pos[0] - self.pos[0])
                self.torus.inject_aversion(angle, fear_intensity * 0.5)
            
            # Same species - social
            elif other.species.name == self.species.name:
                if proximity > 0.3:
                    self.afferent['proprioception'].nucleate(proximity * 0.3, 0.5)
            
            # Different species - caution
            else:
                size_ratio = other.species.body_scale / self.species.body_scale
                if size_ratio > 1.2:
                    self.afferent['env_pain'].nucleate(proximity * 0.15 * size_ratio, 0.3)

    # =========================================================================
    # MAIN UPDATE LOOP
    # =========================================================================
    
    def step(self, all_creatures: List['Creature'], audio_amp: float = 0.0,
             silence_frames: int = 100, audio_system=None, deferred_attacks: bool = False):
        """
        Advance creature by one timestep.
        
        Args:
            all_creatures: List of all creatures for sensing/hunting
            audio_amp: Current audio amplitude
            silence_frames: Frames of silence (for dreaming)
            audio_system: Audio system for sounds
            deferred_attacks: If True, record attacks instead of applying (multicore safe)
        """
        if not self.alive:
            return 'dead'
        
        self.step_count += 1
        self.hunt_cooldown = max(0, self.hunt_cooldown - 1)
        self.damage_taken_this_step = 0.0
        self.lifetime_tracker.update(self, self.step_count)
        
        # === DIGESTION PROCESSING ===
        if self.is_digesting:
            self.digestion_steps_remaining -= 1
            # Slowly absorb energy while digesting
            if self.digestion_energy_pending > 0:
                energy_chunk = self.digestion_energy_pending / max(1, self.digestion_steps_remaining + 1)
                self.body.energy += energy_chunk
                self.digestion_energy_pending -= energy_chunk
                self.metabolism.hunger = max(0, self.metabolism.hunger - 0.002)  # Slowly satisfy hunger
            
            if self.digestion_steps_remaining <= 0:
                self.is_digesting = False
                self.digestion_energy_pending = 0
                from ..events.console_log import console_log
                console_log().log(f"[{self.species.name}]  Finished digesting, ready to hunt again")
        
        # Dream state
        if silence_frames < 80:
            self.dream_depth = max(0.0, self.dream_depth - 0.018)
        else:
            self.dream_depth = 0.94 * self.dream_depth + 0.06 * np.clip((silence_frames - 80) / 180, 0, 1)
        self.is_dreaming = self.dream_depth > 0.15
        
        # Sense other creatures
        self.sense_nearby_creatures(all_creatures)
        
        # === HUNTING BEHAVIOR ===
        hunt_energy = 0.0
        # Don't hunt while digesting!
        if self.species.diet != 'herbivore' and not self.is_dreaming and not self.is_digesting:
            # Larger hunt radius for better prey detection
            hunt_radius = 20.0 if self.species.name == 'Apex' else 12.0
            prey = self.find_prey(all_creatures, hunt_radius=hunt_radius)
            if prey is not None:
                self.current_prey_target = prey.creature_id
                direction = prey.pos - self.pos
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    direction = direction / dist
                    # More aggressive pursuit when hungry
                    pursuit_strength = 0.6 + 0.6 * self.metabolism.hunger
                    self.vel += direction * pursuit_strength * self.species.speed_factor * 0.4
                    self.torus.inject_reward(np.arctan2(direction[1], direction[0]), 0.3)
                
                # Attack at close range
                if dist < 3.5:
                    hunt_energy = self.attack_prey(prey, deferred=deferred_attacks)
            else:
                self.current_prey_target = None
        
        # === AFFERENT CHANNELS ===
        for ch in self.afferent.values():
            ch.evolve()
            for a in ch.get_arrivals():
                self.torus.receive_afferent(ch.name, [a])
        
        # === TORUS ===
        g_target = self.species.torus_g_base
        g_target += 0.6 * audio_amp
        g_target -= 0.5 * self.metabolism.hunger
        g_target -= 1.2 * self.dream_depth
        self.torus.g = G_SMOOTH * self.torus.g + (1 - G_SMOOTH) * g_target
        self.torus.evolve(audio_amp, self.metabolism.hunger, self.dream_depth)
        
        torus_arousal = self.torus.get_arousal()
        torus_phase = float(np.angle(self.torus.psi[np.argmax(np.abs(self.torus.psi)**2)]))
        torus_bias = self.torus.get_directional_bias()
        
        # === EFFERENT FIRING ===
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
        
        # === SKELETON ===
        body_I = self.get_body_I()
        self.skeleton.step(body_I)
        
        # === POTENTIAL ===
        V_total = self._compute_potential()
        
        # === BODY FIELD ===
        g_base = self.species.body_g_base
        g_target = g_base + 0.5 * torus_arousal + self.metabolism.get_g_modifier() + 0.4 * audio_amp - 0.4 * self.dream_depth
        self.body.g = G_SMOOTH * self.body.g + (1 - G_SMOOTH) * g_target
        self.body.evolve(V_total, torus_phase, torus_arousal, self.metabolism.get_g_modifier(),
                         self.dream_depth, audio_amp, hunger=self.metabolism.hunger)
        
        # === LIMBS ===
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
            self.morph.update_limb_angles(self.limbs, torus_bias, self.skeleton.rms,
                                          self.metabolism.hunger, audio_amp)
        
        # === MOVEMENT ===
        body_momentum = self.body.get_momentum()
        
        if np.linalg.norm(torus_bias) > 0.05:
            self.body.inject_momentum(torus_bias, torus_arousal * 0.5)
        if np.linalg.norm(limb_thrust) > 0.02:
            self.body.inject_momentum(limb_thrust, 0.3)
        
        self._update_movement(body_momentum, torus_bias, limb_thrust)
        
        # === WORLD INTERACTION ===
        body_I = self.get_body_I()
        self.total_reward, self.mass_extracted = self.world.process_creature_interactions(
            self, body_I, body_momentum
        )
        
        self.total_reward += hunt_energy
        self.mass_extracted += hunt_energy
        
        # === GRAZING BEHAVIOR ===
        # When getting reward from food, slow down to stay and eat
        # This is the "dwell on food" behavior - don't rush past!
        if self.total_reward > 0.05 and self.metabolism.hunger > 0.1:
            # Eating! Dampen velocity to stay on food
            graze_factor = 0.7  # Slow down by 30%
            self.vel *= graze_factor
        
        # === AFFERENT SENSING ===
        self._do_afferent_sensing(V_total, audio_amp, audio_system)
        
        # === METABOLISM ===
        reward_source = self.world.get_reward_source_for_creature(self)
        vel_mag = np.linalg.norm(self.vel)
        self.metabolism.update(self.total_reward, self.mass_extracted, vel_mag, reward_source)
        
        # === REWARD AFFERENT - critical for learning! ===
        # Eating reward reaches torus brain for associative learning
        if self.total_reward > 0.01 and 'env_reward' in self.afferent:
            reward_signal = min(2.0, self.total_reward * 1.5)
            self.afferent['env_reward'].nucleate(reward_signal, 0.5)
        
        # Defecation
        if self.metabolism.defecation_pending:
            self.world.drop_nutrient(self.pos.copy(), self.metabolism.last_defecation_amount)
            self.lifetime_tracker.log_defecation(self.metabolism.last_defecation_amount, self.step_count)
        
        # Hunger seeking - but not if already eating!
        if self.species.diet == 'herbivore' and self.total_reward < 0.05:
            self._hunger_seeking()
        
        self.display_vel = 0.8 * self.display_vel + 0.2 * self.vel
        
        # === SONIFICATION ===
        # Herbies always sonify at full volume, others only 10% of the time and quieter
        if audio_system is not None and hasattr(audio_system, 'update_sonification'):
            if self.species.name == "Herbie":
                audio_system.update_sonification(
                    self.body.psi, self.torus.psi, self.limbs, self.dream_depth
                )
            elif np.random.random() < 0.1:
                # Other species: temporarily reduce volume
                audio_system.master_amp = max(0.05, audio_system.master_amp * 0.3)
                audio_system.update_sonification(
                    self.body.psi, self.torus.psi, self.limbs, self.dream_depth
                )
                audio_system.master_amp = min(0.4, audio_system.master_amp / 0.3)
        
        # === DEATH/REPRODUCTION ===
        if self.metabolism.is_dead:
            self.alive = False
            print(f"[{self.species.name}] Gen{self.generation} {self.creature_id[:8]} died: {self.metabolism.cause_of_death}")
            return 'dead'
        
        if self.metabolism.ready_to_reproduce:
            self.metabolism.ready_to_reproduce = False
            # Reproduction costs energy - based on current state, not arbitrary threshold
            self.metabolism.total_consumed *= 0.5  # Reset consumption tracking
            self.metabolism.hunger = min(1.0, self.metabolism.hunger + 0.3)  # Get hungrier
            self.body.energy *= 0.6  # Significant energy cost
            self.body.psi *= np.sqrt(0.6)
            self.offspring_count += 1
            self.lifetime_tracker.log_reproduction(self.step_count)
            print(f"[{self.species.name}] Gen{self.generation} {self.creature_id[:8]} reproduced!")
            return 'reproduce'
        
        return 'alive'
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _compute_potential(self) -> np.ndarray:
        """Compute total potential field."""
        V = np.ones((Ny, Nx)) * 9.0
        
        for name, info in self.basins.items():
            bx, by = info['pos']
            V -= info['depth'] * np.exp(-((X - bx)**2 + (Y - by)**2) / info['radius']**2)
        
        if 'core' in self.basins:
            for limb_name in self.limb_defs:
                if limb_name in self.basins:
                    p1 = self.basins['core']['pos']
                    p2 = self.basins[limb_name]['pos']
                    x1, y1 = p1
                    x2, y2 = p2
                    dx_c, dy_c = x2 - x1, y2 - y1
                    length = np.sqrt(dx_c**2 + dy_c**2)
                    if length > 0.01:
                        ux, uy = dx_c / length, dy_c / length
                        proj = np.clip((X - x1)*ux + (Y - y1)*uy, 0, length)
                        dist = np.sqrt((X - (x1 + proj*ux))**2 + (Y - (y1 + proj*uy))**2)
                        V -= 3.5 * np.exp(-dist**2 / 0.35**2)
        
        V = gaussian_filter(V, sigma=0.6)
        V += self.morph.compute_skin()
        V += self.skeleton.get_potential()
        V += self.world.get_object_potential_for_creature(self)
        V += self.world.get_boundary_potential(self.pos)
        
        return V
    
    def _update_movement(self, body_momentum: np.ndarray, torus_bias: np.ndarray, limb_thrust: np.ndarray):
        """Update creature position based on forces."""
        speed_factor = self.get_effective_speed()
        hunger_boost = 1.0 + self.metabolism.hunger * 0.5
        
        force = body_momentum * 5.0 * hunger_boost * speed_factor
        force += torus_bias * 1.5 * hunger_boost * speed_factor
        force += limb_thrust * 2.5 * speed_factor
        
        # Guard against NaN forces
        if np.any(np.isnan(force)):
            force = np.zeros(2)
        
        speed = np.linalg.norm(self.vel)
        if speed < 0.5 or self.metabolism.hunger > 0.5:
            force += np.random.randn(2) * (0.4 + 0.3 * self.metabolism.hunger)
        
        boundary_margin = WORLD_L/2 - 5
        for i in range(2):
            if self.pos[i] < -boundary_margin:
                force[i] += 2.0 * (-boundary_margin - self.pos[i])
            elif self.pos[i] > boundary_margin:
                force[i] -= 2.0 * (self.pos[i] - boundary_margin)
        
        self.vel += force * dt * 20
        self.vel *= 0.96
        
        # NaN protection for velocity
        if np.any(np.isnan(self.vel)):
            self.vel = np.zeros(2)
        
        speed = np.linalg.norm(self.vel)
        max_speed = (2.5 + self.metabolism.hunger * 0.5) * speed_factor
        if speed > max_speed and speed > 0:
            self.vel = self.vel / speed * max_speed
        
        self.pos += self.vel * dt
        self.pos = np.clip(self.pos, -WORLD_L/2 + 2, WORLD_L/2 - 2)
        
        # Final NaN guard on position
        if np.any(np.isnan(self.pos)):
            self.pos = np.zeros(2)
    
    def _do_afferent_sensing(self, V_total: np.ndarray, audio_amp: float, audio_system=None):
        """Full multi-modal afferent sensing."""
        sensitivity = self.effective_afferent
        
        # Terrain/proprioception
        V_local = V_total[Ny//2-5:Ny//2+5, Nx//2-5:Nx//2+5]
        grad_x = np.mean(np.gradient(V_local, axis=1))
        grad_y = np.mean(np.gradient(V_local, axis=0))
        terrain_intensity = np.sqrt(grad_x**2 + grad_y**2)
        
        if terrain_intensity > 0.5:
            self.afferent['proprioception'].nucleate(terrain_intensity * 0.3 * sensitivity, 0.2)
        
        # Object contact
        for i, obj in enumerate(self.world.objects):
            if not obj.alive:
                continue
            rel = obj.pos - self.pos
            dist = np.linalg.norm(rel)
            
            if dist < BODY_L * 0.8 and obj.contact > 0.05:
                touch_idx = i % min(3, max(1, self.species.num_limbs if self.species.num_limbs > 0 else 1))
                self.afferent[f'touch_{touch_idx}'].nucleate(obj.contact * 1.2 * sensitivity, 0.0)
        
        # Audio-driven afferents
        if audio_amp > 0.008:
            self.afferent['env_reward'].nucleate(audio_amp * 0.5 * sensitivity, 0.1)
    
    def _hunger_seeking(self):
        """Move toward food when hungry - EM field gradient following.
        
        This provides additional directed movement on top of the body field
        wave dynamics. The body field evolves toward food potentials naturally,
        but this gives an extra push for reliability.
        """
        if self.metabolism.hunger > 0.15:  # Start seeking earlier
            nearest = self.world.get_nearest_food(self.pos)
            if nearest is not None:
                direction = nearest.pos - self.pos
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    direction = direction / dist
                    
                    # EM field coupling - creatures sense food EM signature
                    base_strength = 0.3 + 1.0 * self.metabolism.hunger  # 0.3 to 1.3
                    proximity_boost = np.exp(-(dist / 15.0)**2)  # Stronger when closer
                    long_range = 0.2 * np.exp(-dist / 50.0)  # Weak long-range attraction
                    
                    strength = base_strength * (proximity_boost + long_range)
                    
                    # Desperate hunger = stronger seeking
                    if self.metabolism.hunger > 0.7:
                        strength *= 1.5
                    if self.metabolism.hunger > 0.9:
                        strength *= 1.3  # Total ~2x when starving
                    
                    # Primary: inject into body field (wave dynamics)
                    self.body.inject_momentum(direction, strength * 0.8)
                    
                    # Secondary: small direct velocity nudge (backup)
                    self.vel += direction * strength * 0.15 * self.species.speed_factor
    
    def to_dict(self) -> dict:
        """Serialize complete creature state for persistence."""
        # Serialize all components
        torus_data = self.torus.to_dict() if hasattr(self.torus, 'to_dict') else {}
        body_data = self.body.to_dict() if hasattr(self.body, 'to_dict') else {}
        skeleton_data = self.skeleton.to_dict() if hasattr(self.skeleton, 'to_dict') else {}
        metabolism_data = self.metabolism.to_dict() if hasattr(self.metabolism, 'to_dict') else {}
        morph_data = self.morph.to_dict() if hasattr(self.morph, 'to_dict') else {}
        
        # Serialize limbs
        limbs_data = {}
        for name, limb in self.limbs.items():
            if hasattr(limb, 'to_dict'):
                limbs_data[name] = limb.to_dict()
        
        # Serialize neural channels
        afferent_data = {}
        for name, ch in self.afferent.items():
            if hasattr(ch, 'to_dict'):
                afferent_data[name] = ch.to_dict()
        
        efferent_data = {}
        for name, ch in self.efferent.items():
            if hasattr(ch, 'to_dict'):
                efferent_data[name] = ch.to_dict()
        
        return {
            # Identity
            'species_name': self.species.name,
            'creature_id': self.creature_id,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'lineage_depth': self.lineage_depth,
            
            # Position/motion
            'pos': self.pos.tolist(),
            'vel': self.vel.tolist(),
            'display_vel': self.display_vel.tolist(),
            
            # Core components (wavefunctions!)
            'torus': torus_data,
            'body': body_data,
            'skeleton': skeleton_data,
            'metabolism': metabolism_data,
            'morph': morph_data,
            'limbs': limbs_data,
            
            # Neural channels
            'afferent': afferent_data,
            'efferent': efferent_data,
            
            # Learning/memory
            'learned_potential': self.learned_potential,
            'traits': self.traits.to_dict() if self.traits else {},
            
            # State counters
            'step_count': self.step_count,
            'offspring_count': self.offspring_count,
            'total_reward': self.total_reward,
            'mass_extracted': self.mass_extracted,
            
            # Behavioral state
            'dream_depth': self.dream_depth,
            'is_dreaming': self.is_dreaming,
            'is_hibernating': self.is_hibernating,
            'hibernation_steps': self.hibernation_steps,
            
            # Combat/hunting state
            'hunt_cooldown': self.hunt_cooldown,
            'meat_consumed': self.meat_consumed,
            'is_defending': self.is_defending,
            'defense_successes': self.defense_successes,
            'defense_attempts': self.defense_attempts,
            'learned_defense_bonus': self.learned_defense_bonus,
            
            # Digestion state
            'is_digesting': self.is_digesting,
            'digestion_steps_remaining': self.digestion_steps_remaining,
            'digestion_energy_pending': self.digestion_energy_pending,
        }
    
    @classmethod
    def from_dict(cls, data: dict, world: 'MultiWorld') -> 'Creature':
        """
        Deserialize creature from saved state.
        
        Args:
            data: Dict from to_dict()
            world: World to place creature in
            
        Returns:
            Restored Creature instance
        """
        from .species import get_species_by_name
        from .traits import MutatedTraits
        from ..brain.torus import TorusBrain
        from ..brain.kdv_channels import KdVChannel
        from ..body.body_field import BodyField
        from ..body.skeleton import PiezoSkeleton
        from ..body.metabolism import Metabolism
        from ..body.morphology import Morphology
        from ..body.limbs import LimbField
        
        # Get species
        species = get_species_by_name(data['species_name'])
        if species is None:
            raise ValueError(f"Unknown species: {data['species_name']}")
        
        # Restore traits first (needed for some component creation)
        traits = MutatedTraits.from_dict(data.get('traits', {})) if data.get('traits') else MutatedTraits()
        
        # Create creature shell (bypassing __init__ to restore state)
        creature = cls.__new__(cls)
        
        # Identity
        creature.species = species
        creature.world = world
        creature.creature_id = data['creature_id']
        creature.generation = data['generation']
        creature.parent_id = data.get('parent_id', '')
        creature.lineage_depth = data.get('lineage_depth', 0)
        
        # Position/motion
        creature.pos = np.array(data['pos'])
        creature.vel = np.array(data['vel'])
        creature.display_vel = np.array(data.get('display_vel', [0.0, 0.0]))
        
        # Species config
        creature.basins = get_species_basins(species)
        creature.limb_defs = get_species_limb_defs(species)
        
        # Traits
        creature.traits = traits
        creature.effective_speed = traits.get('speed_factor', species.speed_factor)
        creature.effective_efficiency = traits.get('energy_efficiency', species.energy_efficiency)
        creature.effective_afferent = traits.get('afferent_sensitivity', species.afferent_sensitivity)
        creature.effective_efferent = traits.get('efferent_strength', species.efferent_strength)
        creature.effective_metabolism = traits.get('metabolism_rate', species.metabolism_rate)
        
        # Restore morphology - always use species-specific creation, then apply saved state
        creature.morph = creature._create_morphology_static(species, creature.basins, creature.limb_defs)
        if data.get('morph'):
            morph_data = data['morph']
            if morph_data.get('limb_angles'):
                for k, v in morph_data['limb_angles'].items():
                    if k in creature.morph.limb_angles:
                        creature.morph.limb_angles[k] = v
            if morph_data.get('limb_velocities'):
                for k, v in morph_data['limb_velocities'].items():
                    if k in creature.morph.limb_velocities:
                        creature.morph.limb_velocities[k] = v
            if morph_data.get('limb_torques'):
                for k, v in morph_data['limb_torques'].items():
                    if k in creature.morph.limb_torques:
                        creature.morph.limb_torques[k] = v
        
        # Restore torus brain
        if data.get('torus'):
            creature.torus = TorusBrain.from_dict(data['torus'])
        else:
            creature.torus = TorusBrain()
        
        # Restore body field
        if data.get('body'):
            creature.body = BodyField.from_dict(data['body'], creature.morph)
        else:
            creature.body = BodyField(creature.morph)
        
        # Restore skeleton
        if data.get('skeleton'):
            creature.skeleton = PiezoSkeleton.from_dict(data['skeleton'])
        else:
            creature.skeleton = PiezoSkeleton()
        
        # Restore metabolism
        if data.get('metabolism'):
            creature.metabolism = Metabolism.from_dict(data['metabolism'])
        else:
            creature.metabolism = Metabolism()
        
        # Restore limbs
        creature.limbs = {}
        limbs_data = data.get('limbs', {})
        for name in creature.limb_defs:
            if name in limbs_data:
                creature.limbs[name] = LimbField.from_dict(limbs_data[name])
            else:
                creature.limbs[name] = LimbField(name)
        
        # Restore neural channels
        creature.afferent = {}
        afferent_data = data.get('afferent', {})
        from ..core.constants import AFFERENT_CHANNELS
        for name in AFFERENT_CHANNELS:
            if name in afferent_data:
                creature.afferent[name] = KdVChannel.from_dict(afferent_data[name])
            else:
                creature.afferent[name] = KdVChannel(name, 'afferent')
        if 'threat' not in creature.afferent:
            creature.afferent['threat'] = KdVChannel('threat', 'afferent')
        
        creature.efferent = {}
        efferent_data = data.get('efferent', {})
        for i in range(min(species.num_limbs, 4)):
            name = f'motor_{i}'
            if name in efferent_data:
                creature.efferent[name] = KdVChannel.from_dict(efferent_data[name])
            else:
                creature.efferent[name] = KdVChannel(name, 'efferent')
        if not creature.efferent:
            creature.efferent['motor_body'] = KdVChannel('motor_body', 'efferent')
        
        # Learning/memory
        creature.learned_potential = data.get('learned_potential', {})
        
        # State counters
        creature.step_count = data.get('step_count', 0)
        creature.offspring_count = data.get('offspring_count', 0)
        creature.total_reward = data.get('total_reward', 0.0)
        creature.mass_extracted = data.get('mass_extracted', 0.0)
        
        # Behavioral state
        creature.dream_depth = data.get('dream_depth', 0.0)
        creature.is_dreaming = data.get('is_dreaming', False)
        creature.is_hibernating = data.get('is_hibernating', False)
        creature.hibernation_steps = data.get('hibernation_steps', 0)
        creature.alive = True
        
        # Combat/hunting state
        creature.current_prey_target = None
        creature.hunt_cooldown = data.get('hunt_cooldown', 0)
        creature.meat_consumed = data.get('meat_consumed', 0.0)
        creature.is_defending = data.get('is_defending', False)
        creature.defense_successes = data.get('defense_successes', 0)
        creature.defense_attempts = data.get('defense_attempts', 0)
        creature.learned_defense_bonus = data.get('learned_defense_bonus', 0.0)
        creature.damage_taken_this_step = 0.0
        
        # Digestion state
        creature.is_digesting = data.get('is_digesting', False)
        creature.digestion_steps_remaining = data.get('digestion_steps_remaining', 0)
        creature.digestion_energy_pending = data.get('digestion_energy_pending', 0.0)
        
        # For meat chunk spawning on death
        creature._pending_meat_chunks = None
        
        # Tracking (fresh tracker for restored creature)
        from ..statistics import LifetimeTracker
        creature.lifetime_tracker = LifetimeTracker(
            creature.creature_id, creature.generation, creature.parent_id, creature.lineage_depth
        )
        
        return creature
    
    @staticmethod
    def _create_morphology_static(species, basins, limb_defs):
        """Create morphology without self reference (for from_dict)."""
        class CreatureMorphology:
            def __init__(cm_self):
                cm_self.limb_angles = {name: info[1] for name, info in limb_defs.items()}
                cm_self.limb_velocities = {name: 0.0 for name in limb_defs}
                cm_self.limb_torques = {name: 0.0 for name in limb_defs}
                cm_self.skin_stiffness = 7.0 * species.body_scale
                cm_self.skin_thickness = 0.55 * species.body_scale
                
            def get_limb_tip(cm_self, limb_name):
                if limb_name not in basins:
                    return (0, 0)
                base = basins[limb_name]['pos']
                angle = cm_self.limb_angles.get(limb_name, 0)
                length = limb_defs[limb_name][2] if limb_name in limb_defs else 1.0
                return (base[0] + np.cos(angle) * length,
                        base[1] + np.sin(angle) * length)
            
            def compute_skin(cm_self):
                from ..core.constants import Nx, Ny, X, Y, BODY_L
                V_skin = np.zeros((Ny, Nx))
                for name, info in basins.items():
                    bx, by = info['pos']
                    r = np.sqrt((X - bx)**2 + (Y - by)**2)
                    r_basin = info['radius'] * 1.2
                    V_skin += cm_self.skin_stiffness * np.maximum(0, r - r_basin)**2
                return V_skin
            
            def update_limb_angles(cm_self, limbs, torus_bias, skeleton_rms, hunger, audio_amp):
                for limb_name in limb_defs:
                    if limb_name not in limbs:
                        continue
                    limb = limbs[limb_name]
                    target_angle = limb_defs[limb_name][1]
                    bias_angle = np.arctan2(torus_bias[1], torus_bias[0]) if np.linalg.norm(torus_bias) > 0.05 else 0
                    target_angle += 0.2 * bias_angle
                    current = cm_self.limb_angles[limb_name]
                    diff = target_angle - current
                    while diff > np.pi: diff -= 2*np.pi
                    while diff < -np.pi: diff += 2*np.pi
                    torque = 0.08 * diff + 0.02 * limb.pulse_amplitude * np.sin(limb.pulse_position * np.pi)
                    cm_self.limb_torques[limb_name] = torque
                    cm_self.limb_velocities[limb_name] = 0.9 * cm_self.limb_velocities.get(limb_name, 0) + torque
                    cm_self.limb_angles[limb_name] += cm_self.limb_velocities[limb_name] * 0.1
            
            def apply_efferent_torque(cm_self, limb_name, amp):
                if limb_name in cm_self.limb_torques:
                    cm_self.limb_torques[limb_name] += amp * 0.3
            
            def to_dict(cm_self):
                return {
                    'limb_angles': cm_self.limb_angles,
                    'limb_velocities': cm_self.limb_velocities,
                    'limb_torques': cm_self.limb_torques,
                    'skin_stiffness': cm_self.skin_stiffness,
                    'skin_thickness': cm_self.skin_thickness,
                }
        
        return CreatureMorphology()
