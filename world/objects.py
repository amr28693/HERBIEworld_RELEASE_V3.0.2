"""
World Objects - Items that creatures can interact with.

Includes food, barriers, nutrients, meat chunks, and corpses.
"""

from typing import List, Tuple, Optional
import numpy as np

from ..core.constants import (
    WORLD_L, BODY_L, Nx, Ny, X, Y, dx, dt
)


class WorldObject:
    """
    An object in the world that creatures can interact with.
    
    Objects have:
    - Position and velocity (can be pushed)
    - Size and compliance (soft vs hard)
    - Energy (for food objects)
    - Contact detection with creature body fields
    """
    
    def __init__(self, pos, size=1.0, compliance=0.5, mass=1.0, color='green', energy=None):
        """
        Initialize a world object.
        
        Args:
            pos: (x, y) position
            size: Object radius
            compliance: 0=hard barrier, 1=soft/edible
            mass: Affects how easily it's pushed
            color: Display color
            energy: Energy content (None = auto from compliance)
        """
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2)
        self.size = size
        self.initial_size = size
        self.compliance = compliance
        self.mass = mass
        self.color = color
        self.contact = 0.0
        self.reward = 0.0
        
        if energy is None:
            self.energy = 40.0 * compliance if compliance > 0.5 else 0.0
        else:
            self.energy = energy
        self.max_energy = self.energy
        self.alive = True
        
    def compute_contact(self, creature_pos: np.ndarray, body_I: np.ndarray) -> float:
        """
        Compute contact strength with creature body field.
        
        Uses both body field intensity AND distance-based proximity for
        reliable contact detection. The body field alone can have low values
        at the edges where eating actually occurs.
        
        Args:
            creature_pos: Creature's world position
            body_I: Creature's body intensity field
            
        Returns:
            Contact strength (0-1)
        """
        if not self.alive:
            self.contact = 0.0
            return 0.0
        
        # Guard against NaN positions
        if np.any(np.isnan(creature_pos)) or np.any(np.isnan(self.pos)):
            self.contact = 0.0
            return 0.0
        
        rel = self.pos - creature_pos
        dist = np.linalg.norm(rel)
        
        # Distance-based contact (reliable proximity detection)
        # INCREASED reach - creatures can eat from further away
        # This represents their ability to reach/grab food
        reach_radius = 5.0 + self.size  # Increased from 3.0
        if dist < reach_radius * 2:
            # Softer falloff for more reliable eating
            proximity_contact = np.exp(-(dist / reach_radius)**2)
        else:
            proximity_contact = 0.0
        
        # Body field based contact (for fine-grained body awareness)
        body_contact = 0.0
        obj_i = int((rel[0] + BODY_L/2) / dx)
        obj_j = int((rel[1] + BODY_L/2) / dx)
        
        if 0 <= obj_i < Nx and 0 <= obj_j < Ny:
            i_min, i_max = max(0, obj_i - 5), min(Nx, obj_i + 6)
            j_min, j_max = max(0, obj_j - 5), min(Ny, obj_j + 6)
            region = body_I[j_min:j_max, i_min:i_max]
            if region.size > 0:
                mean_val = np.nanmean(region)
                body_contact = float(mean_val) if not np.isnan(mean_val) else 0.0
        
        # Combine: use MAX of proximity and body field contact
        # This ensures eating works even if body field has low edge values
        self.contact = max(proximity_contact, body_contact * 2.0)  # Scale body contact up
        
        return self.contact
        
    def compute_reward(self) -> Tuple[float, float]:
        """
        Compute reward and energy extraction from contact.
        
        Returns:
            (reward, extracted_mass) tuple
        """
        if not self.alive or self.contact < 0.03:  # Moderate threshold
            self.reward = 0.0
            return 0.0, 0.0
            
        if self.compliance > 0.5 and self.energy > 0:
            # Soft/edible object - extract energy
            # V2.5.3: moderate extraction - need sustained contact
            extraction = min(self.contact * 1.5, self.energy)
            self.energy -= extraction
            self.reward = extraction * (self.compliance - 0.25)
            self.size = 0.4 + 0.6 * self.initial_size * (self.energy / (self.max_energy + 1e-6))
            if self.energy < 0.5:
                self.alive = False
            return self.reward, extraction
        else:
            # Hard object - just contact reward (negative for barriers)
            self.reward = self.contact * (self.compliance - 0.35) * 2.0
            return self.reward, 0.0
        
    def apply_push(self, creature_pos: np.ndarray, creature_vel: np.ndarray, 
                   body_momentum: np.ndarray, contact_strength: float):
        """
        Apply push force from creature contact.
        
        Args:
            creature_pos: Creature position
            creature_vel: Creature velocity
            body_momentum: Body field momentum
            contact_strength: Contact strength
        """
        if not self.alive or contact_strength < 0.08:
            return
            
        direction = self.pos - creature_pos
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            return
            
        direction = direction / dist
        push_vec = creature_vel * 0.4 + body_momentum * 1.5
        push_along = np.dot(push_vec, direction)
        
        if push_along > 0:
            force = direction * push_along * contact_strength / self.mass
            force *= self.compliance * 0.5 + 0.2
            self.vel += force * dt * 15
            
    def update(self):
        """Update object physics (velocity decay, boundary check)."""
        if not self.alive:
            return
        
        # NaN protection
        if np.any(np.isnan(self.vel)):
            self.vel = np.zeros(2)
            
        self.vel *= 0.94
        self.pos += self.vel * dt
        
        # Final NaN guard on position
        if np.any(np.isnan(self.pos)):
            self.pos = np.zeros(2)
        
        # Boundary containment
        margin = self.size + 1
        self.pos = np.clip(self.pos, -WORLD_L/2 + margin, WORLD_L/2 - margin)
        
        # Bounce off boundaries
        if abs(self.pos[0]) > WORLD_L/2 - margin - 0.1:
            self.vel[0] *= -0.5
        if abs(self.pos[1]) > WORLD_L/2 - margin - 0.1:
            self.vel[1] *= -0.5
            
    def get_potential(self, creature_pos: np.ndarray) -> np.ndarray:
        """
        Get potential field for this object (attraction/repulsion).
        
        Args:
            creature_pos: Creature position
            
        Returns:
            2D potential field
        """
        if not self.alive:
            return np.zeros((Ny, Nx))
            
        rel = self.pos - creature_pos
        dist = np.sqrt((X - rel[0])**2 + (Y - rel[1])**2)
        
        if self.compliance > 0.5:
            # Attractive (food)
            strength = self.compliance * 0.5 * (1 + self.energy / (self.max_energy + 1e-6))
            return -strength * 2.5 * np.exp(-dist**2 / self.size**2)
        else:
            # Repulsive (barrier)
            barrier_dist = dist - self.size
            return (1 - self.compliance) * 5.0 * np.exp(-np.maximum(0, barrier_dist)**2 / 0.6)
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'pos': self.pos.tolist(),
            'vel': self.vel.tolist(),
            'size': self.size,
            'initial_size': self.initial_size,
            'compliance': self.compliance,
            'mass': self.mass,
            'color': self.color,
            'energy': self.energy,
            'max_energy': self.max_energy,
            'alive': self.alive,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WorldObject':
        """Deserialize from persistence."""
        obj = cls(
            pos=data['pos'],
            size=data.get('initial_size', data.get('size', 1.0)),
            compliance=data.get('compliance', 0.5),
            mass=data.get('mass', 1.0),
            color=data.get('color', 'green'),
            energy=data.get('energy', 0.0),
        )
        obj.vel = np.array(data.get('vel', [0.0, 0.0]))
        obj.size = data.get('size', obj.size)
        obj.max_energy = data.get('max_energy', obj.energy)
        obj.alive = data.get('alive', True)
        return obj


class NutrientPatch:
    """
    Dropped nutrients that can sprout into new food.
    
    When creatures defecate, nutrients are deposited that can
    eventually grow into new food plants.
    """
    
    def __init__(self, pos, nutrients: float):
        """
        Initialize nutrient patch.
        
        Args:
            pos: Position
            nutrients: Nutrient amount (affects resulting plant size)
        """
        self.pos = np.array(pos, dtype=float)
        self.nutrients = nutrients
        self.age = 0
        self.sprouted = False
        self.sprout_time = int(np.random.uniform(300, 600))
        
    def update(self, creature_positions: list = None) -> bool:
        """
        Update nutrient patch.
        
        Creature proximity accelerates growth - their EM field (body heat,
        activity, trampling) helps nutrients break down and sprout faster.
        
        Args:
            creature_positions: List of (pos, energy) tuples for nearby creatures
            
        Returns:
            True if ready to sprout
        """
        self.age += 1
        
        # Creature EM field coupling - proximity accelerates growth
        if creature_positions:
            for cpos, cenergy in creature_positions:
                if np.any(np.isnan(cpos)):
                    continue
                dist = np.linalg.norm(self.pos - cpos)
                if dist < 8.0:  # Within creature's EM influence radius
                    # Closer creatures with more energy boost growth more
                    em_coupling = np.exp(-(dist / 4.0)**2) * (cenergy / 100.0)
                    self.age += em_coupling * 0.5  # Accelerate aging/sprouting
        
        if self.age > self.sprout_time and not self.sprouted:
            self.sprouted = True
            return True
        return False


class MeatChunk(WorldObject):
    """
    A chunk of meat from a killed creature.
    
    Can be picked up, carried, and eaten by any creature.
    Decays over time if not consumed.
    Putrid meat can spread disease.
    """
    
    QUADRANT_NAMES = ['head', 'torso', 'left_limbs', 'right_limbs']
    
    # Decay stages
    FRESH_THRESHOLD = 1500
    PUTRID_THRESHOLD = 800
    
    def __init__(self, pos: np.ndarray, source_species: str, quadrant: str, energy: float, 
                 source_name: str = None, is_putrid: bool = False):
        """
        Initialize meat chunk.
        
        Args:
            pos: Position
            source_species: Species the meat came from
            quadrant: Body part ('head', 'torso', 'left_limbs', 'right_limbs')
            energy: Energy content
            source_name: Name of source creature (for Herbies)
            is_putrid: Whether meat is already putrid (from putrid corpse)
        """
        super().__init__(
            pos=pos,
            size=0.8,
            compliance=0.9,  # Very soft/edible
            mass=1.5,
            color='#8B0000',  # Dark red
            energy=energy
        )
        self.source_species = source_species
        self.source_name = source_name
        self.quadrant = quadrant
        self.decay_timer = 2000  # Decays after ~2000 steps if not eaten
        self.is_meat = True
        
        # Grippable properties - meat is easy to grip, not a weapon
        self.grip_props = type('GripProps', (), {
            'grip_difficulty': 0.2,  # Easy to grip (soft)
            'weight': 1.5,
            'tool_damage': 0.05,  # Negligible as weapon
            'tool_type': 'soft',
            'stackable': False
        })()
        
        # Putrefaction
        self.is_putrid = is_putrid
        if is_putrid:
            self.decay_timer = 800  # Already partially decayed
            self.color = '#4a3030'  # Darker, greenish
        
        self.burial_depth = 0.0  # Set externally
        
    def step(self) -> dict:
        """
        Meat decays over time.
        
        Returns:
            dict with:
            - 'alive': bool
            - 'emit_disease': bool - for putrid meat
        """
        result = {'alive': True, 'emit_disease': False}
        
        # Burial slows decay
        decay_rate = 1.0 if self.burial_depth < 0.5 else 0.2
        self.decay_timer -= decay_rate
        
        if self.decay_timer <= 0:
            self.alive = False
            result['alive'] = False
            return result
        
        # Slowly lose energy as it rots
        if self.decay_timer < self.FRESH_THRESHOLD:
            self.energy *= 0.998
        
        # Putrefaction
        if self.decay_timer < self.PUTRID_THRESHOLD and not self.is_putrid:
            self.is_putrid = True
            self.color = '#4a3030'
        
        # Disease from putrid meat (less likely than corpse)
        if self.is_putrid and self.burial_depth < 0.5:
            if np.random.random() < 0.0005:  # 0.05% per step
                result['emit_disease'] = True
        
        return result
    
    def extract_energy(self, amount: float) -> float:
        """
        Extract energy from meat (feeding).
        
        Args:
            amount: Max energy to extract
            
        Returns:
            Actual energy extracted (reduced if putrid)
        """
        efficiency = 0.5 if self.is_putrid else 0.8
        extracted = min(amount * efficiency, self.energy * 0.2)
        self.energy -= extracted / efficiency
        return extracted
        
    def __repr__(self):
        state = " (putrid)" if self.is_putrid else ""
        if self.source_name:
            return f"MeatChunk({self.source_name}'s {self.quadrant}{state}, E={self.energy:.1f})"
        return f"MeatChunk({self.source_species} {self.quadrant}{state}, E={self.energy:.1f})"


class HerbieCorpse(WorldObject):
    """
    A dead Herbie's body.
    
    Can be:
    1. Gripped and moved (for burial, or to keep away from predators)
    2. Dismembered into meat chunks (by sustained interaction)
    3. Eaten directly (partial energy extraction)
    
    Decay physics:
    - Fresh (decay > 2000): Safe, full energy
    - Decaying (1000 < decay < 2000): Losing energy, starting to smell
    - Putrid (500 < decay < 1000): Disease risk to nearby creatures
    - Bones (decay < 500): Mostly harmless, little energy left
    
    Burial (depth > 0.5) slows decay and prevents disease spread.
    """
    
    # Decay stages
    FRESH_THRESHOLD = 2000
    PUTRID_THRESHOLD = 1000
    BONES_THRESHOLD = 500
    
    def __init__(self, pos: np.ndarray, herbie_name: str, body_energy: float, 
                 cause_of_death: str = "unknown"):
        """
        Initialize Herbie corpse.
        
        Args:
            pos: Position
            herbie_name: Name of the deceased Herbie
            body_energy: Energy remaining in body
            cause_of_death: How they died
        """
        super().__init__(
            pos=pos,
            size=1.2,
            compliance=0.4,  # Not directly edible - must be dismembered or worked at
            mass=4.0,  # Heavy - hard to move
            color='#4a4a4a',  # Gray/pale
            energy=body_energy
        )
        self.herbie_name = herbie_name
        self.cause_of_death = cause_of_death
        self.decay_timer = 3000  # Longer decay than meat
        self.is_corpse = True
        self.dismember_progress = 0.0
        self.dismember_threshold = 50.0
        self.gripped_by = None
        
        # Putrefaction state
        self.is_putrid = False
        self.disease_emitted = False  # Only emit disease once
        self.burial_depth = 0.0  # Set externally by DiggingSystem
        
        # For partial consumption
        self.times_fed_from = 0
        
        # Grippable properties - corpses are heavy and hard to grip
        self.grip_props = type('GripProps', (), {
            'grip_difficulty': 0.7,  # Hard to grip (stiff, heavy)
            'weight': 4.0,  # Very heavy
            'tool_damage': 0.2,  # Could be used as blunt weapon
            'tool_type': 'blunt',
            'stackable': False
        })()
        
    def step(self) -> dict:
        """
        Corpse decays over time.
        
        Returns:
            dict with status info:
            - 'alive': bool - still present
            - 'emit_disease': bool - should spawn disease this step
            - 'putrefaction_intensity': float - smell/disease radius multiplier
        """
        result = {
            'alive': True,
            'emit_disease': False,
            'putrefaction_intensity': 0.0,
        }
        
        # Burial slows decay significantly
        decay_rate = 1.0 if self.burial_depth < 0.5 else 0.1
        self.decay_timer -= decay_rate
        
        # Energy rots away
        if self.decay_timer < self.FRESH_THRESHOLD:
            rot_rate = 0.001 if self.burial_depth < 0.5 else 0.0001
            self.energy *= (1.0 - rot_rate)
        
        # Update color based on decay
        if self.decay_timer > self.FRESH_THRESHOLD:
            self.color = '#4a4a4a'  # Gray
        elif self.decay_timer > self.PUTRID_THRESHOLD:
            self.color = '#3a5a3a'  # Greenish gray
        elif self.decay_timer > self.BONES_THRESHOLD:
            self.color = '#2a4a2a'  # Dark green - putrid
            self.is_putrid = True
        else:
            self.color = '#8a8a7a'  # Bone color
            self.is_putrid = False  # Too decayed for disease
        
        # Putrefaction disease emission (only if not buried)
        if self.is_putrid and not self.disease_emitted and self.burial_depth < 0.5:
            # Probabilistic disease emission - not guaranteed
            if np.random.random() < 0.002:  # ~0.2% chance per step while putrid
                result['emit_disease'] = True
                self.disease_emitted = True
                print(f"[PUTREFACTION] '{self.herbie_name}'s rotting corpse spawns disease!")
        
        # Calculate putrefaction intensity for smell/disease radius
        if self.is_putrid and self.burial_depth < 0.5:
            result['putrefaction_intensity'] = 1.0 - (self.decay_timer - self.BONES_THRESHOLD) / (self.PUTRID_THRESHOLD - self.BONES_THRESHOLD)
        
        # Full decay
        if self.decay_timer <= 0:
            self.alive = False
            result['alive'] = False
            print(f"[CORPSE] '{self.herbie_name}'s remains have decayed to bones")
        
        return result
    
    def extract_energy(self, amount: float) -> float:
        """
        Extract energy from corpse (feeding).
        
        Args:
            amount: Max energy to extract
            
        Returns:
            Actual energy extracted
        """
        # Harder to extract from intact corpse than meat chunks
        extraction_efficiency = 0.3  # Only get 30% of what you'd get from proper meat
        
        # Putrid meat is less nutritious and risky
        if self.is_putrid:
            extraction_efficiency *= 0.5
        
        extracted = min(amount * extraction_efficiency, self.energy * 0.1)
        self.energy -= extracted / extraction_efficiency  # Actual mass lost
        self.times_fed_from += 1
        
        # Feeding contributes to dismemberment
        self.dismember_progress += 0.5
        
        return extracted
    
    def add_dismember_progress(self, amount: float) -> bool:
        """
        Add progress toward dismemberment.
        
        Args:
            amount: Progress amount
            
        Returns:
            True if corpse should now become meat chunks
        """
        self.dismember_progress += amount
        if self.dismember_progress >= self.dismember_threshold:
            return True
        return False
    
    def to_meat_chunks(self) -> List['MeatChunk']:
        """
        Convert corpse to 4 meat chunks.
        
        Returns:
            List of MeatChunk objects
        """
        chunks = []
        energy_per_chunk = self.energy / 4
        
        offsets = [
            np.array([0.0, 1.0]),   # head
            np.array([0.0, -1.0]),  # torso  
            np.array([-1.0, 0.0]),  # left limbs
            np.array([1.0, 0.0]),   # right limbs
        ]
        
        for quadrant, offset in zip(MeatChunk.QUADRANT_NAMES, offsets):
            chunk_pos = self.pos + offset * np.random.uniform(0.3, 0.8)
            chunk = MeatChunk(
                chunk_pos, 
                "Herbie", 
                quadrant, 
                energy_per_chunk,
                source_name=self.herbie_name,
                is_putrid=self.is_putrid  # Inherit putrid state
            )
            chunks.append(chunk)
        
        self.alive = False
        print(f"[CORPSE] '{self.herbie_name}'s body was dismembered into meat")
        return chunks
    
    def __repr__(self):
        state = "putrid" if self.is_putrid else "fresh" if self.decay_timer > self.FRESH_THRESHOLD else "decaying"
        buried = ", buried" if self.burial_depth > 0.5 else ""
        return f"HerbieCorpse('{self.herbie_name}', E={self.energy:.1f}, {state}{buried})"


def spawn_meat_chunks(pos: np.ndarray, source_species: str, total_energy: float,
                      source_name: str = None) -> List[MeatChunk]:
    """
    Spawn 4 meat chunks (quadrants) from a killed creature.
    
    Used when Herbies kill an Apex with LITE_ORE.
    
    Args:
        pos: Death position
        source_species: Species that was killed
        total_energy: Total energy to distribute
        source_name: Name of killed creature (optional)
        
    Returns:
        List of 4 MeatChunk objects
    """
    chunks = []
    energy_per_chunk = total_energy / 4
    
    offsets = [
        np.array([0.0, 1.0]),   # head - north
        np.array([0.0, -1.0]),  # torso - south  
        np.array([-1.0, 0.0]),  # left limbs - west
        np.array([1.0, 0.0]),   # right limbs - east
    ]
    
    for quadrant, offset in zip(MeatChunk.QUADRANT_NAMES, offsets):
        chunk_pos = pos + offset * np.random.uniform(0.5, 1.5)
        chunk = MeatChunk(chunk_pos, source_species, quadrant, energy_per_chunk, source_name)
        chunks.append(chunk)
    
    name_str = f"'{source_name}'" if source_name else source_species
    print(f"[MEAT]  {name_str} dismembered into 4 chunks ({energy_per_chunk:.1f} energy each)")
    return chunks
