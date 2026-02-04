"""
LifetimeTracker - Tracks creature statistics over their lifetime.

Records metrics like:
- Distance traveled
- Food consumed
- Reproduction events
- Defecation events
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class LifetimeTracker:
    """Tracks statistics for a single creature over its lifetime."""
    
    creature_id: str
    generation: int
    parent_id: Optional[str]
    lineage_depth: int
    
    # Movement tracking
    total_distance: float = 0.0
    max_speed_achieved: float = 0.0
    
    # Feeding tracking
    total_food_consumed: float = 0.0
    total_defecation: float = 0.0
    defecation_events: List[Dict] = field(default_factory=list)
    
    # Reproduction tracking
    offspring_count: int = 0
    reproduction_events: List[int] = field(default_factory=list)
    
    # Step tracking
    birth_step: int = 0
    last_update_step: int = 0
    
    # Position tracking
    last_pos: Optional[tuple] = None
    
    def update(self, creature, step: int):
        """Update tracking with creature's current state."""
        self.last_update_step = step
        
        if self.birth_step == 0:
            self.birth_step = step
        
        # Track movement
        if hasattr(creature, 'pos') and creature.pos is not None:
            if self.last_pos is not None:
                import numpy as np
                dist = np.linalg.norm(creature.pos - np.array(self.last_pos))
                self.total_distance += dist
            self.last_pos = tuple(creature.pos)
        
        # Track speed
        if hasattr(creature, 'vel'):
            import numpy as np
            speed = np.linalg.norm(creature.vel)
            self.max_speed_achieved = max(self.max_speed_achieved, speed)
        
        # Track food
        if hasattr(creature, 'total_reward'):
            self.total_food_consumed = creature.total_reward
    
    def log_defecation(self, amount: float, step: int):
        """Log a defecation event."""
        self.total_defecation += amount
        self.defecation_events.append({
            'step': step,
            'amount': amount
        })
    
    def log_reproduction(self, step: int):
        """Log a reproduction event."""
        self.offspring_count += 1
        self.reproduction_events.append(step)
    
    def get_age(self, current_step: int) -> int:
        """Get creature's age in steps."""
        return current_step - self.birth_step
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'creature_id': self.creature_id,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'lineage_depth': self.lineage_depth,
            'total_distance': self.total_distance,
            'max_speed_achieved': self.max_speed_achieved,
            'total_food_consumed': self.total_food_consumed,
            'total_defecation': self.total_defecation,
            'offspring_count': self.offspring_count,
            'birth_step': self.birth_step,
            'last_update_step': self.last_update_step,
        }
