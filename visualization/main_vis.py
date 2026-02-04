"""
Main Visualization Class - Comprehensive real-time rendering.

This is the main visualization that brings together all renderers
and panels into a complete matplotlib-based display.

Supports:
- World view with all elements
- Day/night lighting
- Selected creature detail
- Population tracking
- Multiple view modes (world/local/isometric)
- Keyboard controls
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from collections import deque
from typing import Optional, List, TYPE_CHECKING

from .colors import (
    complex_to_rgb, get_sky_color_for_time, get_daynight_overlay_alpha
)
from .renderers import (
    render_terrain_background, render_creature, render_herbie_special,
    render_food, render_nutrients, render_weather, render_leviathan,
    render_ant_colony, render_nests, render_smears, render_holes,
    render_mushrooms, render_aquatic, render_mycelia_overlay
)
from .panels import (
    render_status_panel, render_extended_status, render_population_chart,
    PopulationTracker, render_species_legend, render_creature_detail,
    render_torus_panel, render_hunger_bars, render_achievement_popup,
    render_herbie_family_panel
)
from .specialized_views import IsometricCityView, EvolutionTreeView

if TYPE_CHECKING:
    from ..manager.creature_manager import CreatureManager
    from ..world.terrain import Terrain


class HerbieVisualization:
    """
    Full-featured visualization for HERBIE world simulation.
    
    Layout:
    - Row 0: World view (2 cols), Body field, Brain torus
    - Row 1: Population chart, Species legend, Hunger bars
    - Row 2: Status panel (2 cols), Extended status (2 cols)
    """
    
    def __init__(self, manager: 'CreatureManager', terrain: Optional['Terrain'] = None):
        """
        Initialize visualization.
        
        Args:
            manager: CreatureManager running the simulation
            terrain: Terrain for background rendering
        """
        self.manager = manager
        self.terrain = terrain or getattr(manager, 'terrain', None)
        
        # State
        self.selected_idx = 0
        self.step_counter = 0
        self.local_mode = False  # True = follow selected creature
        self.show_ants = True
        self.show_pheromones = True
        self.show_mycelia = False
        self.show_nests = True
        self.show_smears = True
        self.show_chemistry = True  # Z to toggle
        self.show_resonance = False  # R to toggle - global Schumann field
        self.show_ghosts = False  # O to toggle - ghost/spirit realm (Other side)
        self.ghost_cam = False  # Shift+O to toggle ghost cam mode
        self.ghost_cam_idx = 0  # Which ghost we're following
        self.paused = False
        
        # World size
        self.world_size = manager.world.world_size
        
        # Isometric view (separate window)
        self.isometric_view = IsometricCityView(manager)
        
        # Evolution tree view (separate window)
        self.tree_view = EvolutionTreeView(manager.evo_tree)
        
        # Population tracking
        species_names = [s.name for s in manager.world.species_list] if hasattr(manager.world, 'species_list') else ['Herbie', 'Blob', 'Apex', 'Grazer', 'Scavenger']
        self.pop_tracker = PopulationTracker(species_names)
        
        # Recent events for display
        self.recent_poops = []
        self.poop_glow_duration = 150
        
        # Achievement display
        self.current_achievement = None
        self.achievement_timer = 0
        
        # Setup matplotlib
        self._setup_figure()
        
        # Show help on startup
        self._print_controls()
    
    def _print_controls(self):
        """Print all keyboard controls."""
        print("[Visualization] Initialized")
        print("=" * 60)
        print("  KEYBOARD CONTROLS")
        print("=" * 60)
        print("  NAVIGATION")
        print("    <-/->       Select creature          H   Cycle Herbies")
        print("    L         Local/World view         +/- Zoom in/out")
        print("    Arrow     Pan (in iso view)")
        print("")
        print("  SIMULATION")
        print("    Space     Pause/Resume             V   Cycle log verbosity")
        print("    F5        Quick Save               F9  Load info")
        print("")
        print("  VIEWS (toggle on/off)")
        print("    I         Isometric (SimCity)      T   Evolution Tree")
        print("    U         Underground/Fungal       G   Art Gallery")
        print("    W         Aquatic creatures        D   DNA/Genetics view")
        print("")
        print("  OVERLAYS (toggle on/off)")
        print("    A         Ants                     P   Pheromones")
        print("    M         Mycelia network          N   Nests")
        print("    S         Smear marks              Z   Chemistry")
        print("    E         Terrain elevation        R   Resonance field")
        print("    O         Other side (ghosts)      Shift+O  Ghost Cam mode")
        print("              < > cycle ghosts when in Ghost Cam")
        print("")
        print("  TREE VIEW (when open)")
        print("    [/]       Cycle species            A   Show all species")
        print("=" * 60)
    
    def register_poop(self, pos: np.ndarray, amount: float):
        """Register a new poop for glowing effect."""
        self.recent_poops.append({
            'pos': pos.copy(),
            'birth': self.step_counter,
            'amount': amount
        })
    
    def _setup_figure(self):
        """Create matplotlib figure and axes."""
        # Register poop callback with world
        if hasattr(self.manager, 'world'):
            self.manager.world.on_poop_callback = self.register_poop
        
        plt.ion()
        # Larger figure for better world visibility
        self.fig = plt.figure(figsize=(22, 12), facecolor='black')
        
        # Adjusted grid - world takes more space
        gs = self.fig.add_gridspec(
            3, 5, 
            height_ratios=[2.0, 0.5, 0.35],
            width_ratios=[1.5, 1.5, 0.8, 0.8, 0.8],
            wspace=0.12, hspace=0.15
        )
        
        # Row 0: Main displays - world gets 2 columns worth
        self.ax_world = self.fig.add_subplot(gs[0, 0:2])
        self.ax_world.set_facecolor('black')
        
        self.ax_body = self.fig.add_subplot(gs[0, 2:4])
        self.ax_body.set_facecolor('black')
        
        self.ax_torus = self.fig.add_subplot(gs[0, 4], projection='polar')
        self.ax_torus.set_facecolor('black')
        
        # Row 1: Charts
        self.ax_pop = self.fig.add_subplot(gs[1, 0:2])
        self.ax_pop.set_facecolor('black')
        
        self.ax_legend = self.fig.add_subplot(gs[1, 2])
        self.ax_legend.set_facecolor('black')
        self.ax_legend.axis('off')
        
        self.ax_hunger = self.fig.add_subplot(gs[1, 3:5])
        self.ax_hunger.set_facecolor('black')
        
        # Row 2: Status and Help
        self.ax_status = self.fig.add_subplot(gs[2, 0:2])
        self.ax_status.set_facecolor('black')
        self.ax_status.axis('off')
        
        self.ax_extended = self.fig.add_subplot(gs[2, 2:3])
        self.ax_extended.set_facecolor('black')
        
        # Help panel (always visible)
        self.ax_help = self.fig.add_subplot(gs[2, 3:5])
        self.ax_help.set_facecolor('#0a0a0a')
        self.ax_help.axis('off')
        self._render_help_panel()
        
        # Event handlers
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)
    
    def _render_help_panel(self):
        """Render persistent help panel."""
        self.ax_help.clear()
        self.ax_help.set_facecolor('#0a0a0a')
        self.ax_help.axis('off')
        
        help_text = (
            "CONTROLS\n"
            "\n"
            "<-/->  Select creature\n"
            "H    Cycle Herbies\n"
            "Space Pause/Resume\n"
            "V    Log verbosity\n"
            "\n"
            "VIEWS\n"
            "I Isometric  T Tree\n"
            "W Aquatic    U Fungal\n"
            "E Embryo development\n"
            "\n"
            "OVERLAYS\n"
            "A Ants    P Pheromones\n"
            "M Mycelia N Nests\n"
            "S Smears  Z Chemistry\n"
            "\n"
            "? Show full help"
        )
        
        self.ax_help.text(
            0.05, 0.95, help_text,
            transform=self.ax_help.transAxes,
            fontsize=7, fontfamily='monospace',
            color='#66ccff', va='top',
            linespacing=1.1
        )
    
    def _on_key(self, event):
        """Handle keyboard input."""
        creatures = self.manager.creatures
        alive = [c for c in creatures if c.alive]
        
        # If embryo view is active, arrow keys cycle pregnant creatures
        if getattr(self, 'embryo_view', False) and event.key in ['left', 'right']:
            pregnant = self._get_pregnant_creatures()
            if pregnant:
                if event.key == 'right':
                    self.embryo_view_index = (getattr(self, 'embryo_view_index', 0) + 1) % len(pregnant)
                elif event.key == 'left':
                    self.embryo_view_index = (getattr(self, 'embryo_view_index', 0) - 1) % len(pregnant)
                creature = pregnant[self.embryo_view_index]
                name = creature.mating_state.name if hasattr(creature, 'mating_state') else creature.creature_id[:8]
                print(f"[Embryo View] Watching {name}'s embryo ({self.embryo_view_index + 1}/{len(pregnant)})")
            else:
                print("[Embryo View] No pregnant creatures")
            return
        
        if event.key == 'right':
            self.selected_idx = (self.selected_idx + 1) % max(1, len(alive))
        elif event.key == 'left':
            self.selected_idx = (self.selected_idx - 1) % max(1, len(alive))
        elif event.key == ' ':
            self.paused = not self.paused
            print(f"[Viz] {'PAUSED' if self.paused else 'RUNNING'}")
        elif event.key == 'l':
            self.local_mode = not self.local_mode
            print(f"[Viz] Local mode: {'ON' if self.local_mode else 'OFF'}")
        elif event.key == 'a':
            self.show_ants = not self.show_ants
            print(f"[Viz] Ants: {'ON' if self.show_ants else 'OFF'}")
        elif event.key == 'p':
            self.show_pheromones = not self.show_pheromones
            print(f"[Viz] Pheromones: {'ON' if self.show_pheromones else 'OFF'}")
        elif event.key == 'm':
            self.show_mycelia = not self.show_mycelia
            print(f"[Viz] Mycelia: {'ON' if self.show_mycelia else 'OFF'}")
        elif event.key == 'n':
            self.show_nests = not self.show_nests
            print(f"[Viz] Nests: {'ON' if self.show_nests else 'OFF'}")
        elif event.key == 's':
            self.show_smears = not self.show_smears
            print(f"[Viz] Smears: {'ON' if self.show_smears else 'OFF'}")
        elif event.key == 'h':
            # Cycle through Herbies only
            herbie_indices = [i for i, c in enumerate(alive) 
                            if c.species.name == 'Herbie']
            if herbie_indices:
                current = herbie_indices.index(self.selected_idx) if self.selected_idx in herbie_indices else -1
                next_idx = (current + 1) % len(herbie_indices)
                self.selected_idx = herbie_indices[next_idx]
        elif event.key == 'z':
            self.show_chemistry = not self.show_chemistry
            print(f"[Viz] Chemistry display: {'ON' if self.show_chemistry else 'OFF'}")
        elif event.key == 'g':
            # Art gallery - show smear art clusters
            if hasattr(self.manager, 'smears'):
                clusters = self.manager.smears.find_art_clusters()
                if clusters:
                    print(f"[ART GALLERY] Found {len(clusters)} art clusters")
                    for i, cluster in enumerate(clusters):
                        print(f"  Cluster {i+1}: {cluster.get('element_count', 0)} elements at ({cluster.get('center', [0,0])[0]:.0f}, {cluster.get('center', [0,0])[1]:.0f})")
                else:
                    print("[ART GALLERY] No art clusters detected yet")
        elif event.key == 'e':
            # Toggle embryo development view
            self.embryo_view = not getattr(self, 'embryo_view', False)
            if self.embryo_view:
                self.embryo_view_index = 0
                print("[Viz]  EMBRYO VIEW - watching development")
                print("       <-/-> cycle pregnant creatures")
            else:
                print("[Viz] Embryo view OFF")
        elif event.key == 'w':
            # Toggle aquatic view / cycle water bodies
            if getattr(self, 'aquatic_view', False):
                # Already in aquatic view - cycle to next water body or exit
                if hasattr(self, '_water_bodies') and self._water_bodies and len(self._water_bodies) > 0:
                    self._water_body_idx = getattr(self, '_water_body_idx', 0) + 1
                    if self._water_body_idx >= len(self._water_bodies):
                        # Cycled through all - exit
                        self.aquatic_view = False
                        self._aquatic_center = None
                        self._water_body_idx = 0
                        print("[Viz] Aquatic view OFF")
                    else:
                        # Move to next water body
                        wb = self._water_bodies[self._water_body_idx]
                        self._aquatic_center = wb['center']
                        print(f"[Viz] Lake {self._water_body_idx + 1}/{len(self._water_bodies)} "
                              f"at ({wb['center'][0]:.0f}, {wb['center'][1]:.0f}) size={wb['size']}")
                else:
                    self.aquatic_view = False
                    self._aquatic_center = None
                    print("[Viz] Aquatic view OFF")
            else:
                # Enter aquatic view
                self.aquatic_view = True
                self._aquatic_zoom = 1  # Start at medium
                self._water_body_idx = 0
                print("[Viz]  AQUATIC VIEW")
                print("       W = cycle lakes, +/- = zoom")
                self._show_aquatic_view()
        elif event.key == '+' or event.key == '=':
            # Zoom in (for aquatic view)
            if getattr(self, 'aquatic_view', False):
                self._aquatic_zoom = max(0, getattr(self, '_aquatic_zoom', 1) - 1)
                zoom_names = ['CLOSE', 'MEDIUM', 'WORLD']
                print(f"[Viz] Aquatic zoom: {zoom_names[self._aquatic_zoom]}")
        elif event.key == '-' or event.key == '_':
            # Zoom out (for aquatic view)
            if getattr(self, 'aquatic_view', False):
                self._aquatic_zoom = min(2, getattr(self, '_aquatic_zoom', 1) + 1)
                zoom_names = ['CLOSE', 'MEDIUM', 'WORLD']
                print(f"[Viz] Aquatic zoom: {zoom_names[self._aquatic_zoom]}")
        elif event.key == 'u':
            # Toggle fungal/mycelia view (underground)
            self.fungal_view = not getattr(self, 'fungal_view', False)
            if self.fungal_view:
                print("[Viz]  FUNGAL VIEW - mycorrhizal network")
                self._show_fungal_view()
            else:
                print("[Viz] Fungal view OFF")
        elif event.key == 'r':
            # Toggle resonance field overlay
            self.show_resonance = not self.show_resonance
            print(f"[Viz] Resonance field: {'ON' if self.show_resonance else 'OFF'}")
            if self.show_resonance:
                print("       Showing global Schumann-like EM field")
        elif event.key == 'R':  # Shift+R
            # Cycle EM field mode
            if hasattr(self.manager, 'em_field') and hasattr(self.manager, 'em_config'):
                from ..world.electromagnetic_field import EMFieldMode
                modes = [EMFieldMode.OFF, EMFieldMode.ELECTROSTATIC, EMFieldMode.FULL_MAXWELL]
                mode_names = ['OFF', 'ELECTROSTATIC', 'FULL_MAXWELL']
                current_idx = modes.index(self.manager.em_config.mode)
                next_idx = (current_idx + 1) % len(modes)
                self.manager.em_config.mode = modes[next_idx]
                self.manager.em_field.config.mode = modes[next_idx]
                print(f"[Viz]  EM Field mode: {mode_names[next_idx]}")
                if modes[next_idx] == EMFieldMode.FULL_MAXWELL:
                    print("        WARNING: Full Maxwell may impact performance!")
                elif modes[next_idx] == EMFieldMode.OFF:
                    print("        EM field disabled - no electromagnetic interactions")
            else:
                print("[Viz] No EM field available")
        elif event.key == 'o':
            # Toggle ghost/spirit realm visibility
            self.show_ghosts = not self.show_ghosts
            print(f"[Viz] Spirit realm: {'ON' if self.show_ghosts else 'OFF'}")
            if self.show_ghosts:
                ghost_count = len(self.manager.ghost_field.ghosts) if hasattr(self.manager, 'ghost_field') else 0
                print(f"        {ghost_count} spirits currently wandering...")
                print("        Press Shift+O for GHOST CAM, [/] to cycle ghosts")
        elif event.key == 'O':  # Shift+O
            # Toggle ghost cam mode - follow a specific ghost
            if not hasattr(self.manager, 'ghost_field'):
                print("[Viz] No ghost field available")
            else:
                self.ghost_cam = not self.ghost_cam
                if self.ghost_cam:
                    self.show_ghosts = True  # Auto-enable ghost overlay
                    ghosts = [g for g in self.manager.ghost_field.ghosts if g.alive]
                    if ghosts:
                        self.ghost_cam_idx = 0
                        ghost = ghosts[0]
                        name = ghost.name or f"Spirit #{self.ghost_cam_idx}"
                        print(f"[Viz]  GHOST CAM - Following {name}")
                        print("        [/] cycle between spirits, Shift+O to exit")
                    else:
                        print("[Viz] No spirits to follow...")
                        self.ghost_cam = False
                else:
                    print("[Viz] Ghost cam OFF")
        elif event.key == 'i':
            # Toggle isometric SimCity-style view
            self.isometric_view.toggle()
            if self.isometric_view.visible:
                print("[Viz]  ISOMETRIC VIEW - SimCity style (arrow keys to pan)")
            else:
                print("[Viz] Isometric view OFF")
        elif event.key == 't':
            # Toggle evolution tree view
            self.tree_view.toggle()
            if self.tree_view.visible:
                print("[View] EVOLUTION TREE: ON")
                print("       [/] cycle species, A show all")
            else:
                print("[Viz] Evolution tree OFF")
        elif event.key == 'd':
            # Show DNA/genetics view for selected creature
            self._show_genetics_view()
        elif event.key == 'v':
            # Cycle console verbosity
            from ..events.console_log import console_log
            level_name = console_log().cycle_verbosity()
            print(f"[Viz] Console verbosity: {level_name}")
        elif event.key == 'f5':
            # Quick save
            self._quick_save()
        elif event.key == 'f9':
            # Quick load (note: this won't work mid-session, just prints info)
            print("[Viz] F9: To load a save, restart with: python -m herbie_world --load <file>")
            save_path = getattr(self.manager, '_save_file_path', None)
            if save_path:
                print(f"[Viz] Current save path: {save_path}")
        elif event.key == '?':
            # Show help
            self._print_controls()
        elif event.key in [',', '.', '<', '>'] and self.ghost_cam:
            # Cycle through ghosts when in ghost cam mode
            if hasattr(self.manager, 'ghost_field'):
                ghosts = [g for g in self.manager.ghost_field.ghosts if g.alive]
                if ghosts:
                    if event.key in ['.', '>']:
                        self.ghost_cam_idx = (self.ghost_cam_idx + 1) % len(ghosts)
                    else:
                        self.ghost_cam_idx = (self.ghost_cam_idx - 1) % len(ghosts)
                    ghost = ghosts[self.ghost_cam_idx]
                    name = ghost.name or f"Spirit #{self.ghost_cam_idx}"
                    coherence = ghost._compute_coherence()
                    print(f"[Ghost Cam] Following {name} (coherence: {coherence:.2f})")
    
    def _quick_save(self):
        """Quick save current world state (F5)."""
        from ..persistence.world_state import WorldPersistence
        
        save_path = getattr(self.manager, '_save_file_path', None)
        if save_path is None:
            # Default path
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, '..', 'data', 'world_state.npz')
        
        print(f"[Viz] Quick saving to {save_path}...")
        if WorldPersistence.save_world(self.manager, save_path):
            print("[Viz] Quick save complete!")
        else:
            print("[Viz] Quick save FAILED")
    
    def _show_genetics_view(self):
        """Display genetic information for selected creature."""
        if self.selected_idx >= len(self.manager.creatures):
            print("[Genetics] No creature selected")
            return
        
        creature = self.manager.creatures[self.selected_idx]
        
        # Check if it's a Herbie with genetics
        if creature.species.name != "Herbie" or not hasattr(creature, 'mating_state'):
            print(f"[Genetics] {creature.species.name} creatures don't have detailed genetics")
            return
        
        genome = creature.mating_state.genome
        name = creature.mating_state.name or creature.creature_id[:8]
        gen = creature.generation
        
        print("\n" + "="*70)
        print(f"  HERITABLE STATE: {name} (Generation {gen})")
        print("="*70)
        
        # Show phenotype (observed traits)
        phenotype = genome.get_full_phenotype()
        print("\n  PHENOTYPE (Observed Traits)")
        print("  " + "-"*40)
        for trait, value in phenotype.items():
            bar_len = int(value * 5) if value <= 2 else int(value * 2)
            bar_len = min(10, max(0, bar_len))
            bar = '●' * bar_len + '○' * (10 - bar_len)
            print(f"  {trait:20s}: {value:.2f} [{bar}]")
        
        # Show heritable state (parent info)
        if genome.heritable_state:
            hs = genome.heritable_state
            print("\n  PARENT INHERITANCE (Pure Tracking)")
            print("  " + "-"*40)
            
            if hs.parent_a_id:
                print(f"  Parent A: {hs.parent_a_id}")
                if hs.parent_a_basins:
                    print(f"    Torus basins: {len(hs.parent_a_basins)} attractors")
                    for i, basin in enumerate(hs.parent_a_basins[:3]):
                        print(f"      Basin {i}: depth={basin['depth']:.3f}")
                if hs.parent_a_phenotype:
                    print(f"    Phenotype at conception:")
                    for k, v in list(hs.parent_a_phenotype.items())[:4]:
                        print(f"      {k}: {v:.2f}")
            
            if hs.parent_b_id:
                print(f"  Parent B: {hs.parent_b_id}")
                if hs.parent_b_basins:
                    print(f"    Torus basins: {len(hs.parent_b_basins)} attractors")
                if hs.parent_b_phenotype:
                    print(f"    Phenotype at conception:")
                    for k, v in list(hs.parent_b_phenotype.items())[:4]:
                        print(f"      {k}: {v:.2f}")
            
            print(f"  Mixing ratio: {hs.torus_mixing_ratio:.2f}")
        
        # Show development context
        if genome.development:
            d = genome.development
            print("\n  DEVELOPMENT CONTEXT (Environment)")
            print("  " + "-"*40)
            print(f"  Mother energy at conception: {d.mother_energy:.1f}")
            print(f"  Mother stress: {d.mother_stress:.2f}")
            print(f"  Mother hunger: {d.mother_hunger:.2f}")
            print(f"  Audio exposure: {d.audio_exposure:.2f}")
            print(f"  Season at birth: {d.season_at_birth}")
            print(f"  Nearby herbies: {d.n_nearby_herbies}")
            if d.predator_exposure > 0:
                print(f"  Predator exposure: {d.predator_exposure:.2f}")
        
        # Show current torus state
        if hasattr(creature, 'torus') and creature.torus is not None:
            from ..creature.herbie_social import HeritableState
            basins = HeritableState._extract_basins(creature.torus)
            print("\n  CURRENT TORUS STATE")
            print("  " + "-"*40)
            print(f"  Active basins: {len(basins)}")
            for i, basin in enumerate(basins[:5]):
                print(f"    Basin {i}: pos={basin['pos']}, depth={basin['depth']:.3f}")
        
        # Note about Mendelian (if present, it's optional)
        if genome.mendelian:
            print("\n  [Optional Mendelian layer present - for visualization only]")
        
        print("="*70 + "\n")
    
    def update(self):
        """Update visualization for current frame."""
        if self.paused:
            plt.pause(0.05)
            return
        
        self.step_counter += 1
        creatures = self.manager.creatures
        world = self.manager.world
        
        # Clean old poops
        self.recent_poops = [p for p in self.recent_poops 
                           if self.step_counter - p['birth'] < self.poop_glow_duration]
        
        # Update population tracker
        self.pop_tracker.update(creatures)
        
        # Get day/night state
        dn_state = self.manager.daynight.get_state()
        light_level = dn_state.light_level
        
        # Render world view - check for special view modes
        if getattr(self, 'aquatic_view', False):
            self._render_aquatic_visual(light_level)
        elif getattr(self, 'fungal_view', False):
            self._render_fungal_visual(light_level)
        elif getattr(self, 'ghost_cam', False):
            self._render_ghost_cam(light_level)
        elif self.local_mode:
            self._update_local_view(light_level)
        else:
            self._update_world_view(light_level)
        
        # Update isometric view if active
        if self.isometric_view.visible:
            self.isometric_view.update()
        
        # Update tree view if active
        if self.tree_view.visible:
            step = self.manager.seasons.step_count if self.manager.seasons else self.step_counter
            self.tree_view.update(step)
        
        # Render selected creature detail OR embryo view
        alive = [c for c in creatures if c.alive]
        if getattr(self, 'embryo_view', False):
            # Show embryo development view instead of creature detail
            self._render_embryo_view(self.ax_body)
        elif alive and 0 <= self.selected_idx < len(alive):
            selected = alive[self.selected_idx]
            render_creature_detail(self.ax_body, selected)
            render_torus_panel(self.ax_torus, selected)
        
        # Render population chart
        render_population_chart(self.ax_pop, self.pop_tracker)
        
        # Render species legend
        stats = self.manager.get_statistics()
        render_species_legend(self.ax_legend, stats['species_counts'])
        
        # Render hunger bars
        render_hunger_bars(self.ax_hunger, creatures)
        
        # Render status panels
        render_status_panel(self.ax_status, self.manager, self.step_counter)
        render_extended_status(self.ax_extended, self.manager)
        
        # Check for achievement display
        achievement = self.manager.achievements.get_current_display()
        if achievement and achievement != self.current_achievement:
            self.current_achievement = achievement
            self.achievement_timer = 150
        
        if self.achievement_timer > 0:
            self.achievement_timer -= 1
            # Could overlay achievement popup here
        
        # Update display
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def _update_world_view(self, light_level: float):
        """Render full world view."""
        self.ax_world.clear()
        
        creatures = self.manager.creatures
        world = self.manager.world
        manager = self.manager
        season = manager.seasons.current_season
        dn_state = manager.daynight.get_state()
        
        half = self.world_size / 2
        
        # Sky color
        sky_color = get_sky_color_for_time(dn_state.time_of_day, dn_state.is_day)
        self.ax_world.set_facecolor(sky_color)
        
        # Terrain background
        if self.terrain:
            terrain_img = render_terrain_background(
                self.ax_world, self.terrain, self.world_size, light_level
            )
            self.ax_world.imshow(
                terrain_img, origin='lower',
                extent=[-half, half, -half, half],
                alpha=0.7 * light_level + 0.3
            )
        
        # Set view limits - check for aquatic zoom
        if getattr(self, 'aquatic_view', False) and hasattr(self, '_aquatic_center') and self._aquatic_center is not None:
            # Zoom to water body
            center = self._aquatic_center
            zoom_radius = 35  # Show water body + surrounding area
            self.ax_world.set_xlim(center[0] - zoom_radius, center[0] + zoom_radius)
            self.ax_world.set_ylim(center[1] - zoom_radius, center[1] + zoom_radius)
        else:
            self.ax_world.set_xlim(-half, half)
            self.ax_world.set_ylim(-half, half)
        self.ax_world.set_aspect('equal')
        
        # Title - format time as HH:MM
        season_sym = {'Spring': 'SPR', 'Summer': 'SUM', 'Autumn': 'AUT', 'Winter': 'WIN'}
        hour = int(dn_state.time_of_day * 24)
        minute = int((dn_state.time_of_day * 24 - hour) * 60)
        time_str = f"{hour:02d}:{minute:02d}"
        day_num = manager.daynight.total_days if hasattr(manager.daynight, 'total_days') else manager.seasons.step_count // 500
        self.ax_world.set_title(
            f'{time_str} D{day_num} | {season_sym.get(season.name, "?")} Y{manager.seasons.year} | '
            f'{len(creatures)} creatures | {world.count_food()} food',
            color='white', fontsize=10
        )
        
        # Night overlay
        if not dn_state.is_day:
            overlay_alpha = get_daynight_overlay_alpha(dn_state.light_level)
            night_overlay = Rectangle(
                (-half, -half), self.world_size, self.world_size,
                fc='midnightblue', ec='none', alpha=overlay_alpha, zorder=0.5
            )
            self.ax_world.add_patch(night_overlay)
        
        # Mycelia overlay
        if self.show_mycelia and hasattr(manager, 'mycelia'):
            render_mycelia_overlay(
                self.ax_world, manager.mycelia, self.world_size, light_level
            )
        
        # Resonance field overlay (psychedelic Schumann field visualization)
        if self.show_resonance and hasattr(manager, 'resonance_field'):
            self._render_resonance_field(manager.resonance_field, light_level)
        
        # Ghost/Spirit realm visualization
        if self.show_ghosts and hasattr(manager, 'ghost_field'):
            self._render_ghosts(manager.ghost_field, light_level)
        
        # Ant colony
        if (self.show_ants or self.show_pheromones) and hasattr(manager, 'ant_colony'):
            render_ant_colony(
                self.ax_world, manager.ant_colony, self.world_size,
                show_pheromones=self.show_pheromones, show_ants=self.show_ants
            )
        
        # Weather
        render_weather(self.ax_world, manager.weather.active_events)
        
        # Food and nutrients
        render_food(self.ax_world, world.objects, light_level)
        render_nutrients(self.ax_world, world.nutrients, light_level)
        
        # Glowing poop (recent defecations) - nutrient cycle visualization
        for poop in self.recent_poops:
            age = self.step_counter - poop['birth']
            fade = max(0, min(1.0, 1.0 - age / self.poop_glow_duration))
            
            if fade > 0:
                pos = poop['pos']
                amount = poop.get('amount', 5)
                
                # Size based on amount
                base_size = 0.4 + 0.3 * min(amount / 10, 1.0)
                
                # Pulsing glow effect
                pulse = 0.8 + 0.2 * np.sin(self.step_counter * 0.1 + poop['birth'])
                glow_size = base_size * (1.5 + 0.5 * fade) * pulse
                
                # Outer glow - golden brown (clamp alpha)
                self.ax_world.add_patch(Circle(
                    pos, glow_size, fc='#8B6914', ec='none', alpha=min(1.0, 0.25 * fade * light_level)
                ))
                
                # Middle layer (clamp alpha)
                self.ax_world.add_patch(Circle(
                    pos, base_size * 0.8, fc='#654321', ec='none', alpha=min(1.0, 0.5 * fade * light_level)
                ))
                
                # Core - darker (clamp alpha)
                self.ax_world.add_patch(Circle(
                    pos, base_size * 0.4, fc='#3d2817', ec='none', alpha=min(1.0, 0.7 * fade * light_level)
                ))
                
                # Nitrogen sparkle as it decomposes (becoming fertilizer)
                if age > 30 and fade > 0.3:
                    sparkle_alpha = min(1.0, 0.4 * fade * (1 + np.sin(self.step_counter * 0.3)) * 0.5 * light_level)
                    self.ax_world.plot(pos[0], pos[1], '*', color='#90EE90', 
                                      alpha=sparkle_alpha, ms=4 + 2 * fade)
        
        # Aquatic
        if hasattr(manager, 'aquatic'):
            render_aquatic(self.ax_world, manager.aquatic, light_level)
        
        # Leviathan
        if hasattr(manager, 'leviathan_mgr'):
            render_leviathan(self.ax_world, manager.leviathan_mgr, self.step_counter)
        
        # Emergent features
        if self.show_nests and hasattr(manager, 'nest_tracker'):
            nests = manager.nest_tracker.get_nests_for_display()
            render_nests(self.ax_world, nests, light_level)
        
        if self.show_smears and hasattr(manager, 'smears'):
            marks = manager.smears.get_marks_for_display()
            render_smears(self.ax_world, marks, light_level)
        
        if hasattr(manager, 'digging'):
            holes = manager.digging.get_holes_for_display()
            render_holes(self.ax_world, holes, light_level)
        
        if hasattr(manager, 'mycelia'):
            mushrooms = [(m.pos, m.size, m.age / m.max_age) 
                        for m in manager.mycelia.fruiting_bodies]
            render_mushrooms(self.ax_world, mushrooms, light_level)
        
        # Chemistry elements and constructions (Z to toggle)
        if self.show_chemistry and hasattr(manager, 'element_spawner') and manager.element_spawner is not None:
            from .chemistry_render import render_elements, render_constructions
            render_elements(self.ax_world, manager.element_spawner)
            render_constructions(self.ax_world, manager.element_spawner)
        
        # === GHOSTS ===
        # Render persisting torus wavefunctions as ethereal wisps
        if hasattr(manager, 'ghost_field'):
            for ghost in manager.ghost_field.ghosts:
                coherence = ghost._compute_coherence()
                pos = ghost.pos
                
                # Size based on coherence (more coherent = more visible)
                base_size = 1.5 + coherence * 2.0
                
                # Pulsing effect based on torus phase
                phase = np.angle(np.sum(ghost.psi))
                pulse = 0.8 + 0.2 * np.sin(self.step_counter * 0.1 + phase)
                
                # Outer ethereal glow (very transparent)
                glow_size = base_size * 2.0 * pulse
                glow_alpha = min(1.0, 0.15 * coherence * light_level)
                self.ax_world.add_patch(Circle(
                    pos, glow_size, fc='#E6E6FA', ec='none', alpha=glow_alpha
                ))
                
                # Inner wisp (slightly more visible)
                inner_alpha = min(1.0, 0.3 * coherence * light_level)
                self.ax_world.add_patch(Circle(
                    pos, base_size * pulse, fc='#B0C4DE', ec='none', alpha=inner_alpha
                ))
                
                # Core (most visible part)
                core_alpha = min(1.0, 0.5 * coherence * light_level)
                self.ax_world.add_patch(Circle(
                    pos, base_size * 0.4 * pulse, fc='#87CEEB', ec='none', alpha=core_alpha
                ))
                
                # Circulation indicator - small rotating marker
                circ_angle = ghost.initial_circulation + self.step_counter * 0.05
                marker_offset = np.array([np.cos(circ_angle), np.sin(circ_angle)]) * base_size * 0.6
                marker_alpha = min(1.0, 0.4 * coherence * light_level)
                self.ax_world.plot(pos[0] + marker_offset[0], pos[1] + marker_offset[1],
                                  'o', color='white', alpha=marker_alpha, ms=2)
                
                # Name label for named ghosts (Herbies)
                if ghost.name and coherence > 0.2:
                    self.ax_world.text(pos[0], pos[1] + base_size + 0.5, 
                                      f'{ghost.name}', ha='center', va='bottom',
                                      fontsize=7, color='#B0C4DE', 
                                      alpha=min(1.0, 0.6 * coherence * light_level))
            
            # Mark congregation centers
            centers = manager.ghost_field.get_congregation_centers()
            for center in centers:
                cpos = center['pos']
                strength = center['total_coherence']
                # Draw a subtle "attractor" marker
                self.ax_world.add_patch(Circle(
                    cpos, 3.0, fc='none', ec='#DDA0DD', 
                    alpha=min(0.5, 0.2 * strength * light_level), 
                    lw=1, linestyle='--'
                ))
        
        # Creatures
        alive = [c for c in creatures if c.alive]
        for i, creature in enumerate(alive):
            is_selected = (i == self.selected_idx)
            render_creature(
                self.ax_world, creature, 
                selected=is_selected, light_level=light_level
            )
            
            # Extra rendering for Herbies
            if creature.species.name == 'Herbie':
                render_herbie_special(self.ax_world, creature, light_level)
    
    def _draw_pixelated_terrain(self, center: np.ndarray, view_radius: float, 
                                  light_level: float):
        """
        Draw discrete pixelated terrain for local view.
        
        Philosophically correct: the world is discrete (collapsed),
        creatures are continuous (wavefunction). Each cell represents
        a collapsed quantum of space at the t hyperplane.
        """
        cell_size = 2.0  # Discrete quantum of terrain
        half_world = self.world_size / 2
        
        view_min = center - view_radius
        view_max = center + view_radius
        
        x_start = np.floor(view_min[0] / cell_size) * cell_size
        y_start = np.floor(view_min[1] / cell_size) * cell_size
        
        x = x_start
        while x < view_max[0]:
            y = y_start
            while y < view_max[1]:
                pos = np.array([x + cell_size/2, y + cell_size/2])
                
                # Clamp to world bounds
                clamped = np.clip(pos, -half_world + 1, half_world - 1)
                
                terrain_type = self.terrain.get_terrain_at(clamped)
                color = terrain_type.color
                
                # Slight variation for texture (discrete noise on discrete grid)
                variation = 0.05 * np.sin(x * 1.3) * np.cos(y * 1.7)
                
                rect = Rectangle(
                    (x, y), cell_size * 0.95, cell_size * 0.95,
                    fc=color, ec='none', 
                    alpha=(0.6 + variation) * (0.5 + 0.5 * light_level)
                )
                self.ax_world.add_patch(rect)
                
                y += cell_size
            x += cell_size
    
    def _update_local_view(self, light_level: float):
        """
        Render local view centered on selected creature.
        
        Uses discrete pixelated terrain to represent collapsed
        quantum reality - the substrate through which continuous
        NLSE creatures move.
        """
        self.ax_world.clear()
        
        creatures = self.manager.creatures
        alive = [c for c in creatures if c.alive]
        
        if not alive or self.selected_idx >= len(alive):
            self.ax_world.set_facecolor('black')
            self.ax_world.text(0.5, 0.5, 'No creature selected',
                             color='white', ha='center', va='center',
                             transform=self.ax_world.transAxes)
            return
        
        selected = alive[self.selected_idx]
        center = selected.pos
        view_radius = 25.0
        
        # Dark background for local view
        self.ax_world.set_facecolor('#1a1a2e')
        
        # Set view bounds first
        self.ax_world.set_xlim(center[0] - view_radius, center[0] + view_radius)
        self.ax_world.set_ylim(center[1] - view_radius, center[1] + view_radius)
        self.ax_world.set_aspect('equal')
        
        # Discrete pixelated terrain (collapsed quantum substrate)
        if self.terrain:
            self._draw_pixelated_terrain(center, view_radius, light_level)
        
        # Title with terrain info
        terrain_here = self.terrain.get_terrain_at(selected.pos) if self.terrain else None
        terrain_name = terrain_here.name.upper() if terrain_here else "?"
        
        name = ""
        if hasattr(selected, 'mating_state') and selected.mating_state.name:
            name = f" '{selected.mating_state.name}'"
        
        hunger_pct = ""
        if hasattr(selected, 'metabolism'):
            hunger_pct = f" | Hunger: {selected.metabolism.hunger:.0%}"
        
        self.ax_world.set_title(
            f"[LOCAL] {terrain_name} | {selected.species.name}{name} G{selected.generation}{hunger_pct}",
            color='cyan', fontsize=10
        )
        
        # Render nearby objects (larger in local view)
        world = self.manager.world
        
        for obj in world.objects:
            if obj.alive:
                dist = np.linalg.norm(obj.pos - center)
                if dist < view_radius * 1.5:
                    display_size = obj.size * 1.5
                    is_food = obj.compliance > 0.5
                    
                    if is_food:
                        # Glow for high-energy food
                        energy_ratio = obj.energy / max(getattr(obj, 'max_energy', 50), 1)
                        if energy_ratio > 0.5:
                            self.ax_world.add_patch(Circle(
                                obj.pos, display_size * 1.3,
                                fc='lightgreen', ec='none', 
                                alpha=0.25 * energy_ratio
                            ))
                    
                    self.ax_world.add_patch(Circle(
                        obj.pos, display_size, fc=obj.color, ec='white',
                        alpha=0.9 * light_level, lw=1
                    ))
        
        # Render nearby creatures
        for i, creature in enumerate(alive):
            dist = np.linalg.norm(creature.pos - center)
            if dist < view_radius * 1.5:
                is_selected = (i == self.selected_idx)
                render_creature(
                    self.ax_world, creature,
                    selected=is_selected, light_level=light_level
                )
                if creature.species.name == 'Herbie':
                    render_herbie_special(self.ax_world, creature, light_level)
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)
        print("[Visualization] Closed")
    
    def _render_aquatic_visual(self, light_level: float):
        """Render underwater view in main panel - like a fishtank."""
        self.ax_world.clear()
        self.ax_world.set_facecolor('#001428')  # Deep ocean blue
        
        half = self.world_size / 2
        
        # Zoom levels: 0=close (30), 1=medium (60), 2=far (full world)
        zoom_level = getattr(self, '_aquatic_zoom', 1)
        zoom_names = ['CLOSE', 'MEDIUM', 'WORLD']
        
        # Check if we should center on a specific water body
        if hasattr(self, '_aquatic_center') and self._aquatic_center is not None and zoom_level < 2:
            center = self._aquatic_center
            zoom_radii = [30, 60]  # close, medium
            zoom_radius = zoom_radii[zoom_level]
            self.ax_world.set_xlim(center[0] - zoom_radius, center[0] + zoom_radius)
            self.ax_world.set_ylim(center[1] - zoom_radius, center[1] + zoom_radius)
        else:
            # Full world view (zoom level 2 or no center)
            self.ax_world.set_xlim(-half, half)
            self.ax_world.set_ylim(-half, half)
        self.ax_world.set_aspect('equal')
        
        # Title - show which lake and zoom level
        aq = self.manager.aquatic
        lake_info = ""
        if hasattr(self, '_water_body_idx') and hasattr(self, '_water_bodies') and self._water_bodies:
            lake_info = f" | Lake {self._water_body_idx + 1}/{len(self._water_bodies)}"
        
        zoom_names = ['CLOSE', 'MEDIUM', 'WORLD']
        zoom_name = zoom_names[getattr(self, '_aquatic_zoom', 1)]
        
        self.ax_world.set_title(
            f'AQUATIC [{zoom_name}]{lake_info} | {len(aq.plants)} plants | {len(aq.creatures)} fish | +/- zoom, W cycle lakes',
            color='cyan', fontsize=10
        )
        
        # Draw water terrain zones as lighter areas
        if self.terrain:
            for x in np.linspace(-half + 2, half - 2, 20):
                for y in np.linspace(-half + 2, half - 2, 20):
                    pos = np.array([x, y])
                    t_type = self.terrain.get_terrain_at(pos)
                    if t_type.name in ['water', 'shore']:
                        alpha = 0.3 if t_type.name == 'water' else 0.15
                        self.ax_world.add_patch(Circle(
                            pos, 4, fc='#003366', ec='none', alpha=alpha
                        ))
        
        # Draw aquatic plants with gentle sway
        sway = np.sin(self.step_counter * 0.05) * 0.5
        for plant in aq.plants:
            color = plant.color
            size = plant.size * 1.5
            
            # Kelp sways more
            plant_sway = sway * (1.5 if plant.plant_type == 'kelp' else 0.5)
            
            # Draw plant with slight offset for sway effect
            pos = plant.pos + np.array([plant_sway, 0])
            self.ax_world.add_patch(Circle(
                pos, size, fc=color, ec='white', alpha=0.8, lw=0.5
            ))
            
            # Plant type symbol (ASCII-safe)
            symbols = {'kelp': 'K', 'lily': 'L', 'algae': '.', 'reed': '|'}
            self.ax_world.text(pos[0], pos[1], symbols.get(plant.plant_type, '*'),
                              ha='center', va='center', fontsize=8, color='white')
        
        # Draw fish/creatures with swimming motion based on their heading
        for creature in aq.creatures:
            if not creature.alive:
                continue
            
            # Use actual heading for movement direction
            vel = getattr(creature, 'vel', np.array([1, 0]))
            if np.linalg.norm(vel) > 0.05:
                heading = np.arctan2(vel[1], vel[0])
            else:
                heading = getattr(creature, 'heading', 0.0)
            
            # Swimming wobble perpendicular to heading
            wobble_angle = heading + np.pi/2
            wobble_amt = np.sin(self.step_counter * 0.15 + hash(id(creature)) % 100) * 0.3
            wobble = np.array([np.cos(wobble_angle), np.sin(wobble_angle)]) * wobble_amt
            pos = creature.pos + wobble
            
            species = getattr(creature, 'species', 'minnow')
            
            # Species-specific appearance - more distinct
            if species == 'minnow':
                color = '#C0C0C0'  # Silver
                body_size = 1.2
                label = 'm'
            elif species == 'fish':
                color = '#FFD700'  # Gold
                body_size = 2.0
                label = 'F'
            elif species == 'bass':
                color = '#228B22'  # Forest green
                body_size = 2.5
                label = 'B'
            elif species == 'pike':
                color = '#4682B4'  # Steel blue
                body_size = 3.0
                label = 'P'
            elif species == 'eel':
                color = '#2F4F4F'  # Dark slate
                body_size = 2.0
                label = 'E'
            else:
                color = '#87CEEB'
                body_size = 1.5
                label = '?'
            
            # Draw fish body as elongated ellipse
            from matplotlib.patches import Ellipse
            fish_len = body_size * 2
            fish_width = body_size * 0.8
            ellipse = Ellipse(pos, fish_len, fish_width, angle=np.degrees(heading),
                             fc=color, ec='white', alpha=0.9, lw=1.5)
            self.ax_world.add_patch(ellipse)
            
            # Draw tail (triangle behind)
            tail_offset = -np.array([np.cos(heading), np.sin(heading)]) * fish_len * 0.5
            tail_pos = pos + tail_offset
            perp = np.array([-np.sin(heading), np.cos(heading)])
            tail_points = [
                tail_pos,
                tail_pos + perp * fish_width * 0.6 - np.array([np.cos(heading), np.sin(heading)]) * fish_width,
                tail_pos - perp * fish_width * 0.6 - np.array([np.cos(heading), np.sin(heading)]) * fish_width,
            ]
            from matplotlib.patches import Polygon
            self.ax_world.add_patch(Polygon(tail_points, fc=color, ec='none', alpha=0.7))
            
            # Draw eye to show direction
            eye_offset = np.array([np.cos(heading), np.sin(heading)]) * fish_len * 0.3
            eye_pos = pos + eye_offset
            self.ax_world.add_patch(Circle(
                eye_pos, body_size * 0.25, fc='black', ec='white', lw=0.5
            ))
            
            # Label with species and energy
            energy = getattr(creature, 'energy', 0)
            self.ax_world.text(pos[0], pos[1] + body_size + 0.8, 
                              f'{label} E:{energy:.0f}',
                              color='white', fontsize=7, ha='center',
                              fontweight='bold', alpha=0.9)
        
        # Draw poop/fertilizer deposits with glow effect
        for poop in aq.poops:
            # Use aquatic system's step count for consistency
            age = aq.step_count - poop.get('birth_step', 0)
            decay_time = aq.poop_decay_time
            fade = max(0, min(1.0, 1.0 - age / decay_time))  # Clamp to [0, 1]
            
            if fade > 0:
                pos = poop['pos']
                amount = poop.get('amount', 5)
                
                # Size based on amount
                base_size = 0.3 + 0.2 * min(amount / 10, 1.0)
                
                # Glow (outer) - clamp alpha
                glow_size = base_size * (1.5 + 0.5 * fade)
                self.ax_world.add_patch(Circle(
                    pos, glow_size, fc='#8B4513', ec='none', alpha=min(1.0, 0.2 * fade)
                ))
                
                # Core - clamp alpha
                self.ax_world.add_patch(Circle(
                    pos, base_size, fc='#654321', ec='none', alpha=min(1.0, 0.6 * fade)
                ))
                
                # Nitrogen sparkles as it decomposes
                if age > 30 and np.random.random() < 0.3:
                    sparkle_offset = np.random.randn(2) * base_size
                    self.ax_world.plot(pos[0] + sparkle_offset[0], pos[1] + sparkle_offset[1],
                                      '*', color='lime', alpha=min(1.0, 0.5 * fade), ms=4)
        
        # Ambient bubbles
        for _ in range(5):
            bx = np.random.uniform(-half + 5, half - 5)
            by = np.random.uniform(-half + 5, half - 5)
            self.ax_world.plot(bx, by, 'o', color='white', alpha=0.3, ms=np.random.randint(2, 5))
    
    def _render_fungal_visual(self, light_level: float):
        """Render mycorrhizal network view - underground fungal web."""
        self.ax_world.clear()
        self.ax_world.set_facecolor('#1a0a00')  # Dark earth brown
        
        half = self.world_size / 2
        self.ax_world.set_xlim(-half, half)
        self.ax_world.set_ylim(-half, half)
        self.ax_world.set_aspect('equal')
        
        myc = self.manager.mycelia
        
        # Title
        density_pct = np.mean(myc.density) * 100
        self.ax_world.set_title(
            f'^^ FUNGAL NETWORK | Density: {density_pct:.1f}% | Mushrooms: {len(myc.fruiting_bodies)}',
            color='#DDA0DD', fontsize=11
        )
        
        # Render network density as glowing web
        res = myc.resolution
        cell_size = self.world_size / res
        
        for gy in range(res):
            for gx in range(res):
                d = myc.density[gy, gx]
                if d > 0.02:
                    # Convert grid to world coords
                    wx = (gx - res/2) * cell_size + cell_size/2
                    wy = (gy - res/2) * cell_size + cell_size/2
                    
                    # Pulsing glow based on density
                    pulse = 0.5 + 0.5 * np.sin(self.step_counter * 0.03 + gx * 0.1 + gy * 0.1)
                    alpha = min(0.8, d * pulse * 1.5)
                    
                    # Color shifts with density - blue -> purple -> white
                    if d > 0.5:
                        color = '#DDA0DD'  # Plum
                    elif d > 0.2:
                        color = '#9370DB'  # Medium purple
                    else:
                        color = '#4B0082'  # Indigo
                    
                    self.ax_world.add_patch(Rectangle(
                        (wx - cell_size/2, wy - cell_size/2), cell_size, cell_size,
                        fc=color, ec='none', alpha=alpha
                    ))
        
        # Draw signal propagation (stress signals) as bright flashes
        for gy in range(res):
            for gx in range(res):
                s = myc.signal[gy, gx]
                if s > 0.1:
                    wx = (gx - res/2) * cell_size + cell_size/2
                    wy = (gy - res/2) * cell_size + cell_size/2
                    self.ax_world.add_patch(Circle(
                        (wx, wy), cell_size * 0.8, fc='yellow', ec='none', alpha=s * 0.5
                    ))
        
        # Draw fruiting bodies (mushrooms) 
        for mushroom in myc.fruiting_bodies:
            size = mushroom.size * 3
            
            # Mushroom cap color based on age
            if mushroom.spore_ready:
                cap_color = '#8B4513'  # Saddle brown (mature)
                glow = '#FFD700'  # Gold spore glow
            else:
                cap_color = '#CD853F'  # Peru (growing)
                glow = None
            
            # Stem
            self.ax_world.add_patch(Rectangle(
                (mushroom.pos[0] - size*0.15, mushroom.pos[1] - size*0.5),
                size*0.3, size*0.5, fc='#F5DEB3', ec='none'
            ))
            
            # Cap
            self.ax_world.add_patch(Circle(
                mushroom.pos, size*0.5, fc=cap_color, ec='white', lw=1
            ))
            
            # Spore glow if ready
            if glow:
                self.ax_world.add_patch(Circle(
                    mushroom.pos, size*0.7, fc=glow, ec='none', alpha=0.3
                ))
            
            # Draw mushroom as colored circle instead of emoji
            self.ax_world.add_patch(Circle(
                mushroom.pos, size*0.4, fc='#DDA0DD', ec='#8B668B', lw=1
            ))
    
    def _render_resonance_field(self, field, light_level: float):
        """
        Render the global Schumann resonance field as psychedelic overlay.
        
        Shows amplitude as color intensity and phase as hue.
        Creates a trippy visualization of the global EM field coupling.
        """
        import matplotlib.colors as mcolors
        
        # Get field data
        amplitude = field.get_amplitude_field()
        phase = field.get_phase_field()
        
        # Normalize amplitude for display
        amp_max = np.max(amplitude) + 0.001
        amp_norm = amplitude / amp_max
        
        # Create RGB image from phase (hue) and amplitude (saturation/value)
        # Phase -> Hue (0-1), Amplitude -> Saturation
        h = (phase + np.pi) / (2 * np.pi)  # Map -pi,pi to 0,1
        s = np.clip(amp_norm * 2, 0, 1)
        v = np.clip(0.3 + amp_norm * 0.7, 0, 1)
        
        # Convert HSV to RGB
        hsv = np.stack([h, s, v], axis=-1)
        rgb = mcolors.hsv_to_rgb(hsv)
        
        # Display as semi-transparent overlay
        half = self.world_size / 2
        extent = [-half, half, -half, half]
        
        # Add alpha channel based on amplitude (low amplitude = more transparent)
        alpha = np.clip(amp_norm * 0.6 * light_level, 0, 0.5)
        rgba = np.dstack([rgb, alpha])
        
        self.ax_world.imshow(
            rgba, extent=extent, origin='lower',
            interpolation='bilinear', zorder=1
        )
        
        # Add Schumann mode indicators in corner
        stats = field.get_statistics()
        mode_text = f"Sch: E={stats['total_energy']:.1f} C={stats['coherence']:.2f}"
        self.ax_world.text(
            -half + 2, half - 3, mode_text,
            color='cyan', fontsize=7, alpha=0.7 * light_level
        )
    
    def _render_ghosts(self, ghost_field, light_level: float):
        """
        Render ghosts/spirits as ethereal wisps.
        
        Shows wandering torus wavefunctions of the departed.
        Higher coherence = more visible/solid appearance.
        Circulation direction shown via spiral trails.
        """
        from matplotlib.patches import Circle, FancyArrow
        
        for ghost in ghost_field.ghosts:
            if not ghost.alive:
                continue
            
            pos = ghost.pos
            coherence = ghost._compute_coherence()
            circulation = ghost.get_circulation()
            energy = ghost.get_energy()
            
            # Base visibility scales with coherence
            base_alpha = min(0.8, 0.2 + coherence * 0.6) * light_level
            
            # Outer ethereal glow - pale blue/white
            glow_size = 2.0 + energy * 0.02
            self.ax_world.add_patch(Circle(
                pos, glow_size, fc='#b0c4de', ec='none', 
                alpha=min(1.0, base_alpha * 0.3)
            ))
            
            # Middle wispy layer - ghostly cyan
            self.ax_world.add_patch(Circle(
                pos, glow_size * 0.6, fc='#e0ffff', ec='none',
                alpha=min(1.0, base_alpha * 0.5)
            ))
            
            # Core - bright white for coherent, dim for dispersing
            core_alpha = base_alpha * (0.3 + coherence * 0.7)
            self.ax_world.add_patch(Circle(
                pos, glow_size * 0.25, fc='white', ec='none',
                alpha=min(1.0, core_alpha)
            ))
            
            # Circulation indicator - spiral direction
            if abs(circulation) > 0.1:
                # Small arrow showing drift direction
                angle = ghost.age * circulation * 0.1
                dx = np.cos(angle) * 1.5
                dy = np.sin(angle) * 1.5
                self.ax_world.arrow(
                    pos[0], pos[1], dx, dy,
                    head_width=0.4, head_length=0.2,
                    fc='#87ceeb', ec='none', alpha=min(1.0, base_alpha * 0.6)
                )
            
            # Name label for named ghosts
            if ghost.name and coherence > 0.3:
                self.ax_world.text(
                    pos[0], pos[1] + glow_size + 0.5, f"✦ {ghost.name}",
                    ha='center', va='bottom', fontsize=7,
                    color='#b0c4de', alpha=min(1.0, base_alpha),
                    style='italic'
                )
        
        # Ghost count indicator
        n_ghosts = len([g for g in ghost_field.ghosts if g.alive])
        if n_ghosts > 0:
            half = self.world_size / 2
            self.ax_world.text(
                half - 2, half - 3, f"✦ {n_ghosts} spirits",
                ha='right', color='#b0c4de', fontsize=8, 
                alpha=0.7 * light_level, style='italic'
            )
    
    def _render_ghost_cam(self, light_level: float):
        """
        Ghost Cam - follow a specific ghost with zoomed view.
        
        Shows the spirit's perspective with their wavefunction visualization
        and nearby living creatures as heat signatures.
        """
        from matplotlib.patches import Circle
        
        self.ax_world.clear()
        
        # Dark ethereal background
        self.ax_world.set_facecolor('#0a0a1a')
        
        manager = self.manager
        half = self.world_size / 2
        
        if not hasattr(manager, 'ghost_field'):
            self.ax_world.text(0, 0, "No spirit realm...", ha='center', color='gray')
            self.ax_world.set_xlim(-half, half)
            self.ax_world.set_ylim(-half, half)
            return
        
        ghosts = [g for g in manager.ghost_field.ghosts if g.alive]
        if not ghosts:
            self.ax_world.text(0, 0, "All spirits have dispersed...", ha='center', color='#b0c4de')
            self.ax_world.set_xlim(-half, half)
            self.ax_world.set_ylim(-half, half)
            self.ghost_cam = False
            return
        
        # Get the followed ghost
        self.ghost_cam_idx = self.ghost_cam_idx % len(ghosts)
        ghost = ghosts[self.ghost_cam_idx]
        pos = ghost.pos
        coherence = ghost._compute_coherence()
        circulation = ghost.get_circulation()
        energy = ghost.get_energy()
        
        # View centered on ghost with zoom
        view_radius = 25.0
        self.ax_world.set_xlim(pos[0] - view_radius, pos[0] + view_radius)
        self.ax_world.set_ylim(pos[1] - view_radius, pos[1] + view_radius)
        
        # Draw terrain as faint underlay
        if self.terrain:
            terrain_img = render_terrain_background(
                self.ax_world, self.terrain, self.world_size, light_level * 0.3
            )
            self.ax_world.imshow(
                terrain_img, origin='lower',
                extent=[-half, half, -half, half],
                alpha=0.2
            )
        
        # Draw other ghosts as distant wisps
        for other in ghosts:
            if other is ghost:
                continue
            opos = other.pos
            ocoh = other._compute_coherence()
            dist = np.linalg.norm(opos - pos)
            if dist < view_radius * 2:
                self.ax_world.add_patch(Circle(
                    opos, 1.5 + ocoh, fc='#4a5568', ec='none',
                    alpha=0.3 * ocoh
                ))
                if other.name:
                    self.ax_world.text(opos[0], opos[1] + 2, other.name,
                                      ha='center', fontsize=6, color='#718096', alpha=0.5)
        
        # Draw living creatures as "heat signatures" 
        for creature in manager.creatures:
            if not creature.alive:
                continue
            cpos = creature.pos
            dist = np.linalg.norm(cpos - pos)
            if dist < view_radius * 1.5:
                # Living beings appear as warm glows to spirits
                warmth = 0.3 + 0.7 * (1 - creature.metabolism.hunger) if hasattr(creature, 'metabolism') else 0.5
                size = 1.5 + warmth
                self.ax_world.add_patch(Circle(
                    cpos, size * 1.5, fc='#ff6b35', ec='none', alpha=0.15 * warmth
                ))
                self.ax_world.add_patch(Circle(
                    cpos, size, fc='#ffa500', ec='none', alpha=0.3 * warmth
                ))
                # Name if Herbie
                if creature.species.name == 'Herbie' and hasattr(creature, 'mating_state'):
                    self.ax_world.text(cpos[0], cpos[1] - 2, creature.mating_state.name,
                                      ha='center', fontsize=6, color='#ffa500', alpha=0.6)
        
        # Draw THE followed ghost prominently
        # Outer glow
        glow_size = 4.0 + energy * 0.05
        for i in range(5):
            r = glow_size * (1 - i * 0.15)
            alpha = 0.1 + i * 0.1
            self.ax_world.add_patch(Circle(
                pos, r, fc='#e0ffff', ec='none', alpha=alpha * coherence
            ))
        
        # Core with pulsing
        pulse = 0.8 + 0.2 * np.sin(self.step_counter * 0.1)
        self.ax_world.add_patch(Circle(
            pos, 1.5 * pulse, fc='white', ec='#87ceeb', alpha=0.9, lw=2
        ))
        
        # Circulation spiral trail
        if abs(circulation) > 0.05:
            trail_points = []
            for i in range(20):
                t = i * 0.3
                angle = ghost.age * circulation * 0.1 - t * circulation
                r = 2.0 + t * 0.3
                tx = pos[0] + r * np.cos(angle)
                ty = pos[1] + r * np.sin(angle)
                trail_points.append([tx, ty])
            trail = np.array(trail_points)
            for i in range(len(trail) - 1):
                alpha = 0.5 * (1 - i / len(trail)) * coherence
                self.ax_world.plot([trail[i, 0], trail[i+1, 0]], 
                                  [trail[i, 1], trail[i+1, 1]],
                                  color='#87ceeb', alpha=alpha, lw=1)
        
        # Wavefunction mini-display in corner
        psi = ghost.psi
        intensity = np.abs(psi)**2
        phase = np.angle(psi)
        
        # Status text
        name = ghost.name or f"Spirit #{self.ghost_cam_idx}"
        status = f"✦ GHOST CAM: {name}"
        self.ax_world.text(
            pos[0], pos[1] + view_radius - 3, status,
            ha='center', fontsize=10, color='#e0ffff', 
            fontweight='bold', alpha=0.9
        )
        
        stats = f"Coherence: {coherence:.2f} | Circulation: {circulation:.2f} | Age: {ghost.age}"
        self.ax_world.text(
            pos[0], pos[1] + view_radius - 5, stats,
            ha='center', fontsize=8, color='#b0c4de', alpha=0.7
        )
        
        # Navigation hint
        self.ax_world.text(
            pos[0], pos[1] - view_radius + 2, 
            "< > cycle spirits | Shift+O exit",
            ha='center', fontsize=7, color='#718096', alpha=0.5
        )
    
    def _show_aquatic_view(self):
        """Display aquatic ecosystem - switches to underwater visual mode."""
        if not hasattr(self.manager, 'aquatic'):
            print("  No aquatic system available")
            return
        
        aq = self.manager.aquatic
        print(f"\n{'='*60}")
        print("   UNDERWATER WORLD ")
        print(f"{'='*60}")
        print(f"  Plants: {len(aq.plants)} | Creatures: {len(aq.creatures)}")
        print()
        
        # Find actual water bodies
        if hasattr(self, 'terrain') and self.terrain:
            water_bodies = self.terrain.find_water_bodies()
            if water_bodies:
                print(f"  Water Bodies: {len(water_bodies)}")
                for i, wb in enumerate(water_bodies[:5]):  # Show first 5
                    cx, cy = wb['center']
                    print(f"    Lake {i+1}: ({cx:.0f}, {cy:.0f}) "
                          f"size={wb['size']} depth={wb['depth']:.2f}")
                
                # Center view on largest water body
                largest = water_bodies[0]
                self._aquatic_center = largest['center']
                print(f"\n  [View centered on largest lake at ({largest['center'][0]:.0f}, {largest['center'][1]:.0f})]")
                print(f"  [Press W to cycle to next water body, W again to exit]")
                
                # Store water bodies for cycling
                self._water_bodies = water_bodies
                self._water_body_idx = 0
            else:
                print("  No significant water bodies found")
                self._aquatic_center = None
        
        # Show plants by type
        plant_types = {}
        for p in aq.plants:
            t = p.plant_type
            plant_types[t] = plant_types.get(t, 0) + 1
        
        print("\n  Flora:")
        for ptype, count in sorted(plant_types.items()):
            emoji = {'kelp': '', 'lily': '', 'algae': '', 'reed': ''}.get(ptype, '')
            print(f"    {emoji} {ptype}: {count}")
        
        # Show creatures
        if aq.creatures:
            print("\n  Fauna:")
            for c in aq.creatures[:5]:  # Show first 5
                species = getattr(c, 'species', 'fish')
                emoji = '' if species in ['fish', 'minnow'] else ''
                print(f"    {emoji} {species} at ({c.pos[0]:.0f}, {c.pos[1]:.0f}) E:{c.energy:.0f}")
        
        print(f"{'='*60}\n")
    
    def _show_fungal_view(self):
        """Display mycorrhizal network status - the underground web."""
        if not hasattr(self.manager, 'mycelia'):
            print("  No mycelia system available")
            return
        
        myc = self.manager.mycelia
        print(f"\n{'='*60}")
        print("   MYCORRHIZAL NETWORK ")
        print(f"{'='*60}")
        
        # Network statistics
        density_mean = np.mean(myc.density)
        density_max = np.max(myc.density)
        signal_active = np.sum(myc.signal > 0.1)
        
        print(f"  Network density: {density_mean:.1%} avg, {density_max:.1%} max")
        print(f"  Active signal nodes: {signal_active}")
        print(f"  Fruiting bodies: {len(myc.fruiting_bodies)}")
        print(f"  Total nutrients transferred: {myc.total_nutrient_transferred:.0f}")
        
        # ASCII art of network density
        print("\n  Network Map (density):")
        print("  " + "-" * 22)
        
        # Downsample to 20x10 for display
        h, w = myc.density.shape
        for y in range(0, h, h//10):
            row = "  |"
            for x in range(0, w, w//20):
                d = myc.density[y, x]
                if d > 0.7:
                    row += "#"
                elif d > 0.4:
                    row += "+"
                elif d > 0.2:
                    row += ":"
                elif d > 0.05:
                    row += "."
                else:
                    row += " "
            row += "|"
            print(row)
        
        print("  " + "-" * 22)
        
        # Show mushrooms
        if myc.fruiting_bodies:
            print(f"\n   Mushrooms ({len(myc.fruiting_bodies)}):")
            for m in myc.fruiting_bodies[:5]:
                status = "~sporing" if m.spore_ready else "~growing"
                print(f"    {status} at ({m.pos[0]:.0f}, {m.pos[1]:.0f}) size:{m.size:.1f}")
        
        print(f"{'='*60}\n")
    
    def _get_pregnant_creatures(self):
        """Get list of creatures with active embryos."""
        pregnant = []
        for c in self.manager.creatures:
            if not c.alive:
                continue
            if hasattr(c, 'mating_state') and c.mating_state:
                if getattr(c.mating_state, 'is_pregnant', False):
                    embryo = getattr(c.mating_state, 'embryo', None)
                    if embryo:
                        pregnant.append(c)
        return pregnant
    
    def _render_embryo_view(self, ax):
        """Render embryo development visualization."""
        ax.clear()
        ax.set_facecolor('#0a0a12')
        
        pregnant = self._get_pregnant_creatures()
        
        if not pregnant:
            ax.text(0.5, 0.5, "No active embryos\n\nWait for mating...",
                   transform=ax.transAxes, ha='center', va='center',
                   color='#446688', fontsize=12, fontfamily='monospace')
            ax.set_title("EMBRYO VIEW (E to exit)", color='#66ccff', fontsize=10)
            return
        
        # Get selected pregnant creature
        idx = getattr(self, 'embryo_view_index', 0) % len(pregnant)
        creature = pregnant[idx]
        embryo = creature.mating_state.embryo
        name = creature.mating_state.name if hasattr(creature, 'mating_state') else creature.creature_id[:8]
        
        # Get embryo field intensity
        field = embryo.get_field_image()
        
        # Render the embryo field
        im = ax.imshow(field, cmap='plasma', origin='lower', 
                       extent=[-3.14, 3.14, -3.14, 3.14],
                       vmin=0, vmax=field.max() + 0.1)
        
        # Draw basin positions as markers
        for bx, by in embryo.basin_positions:
            ax.plot(bx, by, 'w+', markersize=8, markeredgewidth=2)
        
        # Title with info
        stage = embryo.stage.name if hasattr(embryo.stage, 'name') else str(embryo.stage)
        progress = min(100, embryo.dev_step / embryo.total_steps * 100)
        
        # Show if ready to be born
        ready_str = "  READY" if embryo.is_ready() else ""
        gestation_progress = creature.mating_state.gestation_progress
        gestation_total = 400  # HERBIE_GESTATION_DURATION
        birth_progress = gestation_progress / gestation_total * 100
        
        title = (f" {name}'s EMBRYO ({idx+1}/{len(pregnant)})\n"
                f"Stage: {stage}{ready_str} | Dev: {progress:.0f}% | Birth: {birth_progress:.0f}%")
        ax.set_title(title, color='#ff66aa', fontsize=10, fontfamily='monospace')
        
        # Stats text
        stats = (f"Basins: {embryo.n_basins}\n"
                f"Bilateral sym: {embryo.bilateral_symmetry:.2f}\n"
                f"Radial sym: {embryo.radial_symmetry:.2f}\n"
                f"Maternal stress: {embryo.maternal_stress:.2f}\n"
                f"Perturbations: {len([e for e in embryo.events if e.event_type == 'stress_perturbation'])}")
        
        ax.text(0.02, 0.98, stats, transform=ax.transAxes,
               fontsize=8, fontfamily='monospace', color='#aaffaa',
               va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='#000000', alpha=0.7))
        
        # Stage indicator on right
        stages = ['ZYGOTE', 'CLEAVAGE', 'MORULA', 'BLASTULA', 
                  'GASTRULA', 'NEURULA', 'ORGANOGENESIS', 'FETAL', 'READY']
        current_idx = stages.index(stage) if stage in stages else 0
        
        stage_text = ""
        for i, s in enumerate(stages):
            if i == current_idx:
                stage_text += f"-> {s} <-\n"
            elif i < current_idx:
                stage_text += f"   {s}\n"
            else:
                stage_text += f"    {s}\n"
        
        ax.text(0.98, 0.98, stage_text, transform=ax.transAxes,
               fontsize=7, fontfamily='monospace', color='#88aaff',
               va='top', ha='right')
        
        ax.set_xlabel("<-/-> cycle embryos | E exit", color='#666666', fontsize=8)

