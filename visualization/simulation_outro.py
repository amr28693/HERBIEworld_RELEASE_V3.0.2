"""
Simulation Outro - Animated Exit Analytics

A beautiful, terminal-based analytics display that runs after Ctrl+C,
matching the aesthetic of the cosmological launcher intro.

Features:
- Animated population graphs (ASCII sparklines)
- Boolean verdict analysis ("Did life flourish?", "Did ghosts persist?", etc.)
- Key statistics with dramatic reveals
- Memorial wall for notable creatures
- Final verdict with CE theory interpretation

Usage:
    Called automatically on clean exit if user opts in,
    or can be run standalone on saved data.
"""

import numpy as np
import time
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# TERMINAL UTILITIES
# =============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def move_cursor(row: int, col: int):
    """Move cursor to position."""
    print(f"\033[{row};{col}H", end='')

def set_color(color: str):
    """Set terminal color."""
    colors = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'cyan': '\033[96m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'magenta': '\033[95m',
        'blue': '\033[94m',
        'white': '\033[97m',
        'gray': '\033[90m',
    }
    print(colors.get(color, ''), end='')

def print_centered(text: str, width: int = 70):
    """Print text centered."""
    padding = (width - len(text)) // 2
    print(' ' * padding + text)

def print_slow(text: str, delay: float = 0.02):
    """Print text with typewriter effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def sparkline(values: List[float], width: int = 40) -> str:
    """Create ASCII sparkline from values."""
    if not values:
        return '─' * width
    
    # Normalize to 0-7 range for block characters
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        normalized = [4] * len(values)
    else:
        normalized = [int((v - min_v) / (max_v - min_v) * 7) for v in values]
    
    # Resample to width
    if len(normalized) > width:
        step = len(normalized) / width
        resampled = [normalized[int(i * step)] for i in range(width)]
    else:
        resampled = normalized
    
    # Convert to block characters
    blocks = ' ▁▂▃▄▅▆▇█'
    return ''.join(blocks[min(v, 8)] for v in resampled)


# =============================================================================
# VERDICT ANALYSIS
# =============================================================================

@dataclass
class Verdict:
    """A boolean verdict with explanation."""
    question: str
    answer: bool
    explanation: str
    importance: str  # 'critical', 'major', 'minor'


def analyze_verdicts(stats: dict) -> List[Verdict]:
    """Generate boolean verdicts from simulation statistics."""
    verdicts = []
    
    # === POPULATION VERDICTS ===
    
    # Did life persist?
    final_pop = stats.get('final_population', 0)
    peak_pop = stats.get('peak_population', 0)
    verdicts.append(Verdict(
        question="Did life persist to the end?",
        answer=final_pop > 0,
        explanation=f"Final population: {final_pop}" if final_pop > 0 else "Extinction occurred",
        importance='critical'
    ))
    
    # Was there population growth?
    initial_pop = stats.get('initial_population', 10)
    verdicts.append(Verdict(
        question="Did the population grow beyond initial?",
        answer=peak_pop > initial_pop,
        explanation=f"Peak: {peak_pop} vs Initial: {initial_pop}",
        importance='major'
    ))
    
    # === HERBIE VERDICTS ===
    
    herbie_births = stats.get('herbie_births', 0)
    herbie_deaths = stats.get('herbie_deaths', 0)
    
    verdicts.append(Verdict(
        question="Did Herbies reproduce?",
        answer=herbie_births > 0,
        explanation=f"{herbie_births} births recorded" if herbie_births > 0 else "No births occurred",
        importance='critical'
    ))
    
    max_gen = stats.get('max_generation', 0)
    verdicts.append(Verdict(
        question="Did lineages extend beyond 3 generations?",
        answer=max_gen >= 3,
        explanation=f"Max generation: {max_gen}",
        importance='major'
    ))
    
    # === GHOST VERDICTS ===
    
    ghosts_spawned = stats.get('ghosts_spawned', 0)
    ghosts_dispersed = stats.get('ghosts_dispersed', 0)
    longest_ghost = stats.get('longest_ghost_age', 0)
    
    verdicts.append(Verdict(
        question="Did spirits walk the world?",
        answer=ghosts_spawned > 0,
        explanation=f"{ghosts_spawned} spirits emerged from death",
        importance='major'
    ))
    
    verdicts.append(Verdict(
        question="Did any ghost persist > 1000 steps?",
        answer=longest_ghost > 1000,
        explanation=f"Longest: {longest_ghost} steps" if longest_ghost > 0 else "No persistent spirits",
        importance='minor'
    ))
    
    singularities = stats.get('ghost_singularities', 0)
    if ghosts_spawned > 0:
        singularity_rate = singularities / ghosts_spawned
        verdicts.append(Verdict(
            question="Did most ghosts disperse naturally (vs singularity)?",
            answer=singularity_rate < 0.5,
            explanation=f"{singularities}/{ghosts_spawned} collapsed to singularity",
            importance='minor'
        ))
    
    # === ECOLOGY VERDICTS ===
    
    bonds_formed = stats.get('bonds_formed', 0)
    verdicts.append(Verdict(
        question="Did pair bonds form?",
        answer=bonds_formed > 0,
        explanation=f"{bonds_formed} bonded pairs" if bonds_formed > 0 else "No bonds formed",
        importance='major'
    ))
    
    # Food/starvation balance
    starvation_deaths = stats.get('starvation_deaths', 0)
    total_deaths = stats.get('total_deaths', 1)
    starvation_rate = starvation_deaths / max(total_deaths, 1)
    verdicts.append(Verdict(
        question="Did fewer than 50% die of starvation?",
        answer=starvation_rate < 0.5,
        explanation=f"{starvation_rate*100:.0f}% died of hunger",
        importance='major'
    ))
    
    # === EMERGENT BEHAVIOR VERDICTS ===
    
    art_created = stats.get('smears_created', 0)
    verdicts.append(Verdict(
        question="Did creatures create art?",
        answer=art_created > 0,
        explanation=f"{art_created} pigment smears" if art_created > 0 else "No artistic expression",
        importance='minor'
    ))
    
    constructions = stats.get('constructions_built', 0)
    verdicts.append(Verdict(
        question="Were structures built?",
        answer=constructions > 0,
        explanation=f"{constructions} constructions" if constructions > 0 else "No building activity",
        importance='minor'
    ))
    
    # === COSMIC VERDICTS ===
    
    years = stats.get('years_elapsed', 0)
    verdicts.append(Verdict(
        question="Did the simulation span multiple years?",
        answer=years >= 2,
        explanation=f"{years} year(s) elapsed",
        importance='minor'
    ))
    
    return verdicts


def compute_final_score(verdicts: List[Verdict]) -> Tuple[int, int, str]:
    """Compute overall score and verdict."""
    weights = {'critical': 3, 'major': 2, 'minor': 1}
    
    total_possible = sum(weights[v.importance] for v in verdicts)
    total_earned = sum(weights[v.importance] for v in verdicts if v.answer)
    
    percentage = (total_earned / total_possible) * 100 if total_possible > 0 else 0
    
    if percentage >= 80:
        verdict = "FLOURISHING WORLD"
    elif percentage >= 60:
        verdict = "VIABLE ECOSYSTEM"
    elif percentage >= 40:
        verdict = "STRUGGLING BIOSPHERE"
    elif percentage >= 20:
        verdict = "COLLAPSE IN PROGRESS"
    else:
        verdict = "EXTINCTION EVENT"
    
    return total_earned, total_possible, verdict


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def display_header():
    """Display the outro header."""
    clear_screen()
    set_color('cyan')
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  ███████╗██╗███╗   ██╗ █████╗ ██╗         ".center(68) + "║")
    print("║" + "  ██╔════╝██║████╗  ██║██╔══██╗██║         ".center(68) + "║")
    print("║" + "  █████╗  ██║██╔██╗ ██║███████║██║         ".center(68) + "║")
    print("║" + "  ██╔══╝  ██║██║╚██╗██║██╔══██║██║         ".center(68) + "║")
    print("║" + "  ██║     ██║██║ ╚████║██║  ██║███████╗    ".center(68) + "║")
    print("║" + "  ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝    ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "A N A L Y S I S".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    set_color('reset')
    print()
    time.sleep(0.5)


def display_population_graph(population_history: List[int]):
    """Display animated population sparkline."""
    set_color('yellow')
    print("  ┌" + "─" * 50 + "┐")
    print("  │" + " POPULATION OVER TIME ".center(50) + "│")
    print("  │" + " " * 50 + "│")
    
    # Animate the sparkline
    spark = sparkline(population_history, 48)
    print("  │ ", end='')
    set_color('green')
    for i, char in enumerate(spark):
        print(char, end='', flush=True)
        if i % 4 == 0:
            time.sleep(0.02)
    set_color('yellow')
    print(" │")
    
    print("  │" + " " * 50 + "│")
    
    # Stats line
    if population_history:
        min_p, max_p = min(population_history), max(population_history)
        final_p = population_history[-1]
        stats_line = f"Min: {min_p}  Peak: {max_p}  Final: {final_p}"
    else:
        stats_line = "No data"
    
    set_color('gray')
    print("  │" + stats_line.center(50) + "│")
    set_color('yellow')
    print("  └" + "─" * 50 + "┘")
    set_color('reset')
    print()
    time.sleep(0.3)


def display_verdicts(verdicts: List[Verdict]):
    """Display verdicts with dramatic reveals."""
    set_color('cyan')
    print("  ╔" + "═" * 50 + "╗")
    print("  ║" + " SIMULATION VERDICTS ".center(50) + "║")
    print("  ╠" + "═" * 50 + "╣")
    set_color('reset')
    
    for v in verdicts:
        # Question
        set_color('white')
        q_display = v.question[:46] + "..." if len(v.question) > 48 else v.question
        print(f"  ║ {q_display:<48} ║")
        
        # Dramatic pause
        time.sleep(0.15)
        
        # Answer with color
        if v.answer:
            set_color('green')
            symbol = "✓ YES"
        else:
            set_color('red')
            symbol = "✗ NO "
        
        # Importance indicator
        importance_char = {'critical': '◆', 'major': '●', 'minor': '○'}[v.importance]
        
        answer_line = f"  {importance_char} {symbol}: {v.explanation}"
        answer_display = answer_line[:50] if len(answer_line) > 50 else answer_line
        print(f"  ║{answer_display:<50}║")
        
        set_color('cyan')
        print("  ╟" + "─" * 50 + "╢")
        set_color('reset')
        time.sleep(0.1)
    
    set_color('cyan')
    print("  ╚" + "═" * 50 + "╝")
    set_color('reset')
    print()


def display_memorial(notable_creatures: List[dict]):
    """Display memorial wall for notable creatures."""
    if not notable_creatures:
        return
    
    set_color('magenta')
    print("  ┌" + "─" * 50 + "┐")
    print("  │" + " ✦ MEMORIAL WALL ✦ ".center(50) + "│")
    print("  ├" + "─" * 50 + "┤")
    set_color('reset')
    
    for creature in notable_creatures[:8]:  # Limit to 8
        name = creature.get('name', 'Unknown')
        achievement = creature.get('achievement', '')
        
        set_color('white')
        line = f"  {name}: {achievement}"
        print(f"  │ {line[:48]:<48} │")
        time.sleep(0.1)
    
    set_color('magenta')
    print("  └" + "─" * 50 + "┘")
    set_color('reset')
    print()


def display_final_verdict(score: int, total: int, verdict: str):
    """Display the final verdict with dramatic effect."""
    percentage = (score / total) * 100 if total > 0 else 0
    
    print()
    set_color('bold')
    print_centered("━" * 40)
    print()
    
    # Dramatic countdown
    set_color('gray')
    print_centered("Computing final analysis...")
    time.sleep(0.5)
    
    for i in range(3, 0, -1):
        print_centered(f"{'.' * i}")
        time.sleep(0.3)
    
    print()
    
    # Verdict color based on result
    if percentage >= 60:
        set_color('green')
    elif percentage >= 40:
        set_color('yellow')
    else:
        set_color('red')
    
    print_centered("╔" + "═" * 36 + "╗")
    print_centered("║" + " " * 36 + "║")
    print_centered("║" + verdict.center(36) + "║")
    print_centered("║" + " " * 36 + "║")
    print_centered("║" + f"Score: {score}/{total} ({percentage:.0f}%)".center(36) + "║")
    print_centered("║" + " " * 36 + "║")
    print_centered("╚" + "═" * 36 + "╝")
    
    set_color('reset')
    print()
    
    # CE Theory interpretation
    set_color('gray')
    if percentage >= 70:
        print_centered("Information successfully collapsed into stable structure.")
        print_centered("The morphospace attractor was reached.")
    elif percentage >= 40:
        print_centered("Partial collapse achieved. Some basins stabilized.")
        print_centered("Entropy and order found temporary balance.")
    else:
        print_centered("Entropy dominated. Structure could not persist.")
        print_centered("The wavefunction dispersed back into noise.")
    
    set_color('reset')
    print()


def display_ghost_summary(ghost_stats: dict):
    """Display ghost statistics with ethereal styling."""
    if not ghost_stats or ghost_stats.get('total_spawned', 0) == 0:
        return
    
    set_color('blue')
    print("  ┌" + "─" * 50 + "┐")
    print("  │" + " ✦ THE SPIRIT REALM ✦ ".center(50) + "│")
    print("  ├" + "─" * 50 + "┤")
    
    spawned = ghost_stats.get('total_spawned', 0)
    dispersed = ghost_stats.get('total_dispersed', 0)
    singularities = ghost_stats.get('total_singularities', 0)
    longest_name = ghost_stats.get('longest_lived_name', 'unnamed')
    longest_age = ghost_stats.get('longest_lived_age', 0)
    
    set_color('cyan')
    print(f"  │ {'Spirits emerged:':<25} {spawned:>22} │")
    print(f"  │ {'Peacefully dispersed:':<25} {dispersed:>22} │")
    print(f"  │ {'Collapsed to singularity:':<25} {singularities:>22} │")
    
    if longest_age > 0:
        longest_str = f"{longest_name} ({longest_age:,} steps)"
        print(f"  │ {'Most persistent:':<25} {longest_str:>22} │")
    
    set_color('blue')
    print("  └" + "─" * 50 + "┘")
    set_color('reset')
    print()


# =============================================================================
# MAIN OUTRO FUNCTION
# =============================================================================

def run_outro(stats: dict, population_history: List[int] = None,
              ghost_stats: dict = None, notable_creatures: List[dict] = None,
              skip_animation: bool = False):
    """
    Run the full outro sequence.
    
    Args:
        stats: Dictionary of simulation statistics
        population_history: List of population counts over time
        ghost_stats: Ghost field statistics
        notable_creatures: List of notable creature dicts
        skip_animation: If True, display without delays
    """
    global_delay = 0.0 if skip_animation else 1.0
    
    # Header
    display_header()
    time.sleep(0.3 * global_delay)
    
    # Population graph
    if population_history:
        display_population_graph(population_history)
        time.sleep(0.3 * global_delay)
    
    # Ghost summary
    if ghost_stats:
        display_ghost_summary(ghost_stats)
        time.sleep(0.3 * global_delay)
    
    # Verdicts
    verdicts = analyze_verdicts(stats)
    display_verdicts(verdicts)
    time.sleep(0.3 * global_delay)
    
    # Memorial
    if notable_creatures:
        display_memorial(notable_creatures)
        time.sleep(0.3 * global_delay)
    
    # Final verdict
    score, total, verdict = compute_final_score(verdicts)
    display_final_verdict(score, total, verdict)
    
    # Exit prompt
    print()
    set_color('gray')
    print_centered("Press Enter to exit...")
    set_color('reset')
    
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass


def prompt_for_outro() -> bool:
    """Ask user if they want to see the outro."""
    print()
    set_color('cyan')
    print("╔" + "═" * 50 + "╗")
    print("║" + " " * 50 + "║")
    print("║" + "View simulation analytics? (y/n)".center(50) + "║")
    print("║" + " " * 50 + "║")
    print("╚" + "═" * 50 + "╝")
    set_color('reset')
    
    try:
        response = input("  > ").strip().lower()
        return response in ('y', 'yes', '')
    except (KeyboardInterrupt, EOFError):
        return False


# =============================================================================
# STATISTICS EXTRACTION HELPERS
# =============================================================================

def extract_stats_from_manager(manager) -> dict:
    """Extract statistics dictionary from creature manager."""
    stats = {}
    
    # Population
    alive = [c for c in manager.creatures if c.alive]
    stats['final_population'] = len(alive)
    stats['initial_population'] = 10  # Default
    
    # From world history if available
    try:
        from ..events.world_history import world_history
        wh = world_history()
        stats['peak_population'] = wh.max_population
        stats['max_generation'] = wh.max_generation
        stats['herbie_births'] = len([b for b in wh.births if b.get('species') == 'Herbie'])
        stats['herbie_deaths'] = len([d for d in wh.deaths if d.get('species') == 'Herbie'])
        stats['bonds_formed'] = len(wh.bonds)
        stats['total_deaths'] = len(wh.deaths)
        
        # Death causes
        stats['starvation_deaths'] = wh.death_causes.get('starvation', 0)
        
        # Time
        if wh.seasons:
            stats['years_elapsed'] = max(s['year'] for s in wh.seasons)
    except:
        stats['peak_population'] = len(manager.creatures)
        stats['max_generation'] = 0
        stats['herbie_births'] = 0
        stats['herbie_deaths'] = 0
        stats['bonds_formed'] = 0
        stats['total_deaths'] = 0
        stats['starvation_deaths'] = 0
        stats['years_elapsed'] = 0
    
    # Emergent behaviors
    if hasattr(manager, 'smears'):
        stats['smears_created'] = len(manager.smears.marks)
    else:
        stats['smears_created'] = 0
    
    if hasattr(manager, 'element_spawner') and manager.element_spawner:
        stats['constructions_built'] = len(manager.element_spawner.constructions)
    else:
        stats['constructions_built'] = 0
    
    return stats


def extract_population_history(manager) -> List[int]:
    """Extract population history from world history."""
    try:
        from ..events.world_history import world_history
        wh = world_history()
        return [p['total'] for p in wh.population_snapshots]
    except:
        return []


def extract_ghost_stats(manager) -> dict:
    """Extract ghost field statistics."""
    if hasattr(manager, 'ghost_field') and manager.ghost_field:
        return manager.ghost_field.get_statistics()
    return {}


def extract_notable_creatures(manager) -> List[dict]:
    """Extract notable creatures for memorial."""
    notable = []
    
    try:
        from ..events.world_history import world_history
        wh = world_history()
        
        # Longest lived
        if wh.oldest_lived:
            notable.append({
                'name': wh.oldest_lived['name'],
                'achievement': f"Lived {wh.oldest_lived['age']:,} steps"
            })
        
        # Most prolific
        if wh.prolific_parents:
            top = max(wh.prolific_parents.items(), key=lambda x: x[1])
            notable.append({
                'name': top[0],
                'achievement': f"Parent of {top[1]} children"
            })
        
        # First death
        if wh.deaths:
            first = wh.deaths[0]
            if first.get('name'):
                notable.append({
                    'name': first['name'],
                    'achievement': "First to fall"
                })
    except:
        pass
    
    return notable


# =============================================================================
# INTEGRATION
# =============================================================================

def run_outro_from_manager(manager, skip_prompt: bool = False):
    """
    Run outro directly from creature manager.
    
    Call this on Ctrl+C or normal exit.
    """
    if not skip_prompt:
        if not prompt_for_outro():
            print("\n  Goodbye!\n")
            return
    
    stats = extract_stats_from_manager(manager)
    pop_history = extract_population_history(manager)
    ghost_stats = extract_ghost_stats(manager)
    notable = extract_notable_creatures(manager)
    
    # Add ghost stats to main stats for verdicts
    if ghost_stats:
        stats['ghosts_spawned'] = ghost_stats.get('total_spawned', 0)
        stats['ghosts_dispersed'] = ghost_stats.get('total_dispersed', 0)
        stats['ghost_singularities'] = ghost_stats.get('total_singularities', 0)
        stats['longest_ghost_age'] = ghost_stats.get('longest_lived_age', 0)
    
    run_outro(stats, pop_history, ghost_stats, notable)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == '__main__':
    # Demo with fake data
    demo_stats = {
        'final_population': 15,
        'initial_population': 10,
        'peak_population': 42,
        'max_generation': 5,
        'herbie_births': 28,
        'herbie_deaths': 23,
        'bonds_formed': 8,
        'total_deaths': 30,
        'starvation_deaths': 12,
        'smears_created': 4,
        'constructions_built': 2,
        'years_elapsed': 3,
        'ghosts_spawned': 23,
        'ghosts_dispersed': 18,
        'ghost_singularities': 3,
        'longest_ghost_age': 4500,
    }
    
    demo_pop = [10, 12, 15, 18, 22, 28, 35, 42, 38, 32, 28, 24, 20, 18, 15]
    
    demo_ghost = {
        'total_spawned': 23,
        'total_dispersed': 18,
        'total_singularities': 3,
        'longest_lived_name': 'Willow',
        'longest_lived_age': 4500,
    }
    
    demo_notable = [
        {'name': 'Orla', 'achievement': 'Lived 12,453 steps'},
        {'name': 'Pavel', 'achievement': 'Parent of 7 children'},
        {'name': 'Ivy', 'achievement': 'First to fall'},
    ]
    
    run_outro(demo_stats, demo_pop, demo_ghost, demo_notable)
