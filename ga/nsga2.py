"""
NSGA-II implementation for multi-objective optimization.

Implements:
- Non-dominated sorting (Pareto ranking)
- Crowding distance calculation
- Selection operator (rank + crowding distance)
- Pareto front extraction
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from ga.population import initialize_population
from ga.evaluation import evaluate_population
from ga.mutation import mutate
from ga.crossover import crossover_set_union

# region agent log
BASE_DIR = Path(__file__).resolve().parents[1]
_AGENT_LOG_PATH = BASE_DIR / "logs" / "debug.log"
_AGENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def _agent_log(hypothesis_id: str, location: str, message: str, data: Dict) -> None:
    try:
        payload = {
            "sessionId": "debug-session",
            "runId": "baseline",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# endregion


def dominates(fitness1: Tuple[float, float, float], fitness2: Tuple[float, float, float]) -> bool:
    """
    Check if fitness1 dominates fitness2.
    
    fitness1 dominates fitness2 if:
    - fitness1 is at least as good in all objectives
    - fitness1 is strictly better in at least one objective
    
    All objectives are to be maximized.
    
    Args:
        fitness1: First fitness tuple (difficulty, diversity, balance)
        fitness2: Second fitness tuple
        
    Returns:
        True if fitness1 dominates fitness2
    """
    # All objectives are to maximize
    at_least_as_good = all(f1 >= f2 for f1, f2 in zip(fitness1, fitness2))
    strictly_better = any(f1 > f2 for f1, f2 in zip(fitness1, fitness2))
    
    return at_least_as_good and strictly_better


def non_dominated_sorting(fitnesses: List[Tuple[float, float, float]]) -> List[List[int]]:
    """
    Perform non-dominated sorting.
    
    Assigns each individual to a front (rank) based on Pareto dominance.
    Front 0 contains all non-dominated solutions, front 1 contains solutions
    dominated only by front 0, etc.
    
    Args:
        fitnesses: List of fitness tuples
        
    Returns:
        List of fronts, where each front is a list of individual indices
    """
    n = len(fitnesses)
    fitness_array = np.array(fitnesses)
    
    # Vectorized dominance computation
    # For maximization: fitness1 dominates fitness2 if all >= and any >
    expanded = fitness_array[:, None, :]  # shape (n,1,3)
    expanded_other = fitness_array[None, :, :]  # shape (1,n,3)
    ge_mask = expanded >= expanded_other  # (n,n,3)
    gt_mask = expanded > expanded_other   # (n,n,3)
    
    dominates_mask = np.all(ge_mask, axis=2) & np.any(gt_mask, axis=2)
    
    # domination_count[i] = how many j dominate i
    domination_count = dominates_mask.T.sum(axis=1)
    # dominated_by[i] = indices j that i dominates
    dominated_by = [list(np.flatnonzero(dominates_mask[i])) for i in range(n)]
    
    # Assign to fronts
    fronts = []
    current_front = [i for i in range(n) if domination_count[i] == 0]
    fronts.append(current_front)
    
    # Find subsequent fronts
    while len(fronts[-1]) > 0:
        next_front = []
        for i in fronts[-1]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break
    
    return fronts


def crowding_distance(
    fitnesses: List[Tuple[float, float, float]],
    front_indices: List[int]
) -> np.ndarray:
    """
    Calculate crowding distance for individuals in a front.
    
    Crowding distance measures how close an individual is to its neighbors
    in objective space. Higher distance = more diverse.
    
    Args:
        fitnesses: List of all fitness tuples
        front_indices: Indices of individuals in the current front
        
    Returns:
        Array of crowding distances for individuals in the front
    """
    if len(front_indices) == 0:
        return np.array([])
    
    if len(front_indices) == 1:
        return np.array([np.inf])  # Single individual gets infinite distance
    
    n = len(front_indices)
    distances = np.zeros(n)
    
    # Get fitnesses for this front
    front_fitnesses = np.array([fitnesses[i] for i in front_indices])
    num_objectives = len(front_fitnesses[0])
    
    # Calculate distance for each objective
    for obj_idx in range(num_objectives):
        # Sort by this objective
        sorted_indices = np.argsort(front_fitnesses[:, obj_idx])
        
        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Get objective range
        obj_min = front_fitnesses[sorted_indices[0], obj_idx]
        obj_max = front_fitnesses[sorted_indices[-1], obj_idx]
        obj_range = obj_max - obj_min
        
        # Early-exit: if all values equal, skip accumulation for this objective
        if obj_range == 0:
            continue  # All same value, skip
        
        # Calculate distance for intermediate points
        for i in range(1, n - 1):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            next_idx = sorted_indices[i + 1]
            
            distances[idx] += (front_fitnesses[next_idx, obj_idx] - 
                              front_fitnesses[prev_idx, obj_idx]) / obj_range
    
    return distances


def select_parents(
    population: List[np.ndarray],
    fitnesses: List[Tuple[float, float, float]],
    num_parents: int,
    rng: np.random.Generator = None
) -> List[np.ndarray]:
    """
    Select parents using tournament selection based on rank and crowding distance.
    
    Args:
        population: List of chromosomes
        fitnesses: List of fitness tuples
        num_parents: Number of parents to select
        rng: Random number generator
        
    Returns:
        List of selected parent chromosomes
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Perform non-dominated sorting
    fronts = non_dominated_sorting(fitnesses)
    
    # Calculate crowding distances for all individuals
    crowding_distances = np.zeros(len(population))
    for front_indices in fronts:
        if len(front_indices) > 0:
            front_distances = crowding_distance(fitnesses, front_indices)
            for i, idx in enumerate(front_indices):
                crowding_distances[idx] = front_distances[i]
    
    # Assign ranks (front number)
    ranks = np.zeros(len(population), dtype=int)
    for rank, front_indices in enumerate(fronts):
        for idx in front_indices:
            ranks[idx] = rank
    
    # Tournament selection
    parents = []
    tournament_size = config.TOURNAMENT_SIZE
    
    for _ in range(num_parents):
        # Select tournament participants
        tournament_indices = rng.choice(len(population), size=tournament_size, replace=False)
        
        # Select winner: lower rank is better, if same rank, higher crowding distance is better
        winner_idx = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if ranks[idx] < ranks[winner_idx]:
                winner_idx = idx
            elif ranks[idx] == ranks[winner_idx]:
                if crowding_distances[idx] > crowding_distances[winner_idx]:
                    winner_idx = idx
        
        parents.append(population[winner_idx])
    
    return parents


def get_pareto_front(
    population: List[np.ndarray],
    fitnesses: List[Tuple[float, float, float]]
) -> Tuple[List[np.ndarray], List[Tuple[float, float, float]]]:
    """
    Extract the Pareto front (non-dominated solutions).
    
    Args:
        population: List of chromosomes
        fitnesses: List of fitness tuples
        
    Returns:
        Tuple of (pareto_chromosomes, pareto_fitnesses)
    """
    fronts = non_dominated_sorting(fitnesses)
    
    if len(fronts) == 0:
        return [], []
    
    # First front is the Pareto front
    pareto_indices = fronts[0]
    pareto_chromosomes = [population[i] for i in pareto_indices]
    pareto_fitnesses = [fitnesses[i] for i in pareto_indices]
    
    return pareto_chromosomes, pareto_fitnesses


def run_nsga2(
    k: int,
    pool_size: int,
    population_size: int = None,
    generations: int = None,
    seed: int = None,
    verbose: bool = True
) -> Dict:
    """
    Run NSGA-II algorithm.
    
    Args:
        k: Subset size
        pool_size: Size of selection pool
        population_size: Population size. If None, uses config.GA_POPULATION_SIZE
        generations: Number of generations. If None, uses config.GA_GENERATIONS
        seed: Random seed. If None, uses config.GA_SEED
        verbose: Whether to print progress
        
    Returns:
        Dictionary with final population, fitnesses, Pareto front, and history
    """
    if population_size is None:
        population_size = config.GA_POPULATION_SIZE
    
    if generations is None:
        generations = config.GA_GENERATIONS
    
    if seed is None:
        seed = config.GA_SEED
    
    rng = np.random.default_rng(seed)
    crossover_prob = config.CROSSOVER_PROB
    
    # Initialize population
    if verbose:
        print(f"Initializing population (size={population_size}, k={k})...")
    population = initialize_population(population_size, k, pool_size, seed=seed)
    
    # Evaluate initial population
    if verbose:
        print("Evaluating initial population...")
    fitnesses = evaluate_population(population)
    
    # History tracking
    history = {
        'generation': [],
        'pareto_front_size': [],
        'best_difficulty': [],
        'best_diversity': [],
        'best_balance': []
    }
    
    # Main evolution loop
    for gen in range(generations):
        if verbose:
            print(f"\nGeneration {gen + 1}/{generations}")
        
        # Get Pareto front
        pareto_chromosomes, pareto_fitnesses = get_pareto_front(population, fitnesses)
        
        # Track history
        history['generation'].append(gen)
        history['pareto_front_size'].append(len(pareto_chromosomes))
        if len(pareto_fitnesses) > 0:
            history['best_difficulty'].append(max(f[0] for f in pareto_fitnesses))
            history['best_diversity'].append(max(f[1] for f in pareto_fitnesses))
            history['best_balance'].append(max(f[2] for f in pareto_fitnesses))
        else:
            history['best_difficulty'].append(0.0)
            history['best_diversity'].append(0.0)
            history['best_balance'].append(0.0)
        
        if verbose:
            print(f"  Pareto front size: {len(pareto_chromosomes)}")
            if len(pareto_fitnesses) > 0:
                print(f"  Best difficulty: {history['best_difficulty'][-1]:.4f}")
                print(f"  Best diversity: {history['best_diversity'][-1]:.4f}")
                print(f"  Best balance: {history['best_balance'][-1]:.4f}")
        
        if gen == generations - 1:
            break  # Last generation, no need to create next generation
        
        # Create offspring
        offspring = []
        
        # Select parents
        parents = select_parents(population, fitnesses, population_size, rng=rng)
        
        # Generate offspring through crossover and mutation
        for i in range(0, population_size, 2):
            parent1 = parents[i]
            parent2 = parents[(i + 1) % population_size]  # Wrap around if odd population
            
            # Crossover
            if rng.random() < crossover_prob:
                child1, child2 = crossover_set_union(parent1, parent2, pool_size, rng=rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = mutate(child1, pool_size, rng=rng)
            child2 = mutate(child2, pool_size, rng=rng)
            
            offspring.extend([child1, child2])
        
        # Combine parent and offspring populations
        combined_population = population + offspring
        eval_start = time.time()
        combined_fitnesses = fitnesses + evaluate_population(offspring)
        eval_duration_ms = (time.time() - eval_start) * 1000
        _agent_log(
            hypothesis_id="H1",
            location="ga/nsga2.py:run_nsga2",
            message="offspring evaluation timing",
            data={"gen": gen, "offspring": len(offspring), "duration_ms": eval_duration_ms},
        )
        
        # Select next generation using NSGA-II selection
        # We'll use non-dominated sorting + crowding distance to select best individuals
        nds_start = time.time()
        fronts = non_dominated_sorting(combined_fitnesses)
        nds_duration_ms = (time.time() - nds_start) * 1000
        _agent_log(
            hypothesis_id="H2",
            location="ga/nsga2.py:run_nsga2",
            message="non_dominated_sorting timing",
            data={"gen": gen, "combined_pop": len(combined_population), "duration_ms": nds_duration_ms},
        )
        
        # Calculate crowding distances
        crowding_distances = np.zeros(len(combined_population))
        for front_indices in fronts:
            if len(front_indices) > 0:
                front_distances = crowding_distance(combined_fitnesses, front_indices)
                for i, idx in enumerate(front_indices):
                    crowding_distances[idx] = front_distances[i]
        
        # Assign ranks
        ranks = np.zeros(len(combined_population), dtype=int)
        for rank, front_indices in enumerate(fronts):
            for idx in front_indices:
                ranks[idx] = rank
        
        # Select best individuals (lower rank is better, if same rank, higher crowding distance is better)
        # Create list of (rank, -crowding_distance, index) for sorting
        # We negate crowding_distance because we want to sort ascending, but higher distance is better
        sort_key = [(ranks[i], -crowding_distances[i], i) for i in range(len(combined_population))]
        sort_key.sort()
        
        # Select top population_size individuals
        selected_indices = [idx for _, _, idx in sort_key[:population_size]]
        
        population = [combined_population[i] for i in selected_indices]
        fitnesses = [combined_fitnesses[i] for i in selected_indices]
    
    # Final Pareto front
    pareto_chromosomes, pareto_fitnesses = get_pareto_front(population, fitnesses)
    
    if verbose:
        print("\n✓ NSGA-II completed!")
        print(f"  Final Pareto front size: {len(pareto_chromosomes)}")
    
    return {
        'population': population,
        'fitnesses': fitnesses,
        'pareto_front': {
            'chromosomes': pareto_chromosomes,
            'fitnesses': pareto_fitnesses
        },
        'history': history
    }


if __name__ == "__main__":
    # Test NSGA-II with a small example
    print("Testing NSGA-II...")
    print("(Note: Requires difficulty scores and embeddings to be computed first)")
    
    try:
        from data.load_data import load_selection_pool
        _, labels = load_selection_pool()
        pool_size = len(labels)
        
        # Run small test
        k = 10
        result = run_nsga2(
            k=k,
            pool_size=pool_size,
            population_size=20,
            generations=5,
            seed=42,
            verbose=True
        )
        
        print("\nFinal results:")
        print(f"  Population size: {len(result['population'])}")
        print(f"  Pareto front size: {len(result['pareto_front']['chromosomes'])}")
        print("\nPareto front fitnesses:")
        for i, fitness in enumerate(result['pareto_front']['fitnesses'][:5]):
            print(f"  Solution {i+1}: difficulty={fitness[0]:.4f}, diversity={fitness[1]:.4f}, balance={fitness[2]:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the following commands first:")
        print("  1. python pretrained_committee_models/run_inference.py")
        print("  2. python embeddings/extract_embeddings.py")
