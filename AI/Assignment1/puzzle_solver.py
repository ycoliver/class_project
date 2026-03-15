"""
8-Puzzle Problem Solver
Student: Dai Xunlian
Student ID: 225040015 (Odd → DFS required)
"""

import heapq
import random
from collections import deque

# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def state_to_grid(state_str):
    """Convert '283164705' → 3x3 list of ints."""
    return [int(c) for c in state_str]

def grid_to_str(grid):
    return ''.join(str(x) for x in grid)

def print_state(state_str, label=""):
    g = state_to_grid(state_str)
    if label:
        print(f"\n{label}:")
    for i in range(3):
        print(f"  {g[3*i]} {g[3*i+1]} {g[3*i+2]}")

def get_neighbors(state_str):
    """Return list of (new_state_str, move_description)."""
    grid = state_to_grid(state_str)
    pos = grid.index(0)
    row, col = divmod(pos, 3)
    moves = []
    directions = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]
    for dr, dc, name in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_grid = grid[:]
            new_pos = nr * 3 + nc
            new_grid[pos], new_grid[new_pos] = new_grid[new_pos], new_grid[pos]
            moves.append((grid_to_str(new_grid), name))
    return moves

def count_inversions(state_str):
    """Count inversions (ignoring 0) for parity check."""
    tiles = [int(c) for c in state_str if c != '0']
    inv = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inv += 1
    return inv

def same_parity(start, goal):
    return count_inversions(start) % 2 == count_inversions(goal) % 2

# ─────────────────────────────────────────────
# DFS
# ─────────────────────────────────────────────

def dfs(start, goal, max_depth=50):
    """DFS with depth limit and visited set per path to avoid revisit."""
    stack = [(start, [start])]
    visited = {start}
    nodes_expanded = 0

    while stack:
        current, path = stack.pop()
        nodes_expanded += 1

        if current == goal:
            return path, nodes_expanded

        if len(path) >= max_depth:
            continue

        for neighbor, _ in reversed(get_neighbors(current)):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))

    return None, nodes_expanded

# ─────────────────────────────────────────────
# A*
# ─────────────────────────────────────────────

def manhattan(state_str, goal_str):
    goal_pos = {}
    for i, c in enumerate(goal_str):
        goal_pos[int(c)] = (i // 3, i % 3)
    dist = 0
    for i, c in enumerate(state_str):
        val = int(c)
        if val != 0:
            gr, gc = goal_pos[val]
            cr, cc = i // 3, i % 3
            dist += abs(gr - cr) + abs(gc - cc)
    return dist

def astar(start, goal):
    h = manhattan(start, goal)
    # (f, g, state, path)
    heap = [(h, 0, start, [start])]
    visited = {}
    nodes_expanded = 0

    while heap:
        f, g, current, path = heapq.heappop(heap)

        if current in visited and visited[current] <= g:
            continue
        visited[current] = g
        nodes_expanded += 1

        if current == goal:
            return path, nodes_expanded

        for neighbor, _ in get_neighbors(current):
            ng = g + 1
            if neighbor not in visited or visited[neighbor] > ng:
                nh = manhattan(neighbor, goal)
                heapq.heappush(heap, (ng + nh, ng, neighbor, path + [neighbor]))

    return None, nodes_expanded

# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────

def print_path(path, title=""):
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    print(f"Number of moves: {len(path)-1}")
    print(f"\nPath ({len(path)} states):")
    for i, state in enumerate(path):
        label = "START" if i == 0 else ("GOAL" if i == len(path)-1 else f"Step {i}")
        print_state(state, label)

def run_task(label, algo_name, algo_func, start, goal):
    print(f"\n{'#'*60}")
    print(f"  {label}  —  {algo_name}")
    print(f"{'#'*60}")
    print_state(start, "Start State")
    print_state(goal, "Goal State")

    result, nodes = algo_func(start, goal)
    if result is None:
        print("\n  >>> No solution found (unreachable or depth limit exceeded)")
    else:
        print(f"\nNodes expanded: {nodes}")
        print_path(result, f"{algo_name} Solution Path")
    return result

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Fixed puzzle states ──
    START_FIXED = "283164705"
    GOAL_FIXED  = "123804765"

    # ── A1: DFS on fixed states ──
    print("\n" + "="*60)
    print("TASK A1: DFS — Fixed Start & Goal States")
    print("="*60)
    result_a1, nodes_a1 = dfs(START_FIXED, GOAL_FIXED)
    print_state(START_FIXED, "Start")
    print_state(GOAL_FIXED, "Goal")
    if result_a1:
        print(f"\nDFS — Nodes expanded: {nodes_a1}")
        print(f"DFS — Number of moves: {len(result_a1)-1}")
        print("\nPath:")
        for i, s in enumerate(result_a1):
            lbl = "START" if i == 0 else ("GOAL" if i == len(result_a1)-1 else f"Step {i}")
            print_state(s, lbl)

    # ── B1: A* on fixed states ──
    print("\n" + "="*60)
    print("TASK B1: A* — Fixed Start & Goal States")
    print("="*60)
    result_b1, nodes_b1 = astar(START_FIXED, GOAL_FIXED)
    print_state(START_FIXED, "Start")
    print_state(GOAL_FIXED, "Goal")
    if result_b1:
        print(f"\nA* — Nodes expanded: {nodes_b1}")
        print(f"A* — Number of moves: {len(result_b1)-1}")
        print("\nPath:")
        for i, s in enumerate(result_b1):
            lbl = "START" if i == 0 else ("GOAL" if i == len(result_b1)-1 else f"Step {i}")
            print_state(s, lbl)

    # ── C1: Parity test ──
    print("\n" + "="*60)
    print("TASK C1: Parity Test")
    print("="*60)
    swapped_start = "823164705"   # swap first two tiles (2 ↔ 8)
    print(f"Original start: {START_FIXED}  (inversions: {count_inversions(START_FIXED)})")
    print(f"Swapped  start: {swapped_start}  (inversions: {count_inversions(swapped_start)})")
    print(f"Goal state    : {GOAL_FIXED}   (inversions: {count_inversions(GOAL_FIXED)})")
    print(f"Same parity (original vs goal): {same_parity(START_FIXED, GOAL_FIXED)}")
    print(f"Same parity (swapped  vs goal): {same_parity(swapped_start, GOAL_FIXED)}")
    print("\nRunning DFS on swapped start (expect: no solution)...")
    result_c1, nodes_c1 = dfs(swapped_start, GOAL_FIXED)
    if result_c1 is None:
        print("  >>> Confirmed: No solution found — different parity!")
    else:
        print(f"  >>> Unexpected: solution found in {len(result_c1)-1} moves")

    # ── A2 & B2: Random states using student ID as seed ──
    print("\n" + "="*60)
    print("TASK A2 & B2: DFS and A* — Random States (seed=225040015)")
    print("="*60)
    rng = random.Random(225040015)
    tiles = list(range(9))

    rng.shuffle(tiles); rand_start = ''.join(str(t) for t in tiles)
    rng.shuffle(tiles); rand_goal  = ''.join(str(t) for t in tiles)

    print(f"Random Start: {rand_start}")
    print(f"Random Goal : {rand_goal}")
    print(f"Inversions (start): {count_inversions(rand_start)}")
    print(f"Inversions (goal) : {count_inversions(rand_goal)}")

    # Fix parity if needed
    if not same_parity(rand_start, rand_goal):
        print("Different parity — swapping first two tiles of start state...")
        lst = list(rand_start)
        lst[0], lst[1] = lst[1], lst[0]
        rand_start = ''.join(lst)
        print(f"Adjusted Start: {rand_start}")

    print_state(rand_start, "Random Start")
    print_state(rand_goal,  "Random Goal")

    # A2: DFS
    print("\n--- A2: DFS ---")
    result_a2, nodes_a2 = dfs(rand_start, rand_goal)
    if result_a2:
        print(f"DFS — Nodes expanded: {nodes_a2}")
        print(f"DFS — Number of moves: {len(result_a2)-1}")
        print("\nPath:")
        for i, s in enumerate(result_a2):
            lbl = "START" if i == 0 else ("GOAL" if i == len(result_a2)-1 else f"Step {i}")
            print_state(s, lbl)
    else:
        print("  DFS: No solution found within depth limit")

    # B2: A*
    print("\n--- B2: A* ---")
    result_b2, nodes_b2 = astar(rand_start, rand_goal)
    if result_b2:
        print(f"A* — Nodes expanded: {nodes_b2}")
        print(f"A* — Number of moves: {len(result_b2)-1}")
        print("\nPath:")
        for i, s in enumerate(result_b2):
            lbl = "START" if i == 0 else ("GOAL" if i == len(result_b2)-1 else f"Step {i}")
            print_state(s, lbl)
    else:
        print("  A*: No solution found")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"A1 (DFS  Fixed ): {'No solution' if result_a1 is None else str(len(result_a1)-1)+' moves'}")
    print(f"B1 (A*   Fixed ): {'No solution' if result_b1 is None else str(len(result_b1)-1)+' moves'}")
    print(f"C1 (Parity test): {'Confirmed unreachable' if result_c1 is None else 'Unexpected solution'}")
    print(f"A2 (DFS  Random): {'No solution' if result_a2 is None else str(len(result_a2)-1)+' moves'}")
    print(f"B2 (A*   Random): {'No solution' if result_b2 is None else str(len(result_b2)-1)+' moves'}")
