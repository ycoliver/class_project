
import collections
import heapq
import random

def count_inversions(state):
    flat = [x for x in state if x != 0]
    inv = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inv += 1
    return inv

def is_solvable(start, goal):
    return count_inversions(start) % 2 == count_inversions(goal) % 2

def get_neighbors(state):
    neighbors = []
    idx = state.index(0)
    r, c = divmod(idx, 3)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            n_idx = nr * 3 + nc
            new_state = list(state)
            new_state[idx], new_state[n_idx] = new_state[n_idx], new_state[idx]
            neighbors.append(tuple(new_state))
    return neighbors

def bfs(start, goal):
    queue = collections.deque([(start, [])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

def manhattan(state, goal):
    dist = 0
    for i in range(1, 9):
        curr_idx = state.index(i)
        goal_idx = goal.index(i)
        r1, c1 = divmod(curr_idx, 3)
        r2, c2 = divmod(goal_idx, 3)
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist

def a_star(start, goal):
    pq = [(manhattan(start, goal), 0, start, [])]
    visited = {start: 0}
    while pq:
        _, g, current, path = heapq.heappop(pq)
        if current == goal:
            return path
        for neighbor in get_neighbors(current):
            new_g = g + 1
            if neighbor not in visited or new_g < visited[neighbor]:
                visited[neighbor] = new_g
                f = new_g + manhattan(neighbor, goal)
                heapq.heappush(pq, (f, new_g, neighbor, path + [neighbor]))
    return None

if __name__ == "__main__":
    sid = 225040014
    start_fixed = (2, 8, 3, 1, 6, 4, 7, 0, 5)
    goal_fixed = (1, 2, 3, 8, 0, 4, 7, 6, 5)
    
    print("--- Task A1 & B1 ---")
    path_bfs = bfs(start_fixed, goal_fixed)
    path_astar = a_star(start_fixed, goal_fixed)
    print(f"BFS Path Length: {len(path_bfs)}")
    print(f"A* Path Length: {len(path_astar)}")
    
    print("\n--- Task C1 (Parity Test) ---")
    start_parity = (8, 2, 3, 1, 6, 4, 7, 0, 5)
    print(f"Solvable: {is_solvable(start_parity, goal_fixed)}")
    
    print("\n--- Task A2 & B2 (Random) ---")
    random.seed(sid)
    nums = list(range(9))
    random.shuffle(nums)
    s2 = tuple(nums)
    random.shuffle(nums)
    g2 = tuple(nums)
    if not is_solvable(s2, g2):
        s2_l = list(s2)
        s2_l[0], s2_l[1] = s2_l[1], s2_l[0]
        s2 = tuple(s2_l)
    
    print(f"Random Start: {s2}")
    print(f"Random Goal: {g2}")
    path_bfs_2 = bfs(s2, g2)
    print(f"BFS Path Length (A2): {len(path_bfs_2)}")
