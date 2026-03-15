def check_win(board, player):
    win_lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    return any(all(board[i] == player for i in line) for line in win_lines)

def dfs(board, depth, is_x_turn, leaf_counts):
    # Check for terminal states (leaf nodes)
    if check_win(board, 'X'):
        leaf_counts[depth] += 1
        return
    if check_win(board, 'O'):
        leaf_counts[depth] += 1
        return
    if board.count('-') == 0:
        leaf_counts[depth] += 1
        return

    # Recursive step
    player = 'X' if is_x_turn else 'O'
    for i in range(9):
        if board[i] == '-':
            board[i] = player
            dfs(board, depth + 1, not is_x_turn, leaf_counts)
            board[i] = '-' # Backtrack

if __name__ == "__main__":
    initial_board = ['-'] * 9
    leaf_node_counts = {i: 0 for i in range(10)}
    
    dfs(initial_board, 0, True, leaf_node_counts)
    
    print("Leaf nodes at Level 5:", leaf_node_counts[5])
    print("Leaf nodes at Level 6:", leaf_node_counts[6])
    print("Leaf nodes at Level 9:", leaf_node_counts[9])