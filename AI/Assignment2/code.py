import math

# Board is represented as a list of 9 elements: 'X', 'O', or empty ' '
def check_winner(board):
    win_paths = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    for path in win_paths:
        if board[path[0]] == board[path[1]] == board[path[2]] and board[path[0]] != ' ':
            return board[path[0]]
    if ' ' not in board:
        return 'Tie'
    return None

def minimax(board, depth, is_maximizing):
    result = check_winner(board)
    if result == 'X':
        return 1
    elif result == 'O':
        return -1
    elif result == 'Tie':
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(board, depth + 1, False)
                board[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(board, depth + 1, True)
                board[i] = ' '
                best_score = min(score, best_score)
        return best_score

# Example: To find the optimal first move (Game 3)
board = [' '] * 9
best_move = -1
best_score = -math.inf
for i in range(9):
    if board[i] == ' ':
        board[i] = 'X'
        score = minimax(board, 0, False)
        board[i] = ' '
        if score > best_score:
            best_score = score
            best_move = i

print(f"The optimal value backup for the root is: {best_score} (0 means optimal play leads to a Tie)")
print(f"An optimal first move position index (0-8) is: {best_move}")