import itertools
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# def check_winner(board):
#     win_patterns = [
#         [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
#         [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
#         [0, 4, 8], [2, 4, 6]              # Diagonals
#     ]
#     for pattern in win_patterns:
#         if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != 0:
#             return board[pattern[0]]
#     if 0 not in board:
#         return 0  # Draw
#     return None

# def minimax(board, depth, is_maximizing):
#     winner = check_winner(board)
#     if winner is not None:
#         return 10 - depth if winner == 1 else depth - 10 if winner == -1 else 0

#     best_score = float('-inf') if is_maximizing else float('inf')
#     for i in range(len(board)):
#         if board[i] == 0:
#             board[i] = 1 if is_maximizing else -1
#             score = minimax(board, depth + 1, not is_maximizing)
#             board[i] = 0
#             if is_maximizing:
#                 best_score = max(score, best_score)
#             else:
#                 best_score = min(score, best_score)
#     return best_score

# def best_move(board):
#     best_score = float('-inf')
#     move = -1
#     for i in range(len(board)):
#         if board[i] == 0:
#             board[i] = 1  # AI is X
#             score = minimax(board, 0, False)
#             board[i] = 0
#             if score > best_score:
#                 best_score = score
#                 move = i
#     return move

"""
Updated minimax Algorithm
"""
def check_winner(board):
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for pattern in win_patterns:
        if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != 0:
            return board[pattern[0]]
    if 0 not in board:
        return 0  # Draw
    return None

def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner is not None:
        return depth - 10 if winner == 1 else 10 - depth if winner == -1 else 0

    best_score = float('-inf') if is_maximizing else float('inf')
    for i in range(len(board)):
        if board[i] == 0:
            temp_board = copy.deepcopy(board)
            temp_board[i] = -1 if is_maximizing else 1
            score = minimax(temp_board, depth + 1, not is_maximizing)
            if is_maximizing:
                best_score = max(score, best_score)
            else:
                best_score = min(score, best_score)
    return best_score

def best_move(board):
    best_score = float('-inf')
    move = -1
    for i in range(len(board)):
        if board[i] == 0:
            temp_board = copy.deepcopy(board)
            temp_board[i] = -1  # Simulate a move for the maximizing player
            score = minimax(temp_board, 0, False)  # Call minimax for the minimizing player
            if score > best_score:
                best_score = score
                move = i
    return move


def board_to_tensor(board):
    mapping = {'x': 1, 'o': -1, None: 0}
    flattened_board = [mapping[cell] for cell in board]
    return torch.tensor(flattened_board, dtype=torch.float32)


# Generate all possible valid Tic-Tac-Toe boards
possible_items = ["x", "o", None]
all_boards = list(itertools.product(possible_items, repeat=9))
valid_boards = [board for board in all_boards if None in board]

# Convert valid boards to tensors and generate labels using minimax
boards = []
labels = []
for board in valid_boards:
    tensor_board = board_to_tensor(board)
    optimal_move = best_move(tensor_board.numpy().tolist())
    if optimal_move != -1:  # Only include boards where a valid move is found
        boards.append(tensor_board)
        labels.append(optimal_move)

# Convert labels to a tensor
labels = torch.tensor(labels)

# Create the custom dataset
class TicTacToeDataset(Dataset):
    def __init__(self, boards, moves):
        self.boards = boards
        self.moves = moves

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        move = self.moves[idx]
        return board, move

# Create the dataset
dataset = TicTacToeDataset(boards, labels)

# Print the number of samples in the dataset
print(f"Number of samples in dataset: {len(dataset)}")

# # Print the first few samples
# num_samples_to_display = 5
# for i in range(num_samples_to_display):
#     board, label = dataset[i]
#     board_str = ','.join(['x' if cell == 1 else 'o' if cell == -1 else '.' for cell in board.numpy()])
#     print(f"Board state:\n{board_str}")
#     print(f"Label (Optimal move for O): {label.item()}")
#     print("--------------------")

# Print to csv
f = open("trainingdata.txt", "a")
for i in range(len(dataset)):
    board, label = dataset[i]
    board_str = ','.join(['x' if cell == 1 else 'o' if cell == -1 else '.' for cell in board.numpy()]) + f",{label.item()}"
    print(f"Board state:\n{board_str}")
    f.write(board_str + "\n")

f.close()
