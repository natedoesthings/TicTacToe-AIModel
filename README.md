# TicTacToe-AIModel! 🤖
This is the model configured and used for the project, [Tik-Tak-AI?](https://github.com/natedoesthings/TicTacToeApp)

# How it Works.

- Minimax Algorithm:
  - Recursive Algorithm configured to find the best possible next move by analyzing the maximizing player (O) and the minimizing player (X).
  - After anaylsis by going through all possible combinations, returns the best move.
  - [Short YT Video!](https://www.youtube.com/watch?v=l-hh51ncgDI&t=189s&ab_channel=SebastianLague)
 
- PyTorch
  - Instead of playing the game over and over, I used pytorch to create multiple datapoints of possible game scenarios.
  - For each scenario, i would use the minimax algorithm to find the best move for player O, the computer.
  - Was able to retrieve a dataset of length 19,000+ and transformed that data into a csv file
 
- Core ML and Create ML
  - Using Apple's built in software Create ML, I was able to take that csv file to use and train my model
  - For my use case, I went with a Tabular Classification since my data was based on finding the best target value based on the state of the board
  - Ex. X,,,X,O,,,,,6
  - This would mean that the computer would go to position 6 as that would block player X of winning
  - I trained my data using the Random Forest algorithm with 100 iterations
  - Using Core ML, I was able to easily import my model into the swift application and process input to get a predicted output!
 



