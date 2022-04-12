# Playing Hanabi using RIS-MCTS (Re-determinizing Information Set Monte Carlo Tree Search)

## Hanabi game

Rules for the game of hanabi can be found [here](https://www.spillehulen.dk/media/102616/hanabi-card-game-rules.pdf).

Develop an AI capable of playing Hanabi is not trivial: the game is cooperative, not zero-sum and with imperfect information.

## Referenced paper

We did lots of research in order to implement the AI, but most of the algorithm is an implementation of the paper [Re-determinizing Information Set Monte Carlo Tree Search in Hanabi](https://arxiv.org/pdf/1902.06075.pdf)

## Results

Playing against himself the AI is capable of obtaining an average score of 18/25 (that is quite remarkable compared to others AI which don't use deep reinforcement learning). Even against other AI we observed that is capable of adapting quite well (being a cooperative game, the score strongly depend on the capability of all the players).
