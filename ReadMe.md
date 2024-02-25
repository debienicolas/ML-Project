
# Multi-agent learning in canonical games and dots-and-boxes

Google deepmind's openspiel package was used for all 3 tasks.

## Task 2

Benckmark games: Biased Rock-Paper-Scissors Game, Dispersion Game, Battle of the Sexes and the Prisoners Dilemma. 
Learning and dynamics analysis using basic reinforcement learning algorithms on benchmark matrix games.

## Task 3

Solving Dots-and-Boxes using the minimax algorithm.\
Optimisations made:
- Transposition tables
- Exploitation of symmetries

## Task 4 

To build an agent capable of playing the game of dots-and-boxes on larger board size, a graph neural network was used to guide the Monte carlo tree search.
The approach is heavily based on the AlphaZero framework. 

This is an overview of all the files used for Task 4

Overview of files:
- main-DAB.py : This file is where all the parameters for agent learning are set and launches the training procedure. This is the only file that need to be run for the training to proceed.
- Coach.py : This file hosts the training environment, orchestrates the self-play and evaluation against previous opponents.
- Arena.py : This class can be used to pit agents against each other, used for evaluation.
- GNNet.py : This file contains the model description and operations of the graphical neural net.
- GNNEvaluator.py : Extends the openspiel evaluator class used for MCTS.
- Graph.py : This file is responsible for converting the openspiel game states to graph representation.
- MCTS.py : Extends openspiel MCTS so it is suitable for self-play.

dotsandboxes_agent.py is where our agent capable of playing the tournament is located.

The Task 4 folder contains extra files used for evaluation and running experiments.
