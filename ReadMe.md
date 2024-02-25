# Task 4 README

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