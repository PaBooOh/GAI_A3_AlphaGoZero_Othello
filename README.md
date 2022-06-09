# Implmentation of the Othello based on AlphaGoZero
## Prerequisites
Our program is entirely based on Python 3.x <br>
The libraries we use to build neural networks and GUI are <u>Tkinter<u> and</u>Pytorch</u>, respectively
## Training
To train game data collected, you need to
```
python training.py
```
Here training and conducting self-play, are alternated.

## Play
Once the training process stop and models are saved to local, you can open GUI to play against yourself or against AI you trained. Since we have already provided two models using CNN and ResNet, you have no need to train from scratch if you just want to play against AI, using the following command
```
python startup.py
```
## GUI
The graphical user interface is designed based on <u>Tkinter</u> in
```
gui.py
```

## Game rule
We define the game rule of Othello in
```
game.py
```

## MCTS
MCTS is a crucial part of our agent, which combine neural network. It is implemented in 
```
mcts.py
```

## (Hyper)parameters
We create a separate script for the adjustment of (hyper)parameters. You could tune them in
```
config.py
```
For example, you could change the learning rate, the number of times MCTS to be performed, the board size, etc, or determine whether to use Dirichlet noise...

## Network and model
The definition of two neural networks <u>CNN</u> and <u>ResNet</u> are written in
```
/network/convnet.py
/network/resnet.py
```
Furthermore, the models we/you trained are saved in folder
```
/model/cnn/optimal.pt
/model/restnet/optimal.pt
```


