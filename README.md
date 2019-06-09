# Human-computer-Interaction-Ludo-Game-Based-on-Image-Recognition
This project is for CS159 2019Spring Course "Python Programming" taught by Professor Bao Yang.
## Basic Information
Project title: Human-computer interaction Ludo game based on image recognition
Team name: Hello full mark
Team members: Cheng Boyuan, Ni Junyan, Bentian Linai, Guo Yi

## Problem Introduction
### The background of Ludo games
Ludo is a highly simplistic version of Pachisi, a game that originated in the 6th century in India. This game is played by younger children all over the country. In this board game 2 to 4, players race their tokens from start to finish according to the dice rolls. Various variations are seen in the way people play Ludo.

Ludo originated in India by the 6th century. The earliest evidence of this game in India is the depiction of boards on the caves of Ajanta. This game was played by the Mughal emperors of India; a notable example being that of Jalaluddin Muhammad Akbar.

Variations of the game made it to England during the late 19th century. One which appeared around 1896 under the name of Ludo was then successfully patented. Special areas of the Ludo board are typically coloured bright yellow, green, red, and blue. Each player is assigned a colour and has four tokens of matching colour (originally bone discs but today often made of cardboard or plastic). The board is normally square with a cross-shaped game track, with each arm of the cross consisting of three columns of squares—usually six squares per column. The middle columns usually have five squares coloured, and these represent a player's home column. A sixth coloured square not on the home column is a player's starting square. At the centre of the board is a large finishing square often composed of triangles in the four colours atop the players' home columns – thus forming "arrows" pointing to the finish.
### Project innovation
We made appropriate improvements to the game based on traditional Ludo games. First of all, in the game to add props features, such as bombs and stars, so that the game's variability is more abundant, the emergence of a large number of props in the late time of the game can also make the difficulty of the game further.At the same time, it also adds to the strategic nature of the game. Second, increase the composition of human-computer interaction, when the player touches the bomb, he needs to perform a "video form" of rock-paper-scissors with the computer. If he loses, he will return to the starting point and start again, adding fun to the game.
### The libraries we use
#### Pygame
Pygame is a Python wrapper module for the SDL multimedia library. It contains python functions and classes that will allow you to use SDL’s support for playing cdroms, audio and video output, and keyboard, mouse and joystick input. We use pygame for the Ludo game part.

#### OpenCv2
Open Source Computer Vision Library was released under a BSD license and hence it’s free for both academic and commercial use. OpenCV2 was designed for computational efficiency and with a strong focus in real-time applications. In this project, the rock-paper-scissors part is created mainly using Opencv.

#### numpy
Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. In the whole project, we use the numpy library for scientific computing.

## The rules of the game
### Equipment
A Ludo board is is square with a pattern on it in the shape of a cross. The middle squares form the home column for each colour and cannot be landed upon by other colours. The middle of the cross forms a large square which is the 'home' area and which is divided into 4 home triangles, one of each colour. At each corner, separate to the main circuit are coloured squares where the pieces are placed to begin.

Counters start their circuit one square in from the end of the arm and adjacent to the starting square. Avoid modern boards which incorrectly place the first square at the end of the arm. Each player chooses one of the 4 colours (green, yellow, red or blue) and places the 4 pieces of that colour in the corresponding starting square. And there will be bombs and stars in the game. If the die is 6, a bomb can be placed on the board. If the die is 1, a star can be placed on the board.
### Play
A player must throw a 6 or 1 to move a piece from the starting circle onto the first square on the track.

Each throw, the player decides which piece to move. If no piece can legally move according to the number thrown, play passes to the next player.

If the player reaches a position with a bomb, the game of “rock、paper、scissors” will be played. The loser needs to return the piece to its starting point.

If the player reaches a position with a star, the pieces will go directly to the position of six squares away from the end point.
### Winning
When all the pieces of the player reach the end point.

## The process of programming
### Basic attributes
Board size = 1000*700
Padding = 10
Outline Width = 5
title = 'Ludo Board'
box size= 50*50
### Recognition of gesture

In this part, we capture and convert image into binary image. Then we compare the extracted feature points with the gesture dictionary and then determine the gesture and shape. Find the largest contour according to the contrast with the background.

Finally, according to the coordinates of concave and convex points in the image, the angle between two fingers is calculated by using the cosine theorem. Fingers can be identified by calculating the angle less than 90 degrees.

### Create the main game

We first design all the tools needed for the Ludo game. This includes a button to press when rolling dice, a button to move after rolling dice and a box to remind players about whose turn it is.

Then we draw the game board to show the position of the tokens. The major element of a game board is considered to be the colored track, whose color shows the start place and ending place of a particular player.

The drawing process is not complicated, mainly to calculate the coordinates of each point, and to match the corresponding color for each part. In addition, some images need to be inserted into the text, for which we define the Rendertext function for text insertion. A big difficulty in this part of the production process is that the coordinates of each point are difficult to calculate accurately. Another difficulty lies in the drawing of irregular figures.

Of course, one of the drawbacks of the game is that there is no distinction between the colors of the individual grids (if a distinction is needed, a grid class should be defined separately to determine whether it is flying to its corresponding position), and the corners of the road sections are not connected. This is also the direction that needs improvement in the future.

Now the game interface is finished as shown below.

Here we get the game board and we need tokens to be put on it. We define a class named player to store and define their behaviors while playing. The player class contains many functions, such as recording the color of the piece, recording the track of the piece, judging whether the piece leaves the starting point, judging whether the piece reaches the end point, moving the piece forward, returning the piece to the starting point, and playing the piece. Leap to the finish, show the pieces, and more.See the specific introduction of these functions in the coding line.

Similarly, we define a class named Ludo Button in Python, adding certain attributes to it. Circle is drawn to represent the token for our Ludo game.

Two functions are defined respectively to show the process and result of dice-rolling. The biggest difficulty in this part is the display of the animation, for which we have taken a number of static pictures of the throwing dice and played them in chronological order. The specific function is implemented in the diceroll function. To save space, the dice-roll function in the picture above exhibits only parts of the code.

We define three events to let the game operate as we planned. Rolling dice is used to decide how many steps the token would move. Clicking the mouse is for players to select which token to move. Finally, move button is used to judge where the token is and to accordingly control the tokens’ moving forward.

The most difficult function in this part is the conditional trigger after move. First, you need to judge whether the current piece reaches the end point. Secondly, you need to judge whether the piece can leave the starting point; then judge the state of the piece on the ending track; finally, judge the function of the piece after the piece has finished walking, such as placing a bomb, a star, and checking whether it touches to bombs, stars, etc.

When players roll the dice and get a number of 1, a star is set on the chess ‘s current position on the gameboard. The next time a token passes this star, it will be directly transferred to a place near the end point. Players encountering the star is lucky enough to predict a fast-win

When players roll the dice and get a number of 6, a bomb is set on the chess’s current position on the gameboard. The next time a token passes this bomb, the player will has to play the \textbf{paper-scissor-rock} game with the computer. If the player loses the game, the token will be forced to fly to the start point. If the player wins, then nothing will happen and he/she will be safe.

## Problems and optimization
### Lack of enough accuracy
In the part of rock-paper-scissors game, we used openCv for gesture recognition. But in many cases, the success rate of recognition is related to the quality of the image and the cleanliness of the background. Recognition is only successful when the background is white. Sometimes it takes a refresh or trial and error to succeed.
### Problems with game props
Props in the game may appear in the same grid in many times, which may cause the game to crash. The overlap of the pieces in the same lattice will not be shown on the board. It's not very gamer friendly.
### Playability of the game
The game is highly repetitive in its steps, which can be annoying to the player. The ultimate success of the game also depends on the luck of the players. Just two items may not please the player.
## Reference
During the project completion process, we looked at GitHub's open source code for image identification and modified it to be added to the Ludo game. 

Address: https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python/blob/master/new.py
