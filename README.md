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
