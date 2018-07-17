# Augmented Random Search

The project includes a simple multiprocess version of the Augmented Random Search(ARS) algorithm. Please find a detailed description of the algorithm in Augmented_Random_Search_Notes.ipynb.

### Results: Half Cheetah Example

[![Watch the video](https://github.com/cpow-89/Augmented-Random-Search/blob/master/monitor/HalfCheetahBulletEnv-v0/Screenshot%20from%202018-07-17%2022-21-23.png)](https://raw.githubusercontent.com/cpow-89/Augmented-Random-Search/master/monitor/HalfCheetahBulletEnv-v0/openaigym.video.0.12196.video000000.mp4)

### Requirements(tested on Linux using Anaconda):

pip install gym==0.10.5<br>
pip install pybullet==2.0.8<br>
conda install -c conda-forge ffmpeg

### How to run the project:
- set up a json config file
	- see as an example ./configs/HalfCheetahBulletEnv_config.json
	- the config file includes hyperparameter settings and general input/output settings
- open the console and run:
	python main.py -c HalfCheetahBulletEnv_config.json -t false<br>
	optional arguments:<br>
  
  -h, --help<br>
    - show help message<br>
    
  -c , --config<br>
    - Config file name - file must be available as .json in ./configs<br>
    
  -t , --train    <br>
    - If true, the program will train, save and evaluate the model. <br>
    - If false, the program will search for a checkpoint in ./checkpoints to load and evaluate the model.<br>

### The project is inspired by:

Simple random search provides a competitive approach to reinforcement learning by Horia Mania, Aurelia Guy, Benjamin Recht
Link: https://arxiv.org/abs/1803.07055

Artificial Intelligence 2018: Build the Most Powerful AI by 
Hadelin de Ponteves and Kirill Eremenko 
