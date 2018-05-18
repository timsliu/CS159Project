This document contains instructions on running additional custom environments.

1) Copy the file HalfInvertedPendulum.py to the directory:

/anaconda/lib/python2.6/site_packages/gym/envs/mujoco

2) Copy the file half_inverted_pendulum.xml to the directory:

/anaconda/lib/python2.6/site_packages/gym/envs/mujoco/assets

3) Replace the __init__.py file in 

/anaconda/lib/python2.6/site_packages/gym/envs/

with the version in this directory. This will register the new environment.

4) Add the line

from gym.envs.mujoco.HalfInvertedPendulum import HalfInvertedPendulumEnv

to the end of the file:

/anaconda/lib/python2.6/site_packages/gym/envs/mujoco/__init__.py

which will ensure the new environment is imported.

5) Copy the file scoreboard.py