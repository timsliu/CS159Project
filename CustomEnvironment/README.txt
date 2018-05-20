This document contains instructions on running additional custom environments.

1) Make a copy the file InvertedPendulum.py in the directory:
_________________

/anaconda/lib/python2.6/site_packages/gym/envs/mujoco

and change the name to HalfInvertedPendulum.py. Change the xml file used 
in the file to "half_inverted_pendulum.xml"

2) Copy the file half_inverted_pendulum.xml to the directory:
_________________

/anaconda/lib/python2.6/site_packages/gym/envs/mujoco/assets

3) Replace the __init__.py file in 
_________________

/anaconda/lib/python2.6/site_packages/gym/envs/

with the version in this directory. This will register the new environment.

4) Add the line
_________________

from gym.envs.mujoco.HalfInvertedPendulum import HalfInvertedPendulumEnv

to the end of the file:

/anaconda/lib/python2.6/site_packages/gym/envs/mujoco/__init__.py

which will ensure the new environment is imported.

-----------------
The link below is a guide to creating new environments, but not all points
apply:

https://github.com/openai/gym/wiki/Environments