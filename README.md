# DetectAndTrack
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

## About
This repo contains detection and tracking code which purposed for tracking the urban traffic over uav. This proposed method consist of consecutive combination of DNN (Deep Neural Network) and fast image processing algorithm to be used on the image taken by the UAV camera. Firstly, target is detected by the DNN which trained on a dataset focused on urban traffic. Thereafter, detected target is started to be tracked with MOSSE(Minimum Output Sum of Squared Error) algorithm. Finally, maneuvers are calculated using the PID control algorithm over the relationship of tracked target position to the center of the window frame, in order to enable the UAV to follow the target in real-time. <br \>

Linux files are written so as to work on Jetson Nano enviroment, the algorithm tested on Jetson as Hardware In the Loop (HIL) with simulation enviroment Gazebo provided by Pixhawk 

## Dependencies
*Opencv 4.4.0 required (in absence of this Mosse will not work, other versions has not tested yet) <br \>
*Mavsdk required _for hardware test_ <br \>
*Cuda >17(optional) 
## Version Notes
Only linux amd64/arm base updates comming between the versions. Windows version still remain v1.0.0 <br \>

~~modelv4 is optional file. It doesnt work current settings in file~~ <br \>
modelv5 will not come :( reverted to v4 


cuda pckg error is fixed as regards to https://askubuntu.com/questions/1276896/error-processing-archive-installing-cuda-on-ubuntu-20-04
