# Project
Code for [Explain to Fix: A Framework to Interpret and Correct DNN Object Detector Predictions](http://arxiv.org/abs/1811.08011) paper presented at [NIPS 2018 Workshop on Systems for ML](http://learningsys.org/nips18/)

# Instructions
Merge code into original [Caffe/SSD branch](https://github.com/weiliu89/caffe/tree/ssd) and recompile. Source code modifications change output detection format to include detection indices. All other processing is done by Python scripts in examples/e2x folders. Details for each script is given by "--help" command. Current script support SSD300 model as in the paper. Other models and datasets can be easily added.

# Demo Video
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/OCXghealLAY/0.jpg)](https://www.youtube.com/watch?v=OCXghealLAY)
