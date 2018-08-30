# ReadingKerasModel
A short script to read a keras model (.h5 or /.hdf5 file) and classify images through a webcam in real-time

##Usage
This script is intended for use in a command line. To use this script you can do *python use_model.py -model my_model.h5 -temp_img_dir C:/images/*
-model is the path to your own model file
-temp_img_dir is the path to a folder where the program can temporarily save the current webcam frame
