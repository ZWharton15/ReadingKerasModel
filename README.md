# ReadingKerasModel
A short script to read a keras model (.h5 or /.hdf5 file) and classify images through a webcam in real-time

## Usage
This script is intended for use in a command line. To use this script you can do:
> *python use_model.py -model my_model.h5 -temp_img_dir C:/images/*

- -model is the path to your own model file
- -temp_img_dir is the path to a folder where the program can temporarily save the current webcam frame

*Note: flags are not required if the model and image directory are entered in that order*

Once the model is loaded, the webcam display will be shown with the class value **(0-num of classes in your model)** or **N/A** in the top-left if the threshold confidence (0.7 by default) is not met. Bellow that is the average FPS of the script, showing how many predictions can be made each second.
