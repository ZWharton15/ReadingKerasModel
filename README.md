# ReadingKerasModel
A short script to read a keras model (.h5 or /.hdf5 file) and classify images through a webcam in real-time

## Usage
This script is intended for use in a command line. To use this script you can do:
> *python use_model.py -model_dir my_model.h5 -model_type inceptionv3*

- -model_dir is the path to your own model file
- -model_type is the model architecure used (inceptionv3, inception_resnetv2, vgg16)
- -img_size is the size in pixel of the preprocessed image (default is 224 for vgg16 and 299 for inception)
- -threshold is the minimum confidence needed for the model to make a prediction (default is 0.7)
 *Note: specifying img_size and threshold are optional, default parameters will be used instead*

Once the model is loaded, the webcam display will be shown with the class value **(0-num of classes in your model)** or **N/A** in the top-left if the threshold confidence is not met. Below that is the average FPS of the script, showing how many predictions can be made each second.

## Performance
22fps on a retrained InceptionV3 model with a Geforce GTX 980
