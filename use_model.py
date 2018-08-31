import cv2
import numpy as np
from keras.models import load_model
from keras import backend
import time
import sys


"""
Resizes the webcam frame into an image array and preprocess using the imported models method.
Returns an array of the prediction likelihood for each class the model knows.

@param model: the loaded model used for predictions
@param img: the current frame of the webcam
@param img_target_size: the default size of the model (299 for InceptionV3 or 224 for VGG16)
"""
def predict(model, img, img_target_size):
    img2 = cv2.resize(img, (img_target_size,img_target_size))
    x = np.expand_dims(img2, axis=0)
    x = preprocess_input(x.astype(float))
    prediction = model.predict(x)
    return prediction


"""
Read the arguments from command line input either as the model directory and image directory in order, or any order using the -model and -temp_img_dir flags.
Returns the model name and image directory to store the webcam frame.
"""
def read_arguments():
    model_name = None
    model_type = None
    img_size = None
    prediction_threshold = None
    
    for i in range(1, len(sys.argv)):
        #If a flag is given, use the next argument index to get the model directory
        if "-model_dir" in sys.argv[i]:
            model_name = sys.argv[i+1]
            ++i
            continue
        elif model_name == None:
            model_name = sys.argv[i]
            continue
        
        if "-model_type" in sys.argv[i]:
            model_type = sys.argv[i+1]
            ++i
            continue
        elif model_type == None:
            model_type = sys.argv[i]
            continue
        
        if "-img_size" in sys.argv[i]:
            img_size = sys.argv[i+1]
            ++i
            continue
        elif img_size == None:
            img_size = sys.argv[i]
            continue
        
        if "-threshold" in sys.argv[i]:
            prediction_threshold = sys.argv[i+1]
            ++i
            continue
        elif img_size == None:
            prediction_threshold = sys.argv[i]
            continue
        
        pass
    return model_name, model_type, img_size, prediction_threshold

model_name, model_type, img_size, threshold = read_arguments()

#Validate the users arguments
if threshold == None:
    threshold = 0.7 #default value
elif not isinstance(threshold, float):
    print("Threshold is not a number")
    sys.exit()
elif threshold < 0.0 or threshold > 1.0:
    print("Threshold is not between 0.0 and 1.0")
    sys.exit(0)

if model_type == "inceptionv3":
    from keras.applications.inception_v3 import preprocess_input
    if img_size == None:
        img_size = 299
    pass
elif model_type == "inception_resnetv2":
    from keras.applications.inception_resnet_v2 import preprocess_input    
    if img_size == None:
        img_size = 299
    pass
elif model_type == "vgg16":
    from keras.applications.vgg16 import preprocess_input
    if img_size == None:
        img_size = 224
    pass
else:
    print("Model type not found (must be inceptionv3, inception_resnetv2, or vgg16)")
    sys.exit(0)
    pass

try:
    model = load_model(model_name)
except:
    print("Model not found")
    sys.exit(0)
    

font = cv2.FONT_HERSHEY_SIMPLEX
frame_counter = 0
avg_frames = 0
cap = cv2.VideoCapture(0)

#Try catch to prevent keyboard interrupt log
try:
    while True:
        #Time each iteration to show the average FPS
        start = time.time()
        success, frame = cap.read()
        
        if success:
            label = predict(model, frame, 299)
            #Get the predicted class index
            pred = label.tolist()
            pred = pred[0]
            pred_index = pred.index(max(pred))
            #print("Prediction: ", pred)
            #Show only the index of the prediction on screen
            #TODO: read tuple to convert index to plaintext name
            if float(pred[pred_index]) >= threshold:
                cv2.putText(frame, str(pred_index),(30,50), font, 2,(255,0,255),2,cv2.LINE_AA)
            else:
                #If the threshold is not met, write N/A instead
                cv2.putText(frame, "N/A",(30,50), font, 2,(255,0,255),2,cv2.LINE_AA)
               
            end = time.time()
            #get the amount of time passed to do 1 iteration
            total_passed = end - start
            try:
                #estimate how many predictions can be done per second
                total_passed = round(1 / total_passed)
            except ZeroDivisionError:
                total_passed = 0
            
            #Write the FPS bellow the prediction
            cv2.putText(frame, f"FPS: {total_passed}", (30,100), font, 1,(255,0,255),2,cv2.LINE_AA)    
            
            #Show the completed frame
            cv2.imshow("output", frame)
        
        else:
            print("Error reading webcam")
        #Close the program when the escape key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
#Notify the user that the keyboard interrupt was used
except KeyboardInterrupt:
    print("program stopped by keyboard input")
#Before ending the program, clean the environment to release the memory and GPU resources
finally:
    print("Cleaning environment")
    del model
    backend.clear_session()
    cap.release()
    cv2.destroyAllWindows()