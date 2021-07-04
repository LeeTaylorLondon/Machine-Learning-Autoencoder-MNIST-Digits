# ML Autoencoder MNIST Digits
## Description 
This project features an autoencoder model trained to encode, compress, and decode hand-written digits.
There are two files, model_functions.py which contains the functions and structure of the model.  
Secondly, there is interactive.py which provides a small GUI screen to draw or load a random digit between 0 and 9. 
The user's drawing can be inputted into the trained model by pressing a key in which the neural network outputs
its auto-encoded version.

## Installation
* Pip install h5py (built with 3.1.0)
* Pip install tensorflow (built with 2.5.0)
* Pip install numpy (built with 1.19.5)
* Pip install pygame (built with 2.0.1 - for interactive.py)

## Usage
Before running any python files please make sure you have completed ALL above installation steps.

Read model_functions.py function build_model, to see the structure of the autoencoder.  
If you run model_functions.py, it will build and train a new model which will overwrite the existing saved model
and weights.
The newly trained model will be evaluated, and it's accuracy outputted.

![Image of interactive.py](/images/Capture.PNG)

Upon running interactive.py, a window will pop up. Pressing D passes the input to the Auto Encoder (AE), which
populates the right group of pixels with output from the AE. Pressing T loads a random image into the input section.
Pressing C clears both the input and output sections. 

## Autoencoder Details
(InputLayer) Input Shape: (28, 28, 1)  
(Flatten) 784 Units  
(Dense) 256 Units  
(LeakyReLU) Activation Layer  
(Dense) 128 Units  
(LeakyReLU) Activation Layer  
(Dense) 256 Units  
(LeakyReLU) Activation Layer  
(Dense) 784 Units  
(LeakyReLU) Activation Layer  
(Output Layer) Output Shape: (28, 28, 1) 
  
Testing accuracy: ~81%

## Credits
* Author: Lee Taylor

## Note
This is my first autoencoder machine learning project, inspired by the book,
Generative Deep Learning by David Foster. Project also inspired by YouTube tutorial by sentdex.
