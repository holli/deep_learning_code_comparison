# Deep learning frameworks from coders perspective

Git includes sample python notebooks for some common usecases for various deep learning frameworks. Notebooks were done from the viewpoint of incrementally solving a specific problems. E.g. after defining a model there is a test that the code works by running on predicition instead of just trusting that the code is right.

## Simple mnist cnn network example

- Load mnist data
  - Show some samples
- Define simple cnn model
  - Test the code
  - Train the model and print progress information
  - Test the accuracy of the model
  - Use gpu for calculations
- Persist the work for future use
  - Save the model
  - Load the model
  - Predict one image with the loaded model

**Different frameworks**

- Pytorch
  - https://github.com/holli/deep_learning_code_comparison/blob/master/mnist/Pytorch.ipynb
- TensoFlow
  - https://github.com/holli/deep_learning_code_comparison/blob/master/mnist/TensorFlow-basic.ipynb
- Keras and TensorFlow
  - https://github.com/holli/deep_learning_code_comparison/blob/master/mnist/Keras_TensorFlow.ipynb

## What to pay attention to while looking the notebooks

Keras feels the most beginner coder friendly. It's nice that when you are defining the model you don't have to calculate the layer's input sizes manually. Fitting prints nice progress to the notebook. Saving and loading a model is straight forward.

Pytorch feels very pythonic. Everything requires a bit more code but it's simple to write and easy to debug. Most flexible option as the model was dynamic

TensorFlow was very hard to code and debug. Even though TF seems to have most candy on the top, for example serving models in production environments or tensorboard visualization, the basic development was unintuitive. Separation of the graph definition and the graph usage complicated the development especially in notebook environment.



