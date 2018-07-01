# Deep learning frameworks from coders perspective

This repository includes sample python codes and notebooks for some common usecases for various deep learning frameworks. Notebooks were done from the viewpoint of incrementally solving a specific problems. E.g. after defining a model there is a test that the code works by running on predicition instead of just trusting that the code is right.


## Mnist cnn image classification example

- Load mnist data
  - Show some samples
- Define simple cnn model
  - Test the code ... Train the model and print progress information ... Test the accuracy of the model
- Persist the work for future use
  - Save the model ... Load the model ... Predict one image with the loaded model

**Different frameworks**

- Pytorch
  - https://github.com/holli/deep_learning_code_comparison/blob/master/mnist/Pytorch.ipynb
- TensoFlow
  - https://github.com/holli/deep_learning_code_comparison/blob/master/mnist/TensorFlow-basic.ipynb
- Keras and TensorFlow
  - https://github.com/holli/deep_learning_code_comparison/blob/master/mnist/Keras_TensorFlow.ipynb
- Specific notes
  - Keras shows all the basic info nicely by default.
  - Pytorch is simple to customize because everything is very dynamic. test_model_accuracy can include calculations inside the function. Model is used by calling it like any other python function.
  - Whereas in TensorFlow the accuracy related graph had to be defined before initialization. Interaction with the model goes through `sess.run(op, feed_dict)`


## Openai gym cartople reinforcement (qdn) example

- Small network that can learn to solve CartPole environment with reinforcement learning
  - https://gym.openai.com/envs/CartPole-v0/
- Code is supposed to be clear and generalizable. Not the most optimized reinforcement learning for this problem.

**Different frameworks**

- Pytorch
  - https://github.com/holli/deep_learning_code_comparison/blob/master/gym_cartpole/pytorch.py
- TensorFlow
  - https://github.com/holli/deep_learning_code_comparison/blob/master/gym_cartpole/tf.py
- Keras and TensorFlow
  - https://github.com/holli/deep_learning_code_comparison/blob/master/gym_cartpole/keras_tf.py
- Specific notes
  - These code differences might be easiest to see by using file diff (or diff view in an IDE)
  - Codes are quite similar to each other. Keras is once again simplest to read.
  - Here Pytorch was easiest to "optimize". Other frameworks evaluate the `train_x` twice, once for getting `train_y` values and once just before the training.
  - TensorFlow version could be optimized by taking `train_y_target` (target rewards) wholly inside the graph but it would be complicated to code.  


## Cats and dogs, finetuning a pretrained model
 
- Load data and set up basic data transformations
- Use pretrained convolutional layers from vgg16 and finetune simple top on that.
- Show some sample predictions


**Different frameworks**

- Data download and preparing:
  - https://github.com/holli/deep_learning_code_comparison/blob/master/cats_dogs_finetune/prepare_data.ipynb
- Pytorch
  - https://github.com/holli/deep_learning_code_comparison/blob/master/cats_dogs_finetune/pytorch.ipynb
- Fast.ai (Pytorch)
  - https://github.com/holli/deep_learning_code_comparison/blob/master/cats_dogs_finetune/fastai_pytorch.ipynb
- Keras and TensorFlow
  - https://github.com/holli/deep_learning_code_comparison/blob/master/cats_dogs_finetune/keras_tf.ipynb
- Notes
  - All TensorFlow examples I found used the whole network and trained it again. There were no simple examples on altering the network although it should be possible.
  - Keras image transformations were more comprehensive by default but Pytorch has a simple interface to do your own.
  - Both Keras and Pytorch had nice single line commands to load pretrained networks. Simple things in Keras were easier to implement although with Pytorch it would be easier to some experimental structures.


## What to pay attention to while looking through the code/notebooks

Keras feels the most beginner coder friendly. It's nice that when you are defining the model you don't have to calculate the layer's input sizes manually. Fitting prints nice progress to the notebook. Saving and loading a model is straight forward.

Pytorch feels very pythonic. Everything requires a bit more code but it's simple to write and easy to debug. Most flexible option as the network was dynamic.

TensorFlow was very hard to code and debug. Even though TF seems to have most candy on the top, for example serving models in production environments or tensorboard visualization, the basic development was unintuitive. Separation of the graph definition and the graph usage complicated the development especially in notebook environment. It was too easy to forget to reset or initialize the graph and get some weird errors. 



