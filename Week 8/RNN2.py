'''
In many real-world applications, we have data that comes in the form of sequences.
For example, in natural language processing, we often work with sequences of words or characters.

In time series analysis, we work with sequences of measurements taken at regular intervals over time.

In speech recognition, we work with sequences of acoustic features representing speech signals.

By learning the relationship between two sequences, we can use that knowledge to make predictions or
generate new sequences. For example, in natural language processing, we can use a trained model to
generate new sentences that have similar structure and content to the training data.

In time series analysis, we can use a trained model to make predictions about future measurements based on past data.

'''


'''
Train an RNN to predict a BINARY OUTPUT sequence based on a BINARY INPUT sequence.
The input sequence is defined as a list of integers, and the target sequence is defined
as a list of integers with the same length as the input sequence.
We convert the input and target sequences to numpy arrays and
define the RNN model using the Keras Sequential API.
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# Define the input sequence as a list of integers
input_sequence = [1, 0, 0, 1, 1, 0, 1, 0, 1]

# Define the target sequence as a list of integers
target_sequence = [1, 1, 0, 1, 0, 1, 0, 0, 1]

# Convert the input and target sequences to numpy arrays
x = np.array([input_sequence])
y = np.array([target_sequence])

'''
The RNN model consists of a SimpleRNN layer with 4 units, followed by 
a Dense layer with 1 unit and a sigmoid activation function. 
We compile the model with binary cross entropy loss and Adam optimizer, 
and train it on the input and target sequences for 100 epochs.

Cross entropy loss is a metric used to measure how well a classification model in machine learning performs. 
The loss (or error) is measured as a number between 0 and 1, with 0 being a perfect model. 
The goal is generally to get your model as close to 0 as possible.
'''
# Define the RNN model
model = Sequential()
model.add(SimpleRNN(units=4, input_shape=(1, 1)))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam')

'''
Note that we reshape the input and output data to have 
the shape (num_samples, num_timesteps, input_dim) and 
(num_samples, output_dim), respectively, using the reshape method of numpy arrays. 
We also pass the reshaped data to the fit method of the model.
'''


'''
When we train the model on the input and target sequences, we're essentially teaching the model 
to recognize patterns in the input sequence that correspond to specific patterns in the target sequence. 
The model learns to do this by adjusting the weights of its internal connections during the training process.
'''
# Train the model on the input and target sequences for 100 epochs
model.fit(x.reshape(len(input_sequence), 1, 1), y.reshape(len(target_sequence), 1), epochs=10000)

'''
Finally, we test the model on a new input sequence and print the predicted output
using the predict method. The output is a probability between 0 and 1, 
representing the likelihood that the target sequence matches 
the input sequence at each position.
'''

# Test the model on a new input sequence
new_sequence = [0, 1, 0, 1, 1, 0, 0, 1, 0]
prediction = model.predict(np.array([new_sequence]).reshape(len(new_sequence), 1, 1))
print(prediction)

'''
The output of the RNN model is a probability between 0 and 1,
representing the likelihood that the target sequence matches 
the input sequence at each position.

The value of the output represents the probability that the corresponding position 
in the target sequence matches the input sequence. For example, the first element of the output
represents the probability that the first position of the target sequence matches 
the first position of the input sequence. 
The second element of the output represents the probability that the second position 
of the target sequence matches the second position of the input sequence, and so on.

In the specific example I provided of an RNN predicting a binary output sequence based on 
a binary input sequence, the input sequence and output sequence represent two related sequences of binary values. 
The model is trained to recognize patterns in the input sequence that correspond to 
specific patterns in the output sequence. Once the model has been trained, 
we can use it to predict the output sequence that 
corresponds to a new input sequence, based on what it has learned during the training process
'''