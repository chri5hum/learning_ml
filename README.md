# ml bandwagoning

A super basic NN for educational purposes

calculates likelihood of a person being female based on height, weight

```python
import numpy as np

```

```python
# activation function - sigmoid: f(x) = 1 / (1 + e^(-x))
# derivative of activation: f'(x) = f(x) * (1 - f(x))
# loss function - MSE (mean squared error): Avg sum of (y_true - y_pred)^2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)
    

def mse_loss(y_true, y_pred):
    #y_true and y_pred are arrays with equal size
    return ((y_true - y_pred) **2).mean()

```

```python
# 2 inputs
# hidden layer of 2 neurons
# output layer of 1 neuron
# total 6 weights, 3 biases
class BasicNeuralNetwork:
```

```python
  def __init__(self):
      self.w1 = np.random.normal()
      self.w2 = np.random.normal()
      self.w3 = np.random.normal()
      self.w4 = np.random.normal()
      self.w5 = np.random.normal()
      self.w6 = np.random.normal()
        
      self.b1 = np.random.normal()
      self.b2 = np.random.normal()
      self.b3 = np.random.normal()
```

```python
  # feedforward loop to calculate output layer
  def feedforward(self, x):
      h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
      h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
      o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
      return o1
```

```python
  #training loop
  def train(self, data, all_y_trues):
      # data is a (n x 2) np array, [weight-135, height-66], or shift by mean
      # all_y_trues is a np array, 1 = female, 0 = male
      
      learn_rate = 0.1
      epochs = 1000 # times to loop through the entire data set

      for epoch in range(epochs):
          for x, y_true in zip(data, all_y_trues):
              # do a feedforward
              sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
              h1 = sigmoid(sum_h1)

              sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
              h2 = sigmoid(sum_h2)

              sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
              o1 = sigmoid(sum_o1)
              y_pred = o1
              
```

```python
            # backpropogation step (calculate partial derivatives)
            # backpropogation allows us to calculate dL/dw, or the change in loss with respect to w
            # so we can increase or decrease the weight/bias of a neuron to decrease loss
            
              d_L_d_ypred = -2 * (y_true - y_pred)

              #neuron o1
              d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
              d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)

              d_ypred_d_b3 = deriv_sigmoid(sum_o1)

              d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
              d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                
              #neuron h1
              d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
              d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
              d_h1_d_b1 = deriv_sigmoid(sum_h1)

              #neuron h2
              d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
              d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
              d_h2_d_b2 = deriv_sigmoid(sum_h2)
```
```python
            # update weight and biases
            # increase or decrease neuron weights by dL/dw, and neuron biases by dL/db

              #neuron h1
              self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
              self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
              self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

              #neuron h2
              self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
              self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
              self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

              #neuron o1
              self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
              self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
              self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
```
```python
            # calculate total loss at end of each epoch
            # pretty much only for the visual component
            
              if epoch % 10 == 0:
                  y_preds = np.apply_along_axis(self.feedforward, 1, data)
                  loss = mse_loss(all_y_trues, y_preds)
                  print("Epoch %d loss: %.3f" % (epoch, loss))
```

