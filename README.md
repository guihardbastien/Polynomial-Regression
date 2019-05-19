#  :fire: Polynomial regression using stochastic gradient descent (SGD)

## Intro
SGD optimization is used in supervised learning for large scale training on massive dataset that can be many directionnal.
SGD is also being used a lot to determine a confidence based on an activation function 

##What's this for ? 

Given a 2 dimensionnal dataset we're trying to fit a polynomial function to this data.

Here's the function: `y = ax^3 + bx^2 + cx + d`  

## How does it work ?

Given an x input, we want to predict the correct y output.

So we create a function with random a,b, c and d

Then, given the actual y output we can calculate the **error** between this function's result and the actual y output, using the mean squared error method :

`LOSS = mean((guess - actual_y)^2)` 

now that we have the loss function, we can minimize it relatively to a,b,c,and d.

ex: How to determine `delta a` 

```
Loss = (error)^2

pdLoss/pda =    pdLoss/pdError  *   pdError/ pda
pdLoss/pda =    (2 * Error)     *   3x*a^2
```
_pd* = partial derivative_

Finally we multiply the result by a learning rate to set the pace of the optimization.