def calculate_error(target, prediction):
    return target - prediction

def calculate_loss(target, prediction):
    error = calculate_error(target,prediction)
    loss = error **2
    return loss
  
def simple_neuron(input, weight, bias):
    return input * weight + bias


def calculate_derivative(weight, input, target):
    prediction = input * weight
    error = prediction - target
    derivative = 2 * error * input
    return derivative

def update_weight(weight, input, target, learning_rate):
    derivative = calculate_derivative(weight, input, target)
    weight = weight - learning_rate * derivative
    return weight

weight = 2
input = 2
target = 10
learning_rate = 0.01

for step in range(10):
    weight = update_weight(weight, input, target, learning_rate)
    prediction = input * weight
    # print(step, weight, prediction)


def calculate_derivatives(weight1, weight2, input1, input2, target):
    prediction = input1 * weight1 + input2 * weight2
    error = prediction - target

    derivative1 = 2 * error * input1
    derivative2 = 2 * error * input2

    return derivative1, derivative2

def update_weights(weight1, weight2, d1, d2, learning_rate):
    weight1 = weight1 - learning_rate * d1
    weight2 = weight2 - learning_rate * d2
    return weight1, weight2

d1, d2 = calculate_derivatives(2, 3, 4, 5, 20)

weight1, weight2 = update_weights(2, 3, d1, d2, 0.1)

weight1 = 2
weight2 = 3

input1 = 4
input2 = 5
target = 20
learning_rate = 0.01

for step in range(20):
    # Calculate both derivatives
    d1, d2 = calculate_derivatives(
        weight1, weight2, input1, input2, target
    )

    # Update both weights
    weight1, weight2 = update_weights(
        weight1, weight2, d1, d2, learning_rate
    )

    # Calculate the new prediction
    prediction = input1 * weight1 + input2 * weight2

    # Print the results
    print(step, prediction, weight1, weight2)