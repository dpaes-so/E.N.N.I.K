def relu(input):
    if(input < 0):
        return (0)
    else:
        return(input)

def relu_derivative(x):
    if(x < 0):
        return (0)
    else:
        return(1)
    
def calculate_error(target, prediction):
    return target - prediction

def calculate_loss(target, prediction):
    error = calculate_error(target,prediction)
    loss = error **2
    return loss

def weighted_sum(inputs, weights, bias):
    res = bias
    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have the same length")
    for i in range(len(inputs)):
        res += inputs[i] * weights[i]
    return res

def neuron_from_lists(inputs, weights, bias):
    return relu(weighted_sum(inputs,weights,bias))

def neuron_gradients(inputs, weights, bias, target):
    z = weighted_sum(inputs,weights,bias)
    output = relu(z)
    switch = relu_derivative(z)
    return (2 * (output - target) * switch)
    
def layer_from_lists(inputs, weights, biases):
    outputs = []
    if len(biases) != len(weights):
        raise ValueError("Different number of weights and biases")
    for i in range(len(biases)):
       outputs.append(neuron_from_lists(inputs,weights[i],biases[i]))
    return outputs

print(neuron_gradients([4, 5], [1, 2], 0, 10))
