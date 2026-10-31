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
    shared = 2 * (output - target) * switch
    weight_grads = []
    for i in range (len(inputs)):
        weight_grads.append (shared * inputs[i])
    bias_grad = shared
    return (weight_grads,bias_grad)

def update_neuron(weights, bias, weight_grads, bias_grad, learning_rate):
    new_weights = []
    for i in range (len(weights)):
        new_weights.append(weights[i] - learning_rate * weight_grads[i])
    new_bias = bias - learning_rate * bias_grad
    return(new_weights,new_bias)

def train_neuron(inputs, weights, bias, target, learning_rate, steps):
    for step in range(steps):
        prediction = neuron_from_lists(inputs,weights,bias)
        loss = calculate_loss(target,prediction)
        print(step, loss)
        weight_grads,bias_grad = neuron_gradients(inputs,weights,bias,target)
        weights, bias = update_neuron(weights,bias,weight_grads,bias_grad,learning_rate)
    return (weights, bias)

def layer_from_lists(inputs, weights, biases):
    outputs = []
    if len(biases) != len(weights):
        raise ValueError("Different number of weights and biases")
    for i in range(len(biases)):
       outputs.append(neuron_from_lists(inputs,weights[i],biases[i]))
    return outputs

print(train_neuron([4, 5], [1, 2], 0, 10, 0.01, 5))
