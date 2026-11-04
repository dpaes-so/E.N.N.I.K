def relu(input: float) -> float:
    if(input < 0):
        return (0)
    else:
        return(input)

def relu_derivative(x: float) -> float:
    if(x < 0):
        return (0)
    else:
        return(1)
    
def calculate_error(target: float, prediction: float) -> float:
    return target - prediction

def calculate_loss(target: float, prediction: float) -> float:
    error = calculate_error(target,prediction)
    loss = error **2
    return loss

def weighted_sum(inputs: list[float], weights: list[float], bias: float) -> float:
    res = bias
    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have the same length")
    for i in range(len(inputs)):
        res += inputs[i] * weights[i]
    return res

def neuron_from_lists(inputs: list[float], weights: list[float], bias: float) -> float:
    return relu(weighted_sum(inputs,weights,bias))

def neuron_gradients(inputs: list[float], weights: list[float], bias: float, target: float) -> tuple[list[float], float]:
    z = weighted_sum(inputs,weights,bias)
    output = relu(z)
    switch = relu_derivative(z)
    shared = 2 * (output - target) * switch
    weight_grads = []
    for i in range (len(inputs)):
        weight_grads.append (shared * inputs[i])
    bias_grad = shared
    return (weight_grads,bias_grad)

def update_neuron(weights: list[float], bias: float, weight_grads: list[float], bias_grad: float, learning_rate: float) -> tuple[list[float], float]:
    new_weights = []
    for i in range (len(weights)):
        new_weights.append(weights[i] - learning_rate * weight_grads[i])
    new_bias = bias - learning_rate * bias_grad
    return(new_weights,new_bias)

def train_neuron(inputs: list[float], weights: list[float], bias: float, target: float, learning_rate: float, steps: int) -> tuple[list[float], float]:
    for step in range(steps):
        prediction = neuron_from_lists(inputs,weights,bias)
        loss = calculate_loss(target,prediction)
        print(step, loss)
        weight_grads,bias_grad = neuron_gradients(inputs,weights,bias,target)
        weights, bias = update_neuron(weights,bias,weight_grads,bias_grad,learning_rate)
    return (weights, bias)

def dataset_loss(all_inputs: list[list[float]], all_targets: list[float], weights: list[float], bias: float) -> float:
    total = 0
    for i in range (len(all_inputs)):
        total += calculate_loss(all_targets[i],neuron_from_lists(all_inputs[i],weights,bias))
    return total/len(all_inputs)

def dataset_gradients(all_inputs: list[list[float]], all_targets: list[float], weights: list[float], bias: float) -> tuple[list[float], float]:
    weight_totals = [0.0] * len(weights)
    bias_total = 0
    for i in range (len(all_inputs)):
        weight_grad , bias_grad = neuron_gradients(all_inputs[i],weights,bias,all_targets[i])
        bias_total += bias_grad
        for j in range (len(weights)):
            weight_totals[j] += weight_grad[j]
    for k in range (len(weight_totals)):
         weight_totals[k] /= (len(all_inputs))
    bias_total /= (len(all_inputs))
    return weight_totals,bias_total

def train_dataset(all_inputs: list[list[float]], all_targets: list[float], weights: list[float], bias: float, learning_rate: float, steps: int) -> tuple[list[float], float]:
    for step in range(steps):
        loss = dataset_loss(all_inputs,all_targets,weights,bias)
        print(step, loss)
        weight_grads, bias_grad = dataset_gradients(all_inputs,all_targets,weights,bias)
        weights, bias = update_neuron(weights,bias,weight_grads,bias_grad,learning_rate)
    return (weights, bias)

def layer_from_lists(inputs: list[float], weights: list[list[float]], biases: list[float]) -> list[float]:
    outputs = []
    if len(biases) != len(weights):
        raise ValueError("Different number of weights and biases")
    for i in range(len(biases)):
       outputs.append(neuron_from_lists(inputs,weights[i],biases[i]))
    return outputs

print(train_dataset([[1, 1], [2, 1], [3, 2], [4, 5]], [3, 5, 8, 13], [0, 0], 0, 0.01, 3000))