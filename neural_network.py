def relu(input):
    if(input < 0):
        return (0)
    else:
        return(input)
    
def calculate_error(target, prediction):
    return target - prediction

def calculate_loss(target, prediction):
    error = calculate_error(target,prediction)
    loss = error **2
    return loss

def neuron_from_lists(inputs, weights, bias):
    res = bias

    if len(inputs) != len(weights):
         raise ValueError("Inputs and weights must have the same length")
    for i in range(len(inputs)):
        res += inputs[i] * weights[i]
    return relu(res)



def layer_from_lists(inputs, weights, biases):
    outputs = []
    if len(biases) != len(weights):
        raise ValueError("Different number of weights and biases")
    for i in range(len(biases)):
       outputs.append(neuron_from_lists(inputs,weights[i],biases[i]))
    return outputs


inputs = [4, 5]

weights = [
    [1, 2],
    [2, -1],
    [0.5, 3]
]

biases = [0, 1, -1]

print(layer_from_lists(inputs, weights, biases))
target = 10
prediction = 14

for weight in [1, 2, 2.5, 3]:
    output = neuron_from_lists([4], [weight], 0)
    print("w =",weight)
    print("Loss:", calculate_loss(target, output))

