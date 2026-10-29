def relu(input):
    if(input < 0):
        return (0)
    else:
        return(input)

def neuron(input1, input2, weight1,weight2, bias):
    res = input1 * weight1 + input2 * weight2 + bias
    return relu(res)

def layer(input1,input2):
    res1 = neuron(input1, input2, 1, 1, 0)
    res2 = neuron(input1, input2, 2, -1, 0)
    return res1,res2

def network(input1,input2):
    rl1a, rl1b = layer(input1,input2)
    rl2a, rl2b = layer(rl1a,rl1b)
    return rl2a,rl2b

def calculate_error(target, prediction):
    return target - prediction

def calculate_loss(target, prediction):
    error = calculate_error(target,prediction)
    loss = error **2
    return loss


print(calculate_loss(1.0, 0.8))  # Expected: 0.04
print(calculate_loss(1.0, 0.2))  # Expected: 0.64
print(calculate_loss(1.0, 1.0))  # Expected: 0.0
output1, output2 = network(2, 3)
print(output1)
print(output2)

