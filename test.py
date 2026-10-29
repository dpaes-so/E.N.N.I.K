def calculate_error(target, prediction):
    return target - prediction

def calculate_loss(target, prediction):
    error = calculate_error(target,prediction)
    loss = error **2
    return loss
  
def simple_neuron(input, weight, bias):
    return input * weight + bias

weight = 2
learning_rate = 0.1
target = 10

for step in range(10):
    prediction = simple_neuron(2, weight, 0)
    error = target - prediction

    weight = weight + learning_rate * error

    print("Step:", step + 1, "Weight:", weight)