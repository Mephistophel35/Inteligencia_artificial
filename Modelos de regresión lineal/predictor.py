def phi(x):
    return [1, x]

def dot(v1, v2):
    sum = 0

    for i in range(len(v1)):
        sum += (v1[i]*v2[i])
    
    return sum

def res(w, x, y):
    return (dot(w, phi(x)) - y)

def Loss(w, x, y):
    return pow(res(w, x, y),2)

def TrainLoss(w, data):

    total_loss = 0

    for i in data:
        x, y = i
        loss = Loss(w, x, y)
        total_loss += loss

    average_loss = total_loss / len(data)

    return average_loss

def gradient_descent(data, learning_rate , ephocs):

    w = [0, 0]

    m = len(data)

    for ephocs in range(ephocs):
        total_loss = 0
    
        for i in data:
            x, y = i
            prediction = dot(w, phi(x))
            loss = res(w, x, y)
            
            for i in range(len(w)):
                w[i] = w[i] - learning_rate * loss * phi(x)[i]
            
            total_loss += Loss(w, x, y)
        
        average_loss = total_loss / m
    
    return w