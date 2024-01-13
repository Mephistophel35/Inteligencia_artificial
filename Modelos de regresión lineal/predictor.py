def phi(x):
    return [1, x]

def dot(v1, v2):
    return (v1[0]*v2[0]) + (v1[1]*v2[1])

def Loss(x, y, w):
    return (dot(w, phi(x))-y) ** 2

def vAdd(v1, v2):
    return [ v1[0] + v2[0] , v1[1] + v2[1] ]

def pScale(k, v):
    return [k*v[0], k*v[1]]

def gradiantDescent(epochs, training_rate, data):

    w = [0,0]

    for epoch in range(epochs):
        
        for i in data:

            x, y = i

            grad = pScale(2, phi(x))
            grad = pScale(dot(w, phi(x)) - y, grad)

            w = vAdd(w, pScale(-training_rate, grad))
    
    return w



data = [(1,1), (2,3), (4, 3)]

final_weights = gradiantDescent(9000, 0.00015, data)

print("Pesos finales: ", final_weights)










# ~Mephisto
