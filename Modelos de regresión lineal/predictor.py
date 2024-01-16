data = [(1,1), (2,3), (4, 3)]

def phi(x):
    return [1, x]

def dot(v1, v2):
    return sum(c1*c2 for c1, c2 in zip(v1, v2))

def loss(x, y, w):
    return (predict(w, x) - y) ** 2

def vAdd(v1, v2):
    return [c1+c2 for c1, c2 in zip(v1, v2)]

def pScale(k, v):
    return [k*c for c in v]

def predict(w, x):
    return dot(w, phi(x))

def train_loss(w, data):
    loss_total = 0 
    for x, y in data:
        loss_total += loss(x, y, w)
    
    return loss_total / len(data)

def grad_train_loss(w):
    total = [0, 0]
    for x, y in data:
        total = vAdd(total, pScale(2*(predict(w, x)-y),phi(x)))

    return pScale(1/len(data), total)

def gd_step(w, eta):
    return vAdd(w, pScale(-eta, grad_train_loss(w)))

def testis(w, eta):
    w2 = gd_step(w, eta)
    return w2, train_loss(w2, data)

w = [0, 0]

print(testis(w, 0.05))
        












# ~Mephisto
