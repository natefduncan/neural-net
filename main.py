from nn import MLP, draw

if __name__=="__main__":
    n = MLP(3, [4, 4, 1])

    # 4 inputs
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5], 
        [0.5, 1.0, 1.0], 
        [1.0, 1.0, -1.0]
    ]

    # binary classifier neural net
    ys = [1.0, -1.0, 1.0, 1.0] # desired targets

    # train
    for k in range(1000):
        ypred = [n(x) for x in xs]

        # mean squared error loss; forward pass
        loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

        # backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -0.05 * p.grad

        # print(k, loss.data)

    print(ypred)
