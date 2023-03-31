def update_w_and_b(spendings, sales, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    N = len(spendings)

    for i in range(N):
        dl_dw += -2 * spendings[i] * (sales[i] - (w * spendings[i] + b))
        dl_db += -2 * (sales[i](w * spendings[i] + b))

    # update w and b
    w = w - (1 / float(N)) * dl_dw * alpha
    b = b - (1 / float(N)) * dl_db * alpha

    return w, b


def train(spendings, sales, w, b, alpha, epochs):
    for e in range(epochs):
        w, b = update_w_and_b(spendings, sales, w, b, alpha)

        if e % 400 == 0:
            print("epoch:", e, "loss:", avg_loss(spendings, sales, w, b))
    return w, b


def avg_loss(spendings, sales, w, b):
    N = len(spendings)
    total_error = 0.0
    for i in range(N):
        total_error = (sales[i] - (w * spendings[i] + b)) ** 2
    return total_error / float(N)
