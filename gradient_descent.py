import csv
import numpy as np


def get_column_data(filename, column_name):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        column_index = header.index(column_name)
        data = []
        for row in reader:
            data.append(float(row[column_index]))
        return data


def update_w_and_b(spendings, sales, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    N = len(spendings)

    for i in range(N):
        dl_dw += -2 * spendings[i] * (sales[i] - (w * spendings[i] + b))
        dl_db += -2 * (sales[i] - (w * spendings[i] + b))

    # update w and b
    w = w - (1 / float(N)) * dl_dw * alpha
    b = b - (1 / float(N)) * dl_db * alpha

    return w, b


# def train(spendings, sales, w, b, alpha, epochs):
#     for e in range(epochs):
#         w, b = update_w_and_b(spendings, sales, w, b, alpha)
#
#         if e % 400 == 0:
#             print("epoch:", e, "loss:", avg_loss(spendings, sales, w, b))
#     return w, b


def avg_loss(spendings, sales, w, b):
    N = len(spendings)
    total_error = 0.0
    for i in range(N):
        total_error = (sales[i] - (w * spendings[i] + b)) ** 2
    return total_error / float(N)


def predict(x, w, b):
    return w * x + b

def train(x,y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(x,y)
    return model


x = np.array(get_column_data('Advertising.csv', 'radio'))
x = x.reshape(-1,1)
y = get_column_data('Advertising.csv', 'sales')
# w, b = train(x, y, 0.0, 0.0, 0.001, 15000)
model = train(x,y)
x_new = np.array(23.0).reshape(1,-1)
y_new = model.predict(x_new)
print(y_new)
