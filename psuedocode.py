x = input
h = previous_input

X_in = Vx + Uh
net_in = W_in * X_in
y_in = sigmoid(net_in)
g = sigmoid(X_in)
g*y_in = sigmoid(X_in) * sigmoid(W_in * X_in)

state[t] = state[t-1] + g*y_in

net_out = W_out * X_in
y_out = sigmoid(net_out)

y_cell_output = y_out * tanh(state[t])

net_network_output = W_network_output * y_cell_output
y_network_output = softmax(net_network_output)