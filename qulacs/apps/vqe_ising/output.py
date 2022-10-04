def output(param_history, cost_history, iter_history):
    print(param_history)
    print(cost_history)
    print(iter_history)


def to_string(two_dim_array):
    s = ""
    for i in range(len(two_dim_array)):
        s += ",".join([str(j) for j in two_dim_array[i]])
        if i == len(two_dim_array):
            break

        s += "\n"

    return s
