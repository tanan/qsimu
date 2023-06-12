import matplotlib.pyplot as plt

def create_qcl_graph(x_init, y_init, x_train, y_train, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(x_init, y_init, "--", label="Initial Model Prediction", c="gray")
    plt.plot(x_train, y_train, "o", label="Teacher")
    plt.plot(x_init, y_pred, label="Final Model Prediction")
    plt.legend()
    plt.show()
