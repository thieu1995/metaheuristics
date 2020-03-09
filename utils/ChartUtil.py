import matplotlib.pyplot as plt


def plot_google_trace(y_pred, y_true, fig_title, fig_name):
    print(y_true.shape)
    plt.plot(y_true[0:1500], label="y_true")
    plt.plot(y_pred[1:1501], label="y_predict")
    plt.legend()
    plt.title(fig_title)
    # plt.savefig(fig_name+".pdf")
    # plt.close()
    plt.show()

def plot_historical_loss(loss, fig_title, fig_name):
    plt.plot(loss, label="loss_record")
    plt.legend()
    plt.title(fig_title)
    plt.savefig(fig_name+".pdf")
    plt.close()