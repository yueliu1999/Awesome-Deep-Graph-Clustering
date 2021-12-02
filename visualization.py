import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def t_sne(embeds, labels, sample_num=2000, show_fig=True):
    """
    visualize embedding by t-SNE algorithm
    :param embeds: embedding of the data
    :param labels: labels
    :param sample_num: the num of samples
    :param show_fig: if show the figure
    :return: figure
    """

    # sampling
    sample_index = np.random.randint(0, embeds.shape[0], sample_num)
    sample_embeds = embeds[sample_index]

    # t-SNE
    ts = TSNE(n_components=2, init='pca', random_state=0)
    ts_embeds = ts.fit_transform(sample_embeds[:, :])

    # remove outlier
    mean, std = np.mean(ts_embeds, axis=0), np.std(ts_embeds, axis=0)
    for i in range(len(ts_embeds)):
        if (ts_embeds[i] - mean < 3 * std).all():
            np.delete(ts_embeds, i)

    # normalization
    x_min, x_max = np.min(ts_embeds, 0), np.max(ts_embeds, 0)
    norm_ts_embeds = (ts_embeds - x_min) / (x_max - x_min)

    # plot
    fig = plt.figure()
    for i in range(norm_ts_embeds.shape[0]):
        plt.text(norm_ts_embeds[i, 0], norm_ts_embeds[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i] % max(labels)),
                 fontdict={'weight': 'bold', 'size': max(labels)})
    plt.xticks([])
    plt.yticks([])
    plt.title('t-SNE', fontsize=14)
    plt.axis('off')
    if show_fig:
        plt.show()

    return fig
