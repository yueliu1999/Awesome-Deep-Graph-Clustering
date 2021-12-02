import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_embedding(embeds, labels, title, dataset):
    """
    :param embeds: embedding of the data
    :param labels: labels
    :param title:图像标题
    :return:图像
    """
    mean = np.mean(embeds, axis=0)
    std = np.std(embeds, axis=0)

    data1 = embeds
    data = []
    for i in range(len(data1)):
        if (data1[i]-mean < 3 * std).all():
            data.append(data1[i])
    data = np.array(data)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        if dataset == "cite":
            labels[i] -= 1
            plt.text(data[i, 0], data[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i] % 7),
                     fontdict={'weight': 'bold', 'size': 7})
        else:
            plt.text(data[i, 0], data[i, 1], str(labels[i]), color=plt.cm.Set1(labels[i] % 7),
                     fontdict={'weight': 'bold', 'size': 7})
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=14)
    plt.axis('off')
    return fig

if __name__ == '__main__':
    for dataset in ["acm", "dblp", "cite", "amap", "pubmed", "corafull", "amac"]:
        if dataset in ["amap", "amac"]:
            x = np.load("./data/{}/feat.npy".format(dataset))
            y = np.load("./data/{}/labels.npy".format(dataset))
        else:
            x = np.loadtxt("data/{}/{}.txt".format(dataset, dataset), dtype=float)
            y = np.loadtxt("data/{}/{}_label.txt".format(dataset, dataset), dtype=int)
        np.random.seed(42)
        sample_index = np.random.randint(0, x.shape[0], 2000)
        x = x[sample_index]
        y = y[sample_index]
        ts = TSNE(n_components=2, init='pca', random_state=0)
        print("Transforming:")
        result = ts.fit_transform(x[:, :])
        print("OK")
        # 调用函数，绘制图像
        fig = plot_embedding(result, y, '', dataset)
        # 显示图像

        plt.savefig("./result/{}.png".format(dataset))
        # plt.show()


