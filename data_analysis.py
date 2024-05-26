import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']


# 参数分析
def parameter_analysis():
    x = ['32', '64', '128', '256', '512']
    y = [
        [0.451, 0.469, 0.478, 0.485, 0.482],
        [0.817, 0.828, 0.835, 0.837, 0.838],
        [0.322, 0.339, 0.347, 0.345, 0.331]
    ]
    labels = ['DB15K', 'FB15K', 'YAGO15K']
    plt.figure(figsize=(15, 5))
    for i in range(len(labels)):
        plt.subplot(1, 3, i + 1)
        plt.plot(x, y[i], marker='^')
        plt.grid()
        plt.xticks(x)
        plt.title(labels[i])
        plt.xlabel('嵌入维度')
        plt.ylabel('MRR')
    # plt.show()
    # 保存svg
    plt.savefig('data/parameter_analysis.svg', format='svg', dpi=1200)


def case_analysis():
    x = ['结构', '文本', '多模态', '最终结果']
    y = ['Los Angeles Lakers', 'Cleveland Cavaliers', 'Golden State Warriors' ]

    data = np.array([
        [0.46, 0.89, 0.95, 0.90],
        [0.64, 0.39, 0.78, 0.65],
        [0.82, 0.05, 0.15, 0.13]
    ])
    plt.xticks(np.arange(len(x)), labels=x)
    plt.yticks(np.arange(len(y)), labels=y)
    plt.title("(LeBron_James, playsFor, ?)")

    plt.imshow(data, cmap='GnBu', interpolation='nearest')
    plt.colorbar()
    plt.tight_layout()
    legend = plt.axis()
    print(legend)
    plt.show()
    # plt.savefig('data/case_analysis.svg', format='svg', dpi=1200)


if __name__ == '__main__':
    # parameter_analysis()
    case_analysis()
