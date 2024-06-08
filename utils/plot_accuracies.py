import matplotlib.pyplot as plt

# Prediction accuracies of stitched models
data_1 = {
    "ResNet 1": 99.36,
    "ResNet 2": 99.33,
    "Linear": 99.34,
    "Affine": 99.22,
    "MLP": 99.38
}

data_2 = {
    "ViT 0": 99.36,
    "ViT 1": 99.33,
    "Linear": 99.34,
    "Affine": 99.22,
    "MLP": 99.38
}

data_3 = {
    "ViT": 89.86,
    "ResNet": 99.36,
    "Linear": 66.82,
    "Affine": 72.28,
    "MLP": 89.00
}

data_4 = {
    "ResNet": 99.36,
    "ViT": 89.86,
    "Linear": 96.11,
    "Affine": 96.01,
    "MLP": 98.77
}


def plot_data(data, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    colors = ['tab:orange'] * 2 + ['tab:blue'] * (len(data) - 2)
    ax.bar(data.keys(), data.values(), color=colors)
    ax.set_title(title)
    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 107)
    for i, v in enumerate(data.values()):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plot_data(data_1, 'ResNet + ResNet', 'Model', 'Accuracy (%)')
plot_data(data_2, 'ViT + ViT', 'Model', 'Accuracy (%)')
plot_data(data_3, 'ViT + ResNet', 'Model', 'Accuracy (%)')
plot_data(data_4, 'ResNet + ViT', 'Model', 'Accuracy (%)')
