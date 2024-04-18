import os
import matplotlib.pyplot as plt


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle='-', color='r')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)


def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')


def plot_validate_curve(episodes, baseline, records, records2, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, linestyle=':', linewidth = 2, color='r', label='ppo')
    plt.plot(episodes, records2, linestyle='--', linewidth = 2, color='y', label='ddqn')
    plt.plot(episodes, baseline, linestyle='-', linewidth = 2, color='c', label='baseline')
    plt.legend()
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)

