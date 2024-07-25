import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from matplotlib import font_manager
from webencodings import labels

Target_Points = np.array([[-260, -260, 450],
                          [70, -540, 400],
                          [70, -540, 345],
                          [70, -540, 400],
                          [-400, -400, 400],
                          [-400, -400, 152],
                          [-400, -400, 400],
                          [-260, -260, 300]]) / 1000

text_offset = np.array([[10, 10],
                        [10, 10],
                        [10, 10],
                        [10, 10],
                        [-15, -15],
                        [10, 10],
                        [10, 10],
                        [-15, -15]]) / 1000


def draw_path_points():
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    for i in range(8):
        # print([Target_Points[i - 1, 0], Target_Points[i, 0]], [Target_Points[i - 1, 1], Target_Points[i, 1]],
        #       [Target_Points[i - 1, 2], Target_Points[i, 2]])
        ax.quiver(Target_Points[i - 1, 0], Target_Points[i - 1, 1], Target_Points[i - 1, 2],
                  Target_Points[i, 0] - Target_Points[i - 1, 0],
                  Target_Points[i, 1] - Target_Points[i - 1, 1],
                  Target_Points[i, 2] - Target_Points[i - 1, 2],
                  color=new_colors[i], arrow_length_ratio=0.2)
        ax.text(Target_Points[i - 1, 0] + text_offset[i, 0], Target_Points[i - 1, 1] + text_offset[i, 1],
                Target_Points[i - 1, 2], str(i), size=12, zorder=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Target Points', size=14, y=0.85)
    ax.view_init(elev=5, azim=30)
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    plt.show(block=True)
    fig.savefig('./figures/target_points.tiff', bbox_inches='tight', dpi=600, transparent=True, format='tiff')


def draw_result_path(path_data: np.array, exp_name, save=False):
    fig = plt.figure(figsize=(10, 10))
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    ax = plt.axes(projection='3d')
    ax.scatter3D(-0.26, -0.26, 0.300, c="k", marker='o', s=10)
    path_data = path_data / 1000
    ax.plot(path_data[:, 0], path_data[:, 1], path_data[:, 2], color='r')
    for i in range(8):
        ax.text(Target_Points[i - 1, 0] + text_offset[i, 0], Target_Points[i - 1, 1] + text_offset[i, 1],
                Target_Points[i - 1, 2], str(i), size=12, zorder=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Planning Result', size=14, y=0.85)
    ax.view_init(elev=5, azim=30)
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    plt.show(block=True)
    if save:
        fig.savefig("./figures/" + exp_name + "_result.tiff", bbox_inches='tight', pad_inches=0.2, dpi=600, transparent=True, format='tiff')


def draw_mul_result_path(input_labels, *traj_data_paths, save=False):
    colors = ["#E44A33", "#4DBAD6"]
    assert len(colors) == len(input_labels)
    fig = plt.figure()
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    ax = plt.axes(projection='3d')
    ax.scatter3D(-0.26, -0.26, 0.300, c="k", marker='o', s=10)

    for i in range(len(traj_data_paths)):
        data = np.load(traj_data_paths[i]) / 1000
        ax.plot(data[:, 0], data[:, 1], data[:, 2], color=colors[i], label=input_labels[i])
    for i in range(8):
        ax.text(Target_Points[i - 1, 0] + text_offset[i, 0], Target_Points[i - 1, 1] + text_offset[i, 1],
                Target_Points[i - 1, 2], str(i), size=12, zorder=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Planning Result', size=14, y=0.85)
    ax.view_init(elev=5, azim=30)
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
    plt.legend(loc=7, ncol=1, prop={'size': 10})
    plt.show(block=True)
    if save:
        fig.savefig("./figures/" +"mul_result.tiff", bbox_inches='tight', dpi=600, transparent=True, format='tiff')



def draw_accumulate_suc_rate(csv_file):
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    df = pd.read_csv(csv_file)
    fig = plt.figure()
    ax = plt.axes()
    cum = np.cumsum(df['Value'].to_numpy())
    suc_rate = cum / (df['Step'].to_numpy() + 1)
    # print(suc_rate)
    ax.plot(df['Step'], suc_rate, color="#A4C8D9")
    # ax.plot([0, 50000], [suc_rate.max(), suc_rate.max()], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Success Episodes Rate', fontsize=12)
    ax.set(xlim=(0, 100001), xticks=np.arange(0, 100001, 10000),
           yticks=np.concatenate((np.arange(0, suc_rate.max(), 0.1), [suc_rate.max()])),
           title='Accumulate Success Rate')
    ax.grid(alpha=0.5)
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    plt.show(block=True)
    if not os.path.exists("./figures/" + csv_file[14:-21]):
        os.mkdir("./figures/" + csv_file[14:-21])
    fig.savefig("./figures/" + csv_file[14:-4] + "suc_episodes.tiff", bbox_inches='tight', dpi=600, transparent=True,
                format='tiff')


def draw_mul_accumulate_suc_rate(input_labels, *csv_files):
    colors = ["#EF767A", "#456990", "#48C0AA"]
    assert len(csv_files) == len(input_labels)
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    dfs = []
    cums = []
    suc_rates = []
    for f in csv_files:
        df = pd.read_csv(f)
        cum = np.cumsum(df['Value'].to_numpy())
        suc_rate = cum / (df['Step'].to_numpy() + 1)
        dfs.append(df)
        cums.append(cum)
        suc_rates.append(suc_rate)
        # print(cum)

    fig = plt.figure()
    ax = plt.axes()

    # print(suc_rate)
    for i in range(len(dfs)):
        ax.plot(dfs[i]['Step'], suc_rates[i], color=colors[i], label=input_labels[i])

    # ax.plot([0, 50000], [suc_rate.max(), suc_rate.max()], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Success Episodes Rate', fontsize=12)
    ax.set(xlim=(0, 100001), xticks=np.arange(0, 100001, 10000),
           yticks=np.concatenate((np.arange(0, np.max(suc_rates), 0.1), [np.max(suc_rates)])),
           title='Accumulate Success Rate')
    ax.grid(alpha=0.5)
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    plt.legend(loc='lower right', ncol=1, prop={'size': 10})
    plt.show(block=True)
    fig.savefig("./figures/mul_suc_episodes.tiff", bbox_inches='tight', dpi=600, transparent=True,
                format='tiff')


def draw_val_suc_rate(csv_file):
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    df = pd.read_csv(csv_file)
    fig = plt.figure()
    ax = plt.axes()
    # cum = np.cumsum(df['Value'].to_numpy())
    # suc_rate = cum / (df['Step'].to_numpy() + 1)
    # print(suc_rate)
    ax.plot(df['Step'], df['Value'], color="#A4C8D9")
    ax.set_xlabel('Val Episode', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set(xlim=(0, 201), xticks=np.arange(0, 201, 20),
           ylim=(0, 1.01), yticks=np.arange(0, 1.01, 0.1),
           title='Val Success Rate')
    ax.grid(alpha=0.5)
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    plt.show(block=True)
    if not os.path.exists("./figures/" + csv_file[14:-21]):
        os.mkdir("./figures/" + csv_file[14:-21])
    fig.savefig("./figures/" + csv_file[14:-4] + "suc_episodes.tiff", bbox_inches='tight', dpi=600, transparent=True,
                format='tiff')


def draw_epi_reward(csv_file):
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    df = pd.read_csv(csv_file)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(df['Step'], df['Value'], color="#4DBAD6")
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set(xlim=(0, 100001), xticks=np.arange(0, 100001, 10000), ylim=(-350, 30), yticks=np.arange(-350, 50, 25),
           title='Trainning Episodes Reward')
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    ax.grid(alpha=0.5)
    plt.show(block=True)
    if not os.path.exists("./figures/" + csv_file[14:-25]):
        os.mkdir("./figures/" + csv_file[14:-25])
    fig.savefig("./figures/" + csv_file[14:-4] + ".tiff", bbox_inches='tight', dpi=600, transparent=True, format='tiff')


def draw_mul_epi_reward(input_labels, *csv_files):
    colors = ["#EF767A", "#456990", "#48C0AA"]
    assert len(csv_files) == len(input_labels)
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))

    fig = plt.figure()
    ax = plt.axes()

    # print(suc_rate)
    for i in range(len(dfs)):
        ax.plot(dfs[i]['Step'], dfs[i]['Value'], color=colors[i], label=input_labels[i], alpha=0.5)

    # ax.plot([0, 50000], [suc_rate.max(), suc_rate.max()], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set(xlim=(0, 100001), xticks=np.arange(0, 100001, 10000),
           ylim=(-350, 30), yticks=np.arange(-350, 50, 25),
           title='Training Reward')
    ax.grid(alpha=0.5)
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    plt.legend(loc='lower right', ncol=1, prop={'size': 10})
    plt.show(block=True)
    output_name = ""
    for i in range(len(input_labels)):
        output_name += str(input_labels[i]) + "_"
    fig.savefig("./figures/" + output_name + "_epi_reward.tiff", bbox_inches='tight', dpi=600, transparent=True,
                format='tiff')


def draw_mul_val_suc_rate(input_labels, *csv_files):
    colors = ["#EF767A", "#456990", "#48C0AA"]
    assert len(csv_files) == len(input_labels)
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))

    fig = plt.figure()
    ax = plt.axes()

    # print(suc_rate)
    for i in range(len(dfs)):
        ax.plot(dfs[i]['Step'], dfs[i]['Value'], color=colors[i], label=input_labels[i])

    ax.set_xlabel('Val Episode', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set(xlim=(0, 201), xticks=np.arange(0.0, 201, 10),
           ylim=(-0.05, 1.0), yticks=np.arange(0, 1.01, 0.1),
           title='Val Success Rate')
    ax.grid(alpha=0.5)
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    plt.legend(loc='lower right', ncol=1, prop={'size': 10})
    plt.show(block=True)
    output_name = ""
    for i in range(len(input_labels)):
        output_name += str(input_labels[i]) + "_"

    fig.savefig("./figures/" + output_name + "_val_suc_rate.tiff", bbox_inches='tight', dpi=600, transparent=True,
                format='tiff')


def draw_val_epi_reward(csv_file):
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'Arial'
    df = pd.read_csv(csv_file)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(df['Step'], df['Value'], color="#4DBAD6")
    ax.set_xlabel('Val Episode', fontsize=12)
    ax.set_ylabel('Ava Reward', fontsize=12)
    y_min = (np.min(df['Value']) // 25 - 1) * 25
    ax.set(xlim=(0, 201), xticks=np.arange(0, 201, 10),
           ylim=(y_min, 30), yticks=np.arange(y_min, 30, 25),
           title='Validating Average Reward')
    ax.patch.set_facecolor("white")  # 设置 ax1 区域背景颜色
    ax.patch.set_alpha(0.1)  # 设置 ax1 区域背景颜色透明度
    ax.grid(alpha=0.5)
    plt.show(block=True)
    # if not os.path.exists("./figures/" + csv_file[14:-25]):
    #     os.mkdir("./figures/" + csv_file[14:-25])
    fig.savefig("./figures/" + csv_file[14:-4] + ".tiff", bbox_inches='tight', dpi=600, transparent=True, format='tiff')


