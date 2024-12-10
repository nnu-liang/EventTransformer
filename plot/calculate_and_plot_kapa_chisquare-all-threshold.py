import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 全局字体设置
plt.rcParams.update({
    "font.family": "serif",       # 设置全局字体为衬线字体
    "font.serif": ["Times New Roman"],  # 具体设置为 Times New Roman 或类似字体
    "mathtext.fontset": "stix",   # 数学公式使用 Stix 字体，接近 LaTeX
    "axes.titlesize": 25,         # 标题字体大小
    "axes.labelsize": 20,         # 坐标轴标签字体大小
    "xtick.labelsize": 16,        # x轴刻度字体大小
    "ytick.labelsize": 16,        # y轴刻度字体大小
    "legend.fontsize": 18,        # 图例字体大小
    "legend.title_fontsize": 20   # 图例标题字体大小
})
# 常量
BR_hh_4b = 0.58 * 0.58
L = 126
sigma_b = np.array([3.1318e+05, 9.370e+01, 5.996e+03, 2.506e+01, 3.952e+06, 7.371e+08, 2.135e+02, 4.647e+01])
#sigma_b = np.array([0, 0, 0, 0, 3.952e+06, 0, 0, 0])
# 四组混淆矩阵数据
confusion_matrices = [
    np.array([
        [5.34493333e-01],  # 信号
        [0.00336667],      # tt_2b4j
        [0.0150233],       # tth_4b4j
        [0.00524835],      # ttbb_4b4j
        [0.0246133],       # hbb_4b
        [0.00433468],      # 4b
        [0.000206667],     # 2b2j
        [0.00784333],      # zz_4b
        [0.0400967]        # zh_4b
    ]),
    np.array([
        [5.34493333e-01],  # 信号
        [0.00138333],      # tt_2b4j
        [0.00889333],      # tth_4b4j
        [0.00232883],      # ttbb_4b4j
        [0.01424],         # hbb_4b
        [0.00198321],      # 4b
        [0.0000966667],    # 2b2j
        [0.00325333],      # zz_4b
        [0.02443]          # zh_4b
    ]),
    np.array([
        [5.34493333e-01],  # 信号
        [0.00045],         # tt_2b4j
        [0.00378],         # tth_4b4j
        [0.00072309],      # ttbb_4b4j
        [0.00664],         # hbb_4b
        [0.000736524],     # 4b
        [0.00003],         # 2b2j
        [0.000996667],     # zz_4b
        [0.0106067]        # zh_4b
    ]),
    np.array([
        [5.34493333e-01],  # 信号
        [0.00007],         # tt_2b4j
        [0.00086],         # tth_4b4j
        [0.000118818],     # ttbb_4b4j
        [0.00145667],      # hbb_4b
        [0.000125006],     # 4b
        [0.00000666667],   # 2b2j
        [0.000143333],     # zz_4b
        [0.00234667]       # zh_4b
    ])
]

# 数据：kappa_lambda 和 hh_cross_sections
kappa_lambda = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#hh_cross_sections = [0.2353, 0.1778, 0.1286, 0.0875, 0.05487, 0.006716, 0.007264, 0.01613, 0.03336, 0.05871, 0.09263, 0.1346, 0.1848, 0.2434, 0.3000, 0.3500]
hh_cross_sections = (9.96265e-3*kappa_lambda**2 - 4.84745e-2*kappa_lambda + 7.32829e-2)*1000
hh_cross_section_sm = (9.96265e-3 - 4.84745e-2 + 7.32829e-2)*1000
#hh_cross_sections = [[9.96265e-3*y**2 - 4.84745e-2*y + 7.32829e-2 for y in group] for group in kappa_lambda]
# 四组 hh_accuracies 数据
hh_accuracies_list = [
    [35.5, 37.3, 38.8, 40.3, 43, 54, 53, 58, 30.8, 19.24, 18.8, 20.46, 22.23, 23.57, 23.71, 24.09],
    [29.55, 30.72, 32.44, 33.86, 36.1, 40.44, 46.65, 52.03, 27.17, 14.81, 13.97, 15.39, 16.57, 18.1, 18.49, 18.64],
    [21.6, 21.93, 23.39, 24.68, 26.56, 30.14, 35.79, 41.48, 21.76, 10.48, 9.3, 10.32, 11.05, 12.23, 12.38, 12.4],
    [11.6, 11.41, 12.13, 12.97, 14.03, 16.52, 20.43, 25.3, 13.16, 5.48, 4.64, 4.89, 5.1, 5.61, 5.81, 5.9]
]
hh_accuracies_list = [[x / 100 for x in group] for group in hh_accuracies_list]

# 图例标签
threshold_labels = ["Threshold = 0", "Threshold = 0.5", "Threshold = 0.7", "Threshold = 0.9"]

# 颜色列表
colors = ["purple", "blue", "green", "orange"]

# 绘制曲线
plt.figure(figsize=(20, 20))
chi_square_targets = [1, 3.841]

for i, (hh_accuracies, confusion_matrix_new) in enumerate(zip(hh_accuracies_list, confusion_matrices)):
    # 这里的hh_accuracies是一个list，相当于是每次处理一个hh_accuracies_list内部的一个list，confusion_matrix_new同理
    #sigma_di_higgs_values = [x * 2400 for x in hh_cross_sections]
    confusion_matrix_ps_values = hh_accuracies

    # 计算 chi-square
    results = []
    # 空的list，用来存储kappa_lambda 和 chi-square的值，list中每一个值都是一个列表，（kappa，chi_square_value）
    for kappa, sigma_di, ps in zip(kappa_lambda, hh_cross_sections, confusion_matrix_ps_values):
        pb = confusion_matrix_new[1:, 0]  # 背景部分
        numerator = (ps * BR_hh_4b * L * (sigma_di - hh_cross_section_sm)) ** 2
        denominator = ps * hh_cross_section_sm * BR_hh_4b * L + np.sum(pb * sigma_b * L)
        chi_square_value = numerator / denominator if denominator > 0 else float('inf')
        results.append((kappa, chi_square_value))

    # 提取 kappa_lambda 和 chi_square 两个新的数组，分别存储kappa和chi_square
    kappa_lambda = np.array([result[0] for result in results])
    chi_square = np.array([result[1] for result in results])

    # 插值
    interpolation_func = interp1d(kappa_lambda, chi_square, kind='cubic', fill_value="extrapolate")
    #interp1d是一个用来插值的函数，kappa_lambda作为x轴，chi_square作为纵轴，cubic参数表示三次样条插值，extrapolate参数表示超出给定范围也会推算
    kappa_lambda_fine = np.linspace(-7, 14, 500)
    #np.linspace可以用来产生密集的点，相当于是产生更密集的x轴上的数据，这样我们就可以根据前面的到的插值函数，基于kappa_lambda_fine来算出更密集的y轴对应的数据
    chi_square_fine = interpolation_func(kappa_lambda_fine)

    # 找交点
    intersections = {}
    for target in chi_square_targets:
        indices = np.where(np.diff(np.sign(chi_square_fine - target)))[0]
        #找到交点的索引
        intersections[target] = [kappa_lambda_fine[idx] for idx in indices]
        #根据索引的到交点的kappa_lambda值

    # 格式化交点坐标，这是一个字符串列表，用于固定图例中的格式，chi-sqaure = target，（*，*）
    formatted_intersections = [
        f"χ²={target}, ({', '.join([f'{val:.2f}' for val in intersections[target]])})"
        for target in chi_square_targets
    ]

    # 绘制曲线和添加图例的信息，这里的i从前面的threshold_labels取，一共有0，0.5，0.7，0.9四个值
    plt.plot(kappa_lambda_fine, chi_square_fine, label=f"{threshold_labels[i]}: {', '.join(formatted_intersections)}", color=colors[i], linewidth=2)
    #绘制原始的数据点，s=50表示点的大小
    plt.scatter(kappa_lambda, chi_square, color=colors[i], s=50)  # 数据点

# 添加水平线，使用不同颜色
plt.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label="$\\chi^2 = 1$")
plt.axhline(y=3.841, color='red', linestyle='--', linewidth=1.5, label="$\\chi^2 = 3.841$")

# 设置纵轴和横轴范围
plt.ylim(0, 4)
plt.xlim(-8, 14)

# 添加标题、标签和图例
plt.xlabel(r'$\kappa_\lambda$', fontsize=30)
plt.ylabel(r'$\chi^2$', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

legend_text = (
    r"$\mathcal{L}_{\text{int}} = 126\ \mathrm{fb^{-1}}$" ,
    r"model: BP2"
)
# 调整图例位置
plt.legend(fontsize=15, loc='center', bbox_to_anchor=(0.5, 0.7), title="\n".join(legend_text), title_fontsize=18)
plt.grid()

# 显示图形
plt.show()


