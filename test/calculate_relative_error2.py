import numpy as np
import math

# 混淆矩阵
confusion_matrix = np.array([
    [9.99939999e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00002000e-05, 3.00006000e-05, 2.00004000e-05, 0.00000000e+00],
    [0.00000000e+00, 9.99689984e-01, 0.00000000e+00, 8.00040002e-05, 0.00000000e+00, 0.00000000e+00, 4.00020001e-05, 1.90009500e-04, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 9.99909994e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00007000e-05, 8.00056004e-05, 0.00000000e+00],
    [0.00000000e+00, 9.45847292e-02, 4.39021951e-02, 7.32806640e-01, 8.09040452e-03, 4.47022351e-03, 4.82024101e-02, 9.40047002e-04, 6.70033502e-02],
    [5.90011800e-04, 1.40002800e-04, 0.00000000e+00, 2.00004000e-05, 9.27018540e-01, 1.39002780e-03, 4.82509650e-02, 2.24504490e-02, 1.40002800e-04],
    [1.20004800e-04, 4.84719389e-02, 0.00000000e+00, 1.01004040e-03, 0.00000000e+00, 8.52764111e-01, 7.82031281e-03, 8.97035881e-02, 1.10004400e-04],
    [2.82102821e-02, 1.07501075e-02, 1.17891179e-01, 2.40702407e-02, 1.32121321e-01, 1.00091001e-01, 3.52423524e-01, 7.20407204e-02, 1.62401624e-01],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.00009000e-05, 1.09003270e-03, 9.00027001e-04, 9.97659930e-01, 3.20009600e-04],
    [7.00021001e-05, 1.17003510e-02, 6.00018001e-04, 1.14703441e-02, 1.39004170e-03, 2.45007350e-03, 1.06203186e-02, 3.62310869e-02, 9.25467764e-01]
])

# 计算 S_predictS 和 B_predictS
def compute_predict_values(S_sim, B_sim, confusion_matrix):
    P_signal_correct = confusion_matrix[0, 0]
    P_background_wrong = confusion_matrix[1:, 0]
    
    S_predictS = S_sim * P_signal_correct
    B_predictS = np.sum(B_sim * P_background_wrong)
    
    return S_predictS, B_predictS

# 计算 chi-square
def compute_chi_squared(S, S_sm, zeta_s):
    return (S - S_sm)**2 / (S_sm + (zeta_s * S_sm)**2)

# 计算 sigma_S
def compute_sigma_S(S_predictS, S_exp, S_sim, B_predictS, B_exp, B_sim):
    partial_S_sim = - S_predictS * S_exp / S_sim**2
    partial_B_sim = - B_predictS * B_exp / (B_sim[0]**2)  # 由于每个背景的数量是相同的，可以使用 B_sim[0]
    sigma_S_sim = np.sqrt(S_sim)
    sigma_B_sim = np.sqrt(B_sim[0])  # 由于每个背景的数量是相同的，可以使用 B_sim[0]
    
    sigma_S_squared = (partial_S_sim**2 * sigma_S_sim**2) + (partial_B_sim**2 * sigma_B_sim**2)
    return np.sqrt(sigma_S_squared)

# 计算相对误差
def compute_relative_error(S_sm, zeta_s, S_predictS, S_exp, S_sim, B_predictS, B_exp, B_sim):
    # 使用公式计算 S
    S = S_predictS * S_exp / S_sim + B_predictS * B_exp / B_sim.sum()
    
    chi_squared = compute_chi_squared(S, S_sm, zeta_s)
    
    partial_chi_squared_S = 2 * (S - S_sm) / (S_sm + (zeta_s * S_sm)**2)
    
    sigma_S = compute_sigma_S(S_predictS, S_exp, S_sim, B_predictS, B_exp, B_sim)
    sigma_chi_squared = abs(partial_chi_squared_S) * sigma_S
    
    relative_error = sigma_chi_squared / chi_squared
    return relative_error

# 示例赋值
S_sim = 500000.0  # 模拟信号数量
B_sim = np.full(8, 500000.0)  # 每个模拟背景的数量，总共8个背景

# 计算 S_predictS 和 B_predictS
S_predictS, B_predictS = compute_predict_values(S_sim, B_sim, confusion_matrix)

# 其他参数
S_sm = 1.23e+04  # 标准模型下的信号数量
zeta_s = 0.003  # 系统误差
S_exp = 1.23e+04  # 实验信号数量
B_exp = 1.71e+12  # 实验背景数量

# 计算相对误差
relative_error = compute_relative_error(S_sm, zeta_s, S_predictS, S_exp, S_sim, B_predictS, B_exp, B_sim)
print(f"Chi-square Relative Error: {relative_error}")

