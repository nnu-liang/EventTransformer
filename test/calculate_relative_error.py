import numpy as np

def calculate_S_B_predict(confusion_matrix, label_counts):
    S_predictS = confusion_matrix[0, 0] * label_counts[0]
    
    B_predictS = 0
    for i in range(1, 9):
        B_predictS += confusion_matrix[i, 0] * label_counts[i]
    
    return S_predictS, B_predictS

def relative_chi_squared_error(confusion_matrix, label_counts, S_exp, S_sim, B_exp, B_sim, 
                               S_SM, zeta_s, zeta_b):
    S_predictS, B_predictS = calculate_S_B_predict(confusion_matrix, label_counts)
    
    S_predictB = S_sim - S_predictS
    B_predictB = B_sim - B_predictS
    
    S = S_predictS * (S_exp / S_sim) + B_predictS * (B_exp / B_sim)
    B = S_predictB * (S_exp / S_sim) + B_predictB * (B_exp / B_sim)
    
    sigma_S_sim = np.sqrt(S_sim)
    sigma_B_sim = np.sqrt(B_sim)
    
    Delta_S = np.sqrt((-S_predictS * (S_exp / S_sim**2) * sigma_S_sim)**2 + 
                      (-B_predictS * (B_exp / B_sim**2) * sigma_B_sim)**2)
    
    Delta_B = np.sqrt((-S_predictB * (S_exp / S_sim**2) * sigma_S_sim)**2 + 
                      (-B_predictB * (B_exp / B_sim**2) * sigma_B_sim)**2)
    
    chi2 = (S - S_SM)**2 / (S + (zeta_s * S)**2 + B + (zeta_b * B)**2)
    

    dchi2_dS = (2 * (S - S_SM) * (S + (zeta_s * S)**2 + B + (zeta_b * B)**2) - 
                (S - S_SM)**2 * (1 + 2 * zeta_s**2 * S)) / (S + (zeta_s * S)**2 + B + (zeta_b * B)**2)**2
    
    dchi2_dB = (-(S - S_SM)**2 * (1 + 2 * zeta_b**2 * B)) / (S + (zeta_s * S)**2 + B + (zeta_b * B)**2)**2
    
    Delta_chi2 = np.sqrt((dchi2_dS * Delta_S)**2 + (dchi2_dB * Delta_B)**2)
    
    relative_error = Delta_chi2 / chi2
    
    return relative_error, S_predictS, B_predictS

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

label_counts = np.array([100000, 2000000, 1000000, 1000000, 1000000, 8000000, 7000000, 1000000, 1000000])

S_exp = 13483.2
S_sim = 1000000
B_exp = 74529239173
B_sim = 22000000
S_SM = 13483.2
zeta_s = 0
zeta_b = 0.003

relative_error, S_predictS, B_predictS = relative_chi_squared_error(confusion_matrix, label_counts, S_exp, S_sim, B_exp, B_sim, 
                                                                    S_SM, zeta_s, zeta_b)

print(f"χ² 的相对误差 = {relative_error}")

