import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines


baseline_dcevae_factual = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_True/False_baseline_factual"
baseline_dcevae_counter = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_True/False_baseline_counter"
dcevae_factual = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_True/False_dcevae_factual"
dcevae_counter = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_True/False_dcevae_counter"
i_dcevae_factual = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_True/True_dcevae_factual"
i_dcevae_counter = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_True/True_dcevae_counter"
cf_factual = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_False/False_l2_factual"
cf_counter = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_False/False_l2_counter"
reg_factual = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_True/False_reg_factual"
reg_counter = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_True/False_reg_counter"
ours_factual = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_False/False_ours_factual"
ours_counter = "DCEVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_h_0.4_a_f_0.2_u_1.0_ur_3_ud_4_run_1_use_label_False/False_ours_counter"


"""
baseline_dcevae_factual = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_True/False_baseline_factual"
baseline_dcevae_counter = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_True/False_baseline_counter"
dcevae_factual = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_True/False_cvae_factual"
dcevae_counter = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_True/False_cvae_counter"
i_dcevae_factual = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_True/True_cvae_factual"
i_dcevae_counter = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_True/True_cvae_counter"
cf_factual = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_False/False_l2_factual"
cf_counter = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_False/False_l2_counter"
reg_factual = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_True/False_reg_factual"
reg_counter = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_True/False_reg_counter"
ours_factual = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_False/False_ours_factual"
ours_counter = "CVAE/law_result/a_r_1.0_a_d_1.0_a_y_1.0_a_f_0.15_u_7_run_1_use_label_False/False_ours_counter"
"""

factuals = [baseline_dcevae_factual, dcevae_factual, i_dcevae_factual, cf_factual, reg_factual, ours_factual]
counters = [baseline_dcevae_counter, dcevae_counter, i_dcevae_counter, cf_counter, reg_counter, ours_counter]

plt.rcParams['font.size'] = '25'

def plot_single_graph(factual, counter, ax, title):
    factual_data = np.load(factual + ".npy")
    counter_data = np.load(counter + ".npy")
    kde1 = gaussian_kde(factual_data)
    kde2 = gaussian_kde(counter_data)
    
    x = np.linspace(0.0, 0.5, 100)
    density1 = kde1(x)
    density2 = kde2(x)

    ax.plot(x, density1, color="b", label="factual")
    ax.plot(x, density2, color="r", label="counterfactual")

    ax.fill_between(x, density1, color='b', alpha=0.2)
    ax.fill_between(x, density2, color='r', alpha=0.2)
    
    #ax.legend()
    ax.set_xlabel("$\hat{\mathrm{FYA}}$")
    ax.set_ylabel("density")
    ax.set_title(title)

fig, axs = plt.subplots(2, 3, figsize=(26, 10))


titles = ["Baseline", "CA", "ICA", "CE", "CR", "Ours"]
for i, title in enumerate(titles):
    plot_single_graph(factuals[i], counters[i], axs[i // 3][i % 3], title)
#plt.subplots_adjust(left=None, bottom=0.15, right=None, top=0.85, wspace=0.3, hspace=0.8)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
lineA = mlines.Line2D([], [], color='b', label='factual')
lineB = mlines.Line2D([], [], color='r', label='counterfactual')
# Create a legend for the whole figure
fig.legend(handles=[lineA, lineB], loc='upper center', ncol=2)

plt.show()
plt.savefig("density_dcevae.pdf")
#plt.savefig("density_cvae.pdf")