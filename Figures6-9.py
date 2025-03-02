import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import random
from statsmodels import robust
import matplotlib.patches as mpatches
import scipy.stats as stats

plt.rcParams.update({'font.size': 15,'font.family':'serif'})
plt.rcParams['mathtext.fontset'] = 'dejavuserif'  # 设置LaTeX字体为Computer Modern
cmap = plt.get_cmap('coolwarm')

def create_learning_curve(scene):
    RL_agent=[]
    RL_list=[]
    for i in [1,2,3,4,5,6,7,8,9,10]:
        detect=np.load(r'.\Human-in-loop Data\{}\total_rewards_list_test{}.npy'.format(scene,i),allow_pickle=True)
        detect = np.hstack(detect)
        detect = detect.reshape((30, 2))
        RL_agent.append(detect[:, 1])

        RL = np.load(r'.\Pure RL Data\{}\total_rewards_list_test{}.npy'.format(scene,i), allow_pickle=True)
        RL = np.hstack(RL[:30])
        RL_list.append(RL)

    RL_agent=np.array(RL_agent)
    RL_list=np.array(RL_list)

    RL_agent_mean=np.mean(RL_agent,axis=0)
    RL_list_mean=np.mean(RL_list,axis=0)

    x=np.linspace(1,30,30)
    # Calculate Standard Error
    RL_agent_sde = np.std(RL_agent, axis=0)/np.sqrt(10)
    RL_list_sde = np.std(RL_list, axis=0)/np.sqrt(10)


    RL_agent_upper = RL_agent_mean + RL_agent_sde
    RL_agent_lower = RL_agent_mean - RL_agent_sde
    RL_list_upper = RL_list_mean + RL_list_sde
    RL_list_lower = RL_list_mean - RL_list_sde

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(x, RL_list_mean, label='w/o fNIRS', color=cmap(0.0625))
    plt.fill_between(x, RL_list_lower, RL_list_upper, alpha=0.8, facecolor=cmap(0.0625+0.25))
    plt.plot(x, RL_agent_mean, label='w/ fNIRS', color=cmap(0.8125+0.125))
    plt.fill_between(x, RL_agent_lower, RL_agent_upper, alpha=0.8, facecolor=cmap(0.8125+0.125-0.25))

    plt.xlabel('Episodes')
    plt.ylabel('Rewards')

    plt.xlim(1,30)
    plt.grid()
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=5,handletextpad=0)
    if scene=='aeb':
        sub_no='a'
    elif scene=='cutin':
        sub_no='b'
    else:
        sub_no='c'
    plt.savefig('Figure6({}).pdf'.format(sub_no), format='pdf', bbox_inches='tight')
    plt.show()

for scene in ['aeb','cutin','ped']:
    create_learning_curve(scene)


def calculate_ttc(obs):
    obs[:, 0] = obs[:, 0] / 3.6
    obs[:, 4] = obs[:, 4] / 3.6
    ttc_list=[]
    for i in range(25,len(obs)):
        if abs(obs[i, 0]) > abs(obs[i, 4]):
            distance = max(abs(obs[i, 2] - obs[i, 6])-4,0)
            velocity = abs(obs[i, 0] - obs[i, 4])
            ttc=min(distance/velocity,10)
        else:
            ttc=10#float('inf')
        # if ttc<10:
        ttc_list.append(ttc)
    ttc_list=np.array(ttc_list)
    return ttc_list

def calculate_jerk(obs,time):
    velocity=obs[25:, 0]/3.6
    time=time[25:]
    first_derivative = np.gradient(velocity, time)

    second_derivative = np.gradient(first_derivative, time)
    positive_values = second_derivative[second_derivative > 0]
    negative_values = second_derivative[second_derivative < 0]

    return np.mean(positive_values),np.mean(negative_values)


def get_risk_field(obs):
    # risk_field = 0
    G = 0.001
    # k1 = 1
    k2 = 0.05
    M = 1705  # 920
    risk_fields=[]
    for i in range(len(obs)):
        ego_x = obs[i,2]
        ego_y = obs[i,3]


        actor_x = obs[i,6]
        actor_y = obs[i,7]
        actor_vx = obs[i,4]
        actor_vy = obs[i,5]
        distance = np.sqrt((ego_x - actor_x) ** 2 + (ego_y - actor_y) ** 2)
        actor_v = np.sqrt(actor_vx ** 2 + actor_vy ** 2)
        if actor_v!=0:
            cos_theta = ((actor_vx * (ego_x - actor_x) + actor_vy * (ego_y - actor_y)) / (distance * actor_v))
            risk_field = G * M / distance * np.exp(k2 * cos_theta * actor_v/3.6)
        else:
            risk_field =G*M/distance
        risk_fields.append(risk_field)

    return risk_fields

def process_data(scene):
    ttc_min_RL=[]
    ttc_min_fNIRS=[]
    ttc_danger_RL=[]
    ttc_danger_fNIRS=[]
    jerkpos_RL=[]
    jerkpos_fNIRS=[]
    jerkneg_RL=[]
    jerkneg_fNIRS=[]
    risk_RL=[]
    risk_fNIRS=[]

    for i in [1,2,3,4,5,6,7,8,9,10]:
        print(i)
        obs1 = np.load(r'.\Pure RL Data\{}\observation_test_total{}.npy'.format(scene,i), allow_pickle=True)
        time1 = np.load(r'.\Pure RL Data\{}\time_test_total{}.npy'.format(scene,i), allow_pickle=True)
        for j in range(0,30):
            obs1j = abs(obs1[j])
            if j in range(0,30):
                ttc1=calculate_ttc(obs1j)
                ttc_min_RL.append(min(ttc1))
                if j in range(25,30):
                    posjerk1,negjerk1=calculate_jerk(obs1j,time1[j])
                    jerkpos_RL.append(posjerk1)
                    jerkneg_RL.append(negjerk1)

            risk1 = get_risk_field(obs1j)
            risk_RL.append(max(risk1))
        ttc_danger_RL_j=[]

    for i in [1,2,3,4,5,6,7,8,9,10]:
        print(i)
        obs2 = np.load(r'.\Human-in-loop Data\{}\observation_test_total{}.npy'.format(scene,i), allow_pickle=True)
        time2 = np.load(r'.\Human-in-loop Data\{}\time_test_total{}.npy'.format(scene,i), allow_pickle=True)

        for j in range(0,30):
            obs2j = abs(obs2[2 * j])
            if j in range(0,30):
                ttc2=calculate_ttc(obs2j)
                ttc_min_fNIRS.append(min(ttc2))
                if j in range(25, 30):
                    posjerk2,negjerk2=calculate_jerk(obs2j,time2[2*j])
                    jerkpos_fNIRS.append(posjerk2)
                    jerkneg_fNIRS.append(negjerk2)

            risk2 = get_risk_field(obs2j)
            risk_fNIRS.append(max(risk2))

        ttc_danger_fNIRS_j=[]
    return risk_RL, risk_fNIRS, ttc_min_RL, ttc_min_fNIRS, jerkpos_RL, jerkpos_fNIRS, jerkneg_RL, jerkneg_fNIRS

def create_risk_field_plot(risk_RL,risk_fNIRS, scene):
    data1=np.array(risk_RL)
    data2=np.array(risk_fNIRS)
    data1=data1.reshape(10,30)
    data2=data2.reshape(10,30)

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    means1 = np.mean(data1, axis=0)
    stds1 = np.std(data1, axis=0)

    means2 = np.mean(data2, axis=0)
    stds2 = np.std(data2, axis=0)

    data1_higher = means1 + stds1
    data1_lower = means1 - stds1
    data2_higher = means2 + stds2
    data2_lower = means2 - stds2

    x = np.arange(1,31)


    plt.plot(x, means1, label='w/o fNIRS', color=cmap(0.0625))
    plt.fill_between(x, data1_lower, data1_higher, alpha=0.8, facecolor=cmap(0.0625+0.25))  # 填充方差范围
    plt.plot(x, means2, label='w/ fNIRS', color=cmap(0.8125+0.125))
    plt.fill_between(x, data2_lower, data2_higher, alpha=0.8, facecolor=cmap(0.8125+0.125-0.25))  # 填充方差范围
    plt.grid()
    plt.xlim((1,30))
    plt.xlabel('Episodes')
    plt.ylabel('Maximum Risk Field')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=5,handletextpad=0)

    if scene=='aeb':
        fig_no='a'
    elif scene=='cutin':
        fig_no='b'
    else:
        fig_no='c'
    plt.savefig('Figure7({}).pdf'.format(fig_no), format='pdf', bbox_inches='tight')

    plt.show()

# def create_violin_plot(data1, data2, y_label, test_side):
#     fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     box = plt.boxplot([data1, data2], positions=[1, 2], widths=0.25)
#
#     for median in box['medians']:
#         median.set(color='red')
#     violin_parts = plt.violinplot([data1, data2], bw_method="scott", showmedians=False, showextrema=False)
#
#     violin_parts['bodies'][0].set_facecolor(cmap(0.0625 + 0.25))
#     violin_parts['bodies'][0].set_alpha(1)
#     violin_parts['bodies'][1].set_facecolor(cmap(0.8125 - 0.125))
#     violin_parts['bodies'][1].set_alpha(1)
#     plt.grid()
#     print(violin_parts)
#
#     outliers = [flier.get_ydata() for flier in box['fliers']]
#     colors = [cmap(0.0625), cmap(0.8125 + 0.125)]
#
#     for i, data in enumerate([data1, data2]):
#         for idx, val in enumerate(data):
#             if val in outliers[i]:
#                 plt.scatter(i + 1, val, edgecolors='none', facecolors=colors[i], alpha=0.7, marker='o', s=20)
#             else:
#                 plt.scatter(np.random.normal(i + 1, 0.02, 1), val, edgecolors='none', facecolors=colors[i], alpha=0.7,
#                             marker='o', s=20)
#
#     plt.xticks([1, 2], ['w/o fNIRS', 'w/ fNIRS'])
#
#     U_stat, p_val = stats.wilcoxon(data1, data2, alternative=test_side)
#     print(U_stat, p_val)
#
#
#     y_pos = max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2))
#     size = max(max(data1) - min(data1), max(data2) - min(data2))
#     if p_val < 0.001:
#         plt.text(1.5, max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2)),
#                  '***', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
#         plt.text(1.5, max(max(data1), max(data2)) + 0.25 * max(max(data1) - min(data1), max(data2) - min(data2)),
#                  f'p = {p_val:.2e}', ha='center', va='bottom', color='k', fontsize=15)
#         plt.plot([1, 2], [y_pos, y_pos], linewidth=1, color='k')
#         plt.plot([2, 2], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
#         plt.plot([1, 1], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
#     elif p_val < 0.01:
#         plt.text(1.5, max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2)),
#                  '**', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
#         plt.text(1.5, max(max(data1), max(data2)) + 0.25 * max(max(data1) - min(data1), max(data2) - min(data2)),
#                  f'p = {p_val:.3f}', ha='center', va='bottom', color='k', fontsize=15)
#         plt.plot([1, 2], [y_pos, y_pos], linewidth=1, color='k')
#         plt.plot([2, 2], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
#         plt.plot([1, 1], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
#     elif p_val < 0.05:
#         plt.text(1.5, max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2)),
#                  '*', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
#         plt.text(1.5, max(max(data1), max(data2)) + 0.25 * max(max(data1) - min(data1), max(data2) - min(data2)),
#                  f'p = {p_val:.3f}', ha='center', va='bottom', color='k', fontsize=15)
#         plt.plot([1, 2], [y_pos, y_pos], linewidth=1, color='k')
#         plt.plot([2, 2], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
#         plt.plot([1, 1], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
#
#     plt.ylabel(y_label)
#     plt.xlim((0.5, 2.5))
#     plt.show()

def create_ttc_plot1(data1, data2, scene):
    list1a = [x for i, x in enumerate(data1) if i % 30 < 10]
    list1b = [x for i, x in enumerate(data1) if 10 <= i % 30 < 20]
    list1c = [x for i, x in enumerate(data1) if 20 <= i % 30 < 30]

    list2a = [x for i, x in enumerate(data2) if i % 30 < 10]
    list2b = [x for i, x in enumerate(data2) if 10 <= i % 30 < 20]
    list2c = [x for i, x in enumerate(data2) if 20 <= i % 30 < 30]

    data_combined = [list1a, list2a, list1b, list2b, list1c, list2c]

    labels = ['List1 0-9', 'List2 0-9', 'List1 10-19', 'List2 10-19', 'List1 20-29', 'List2 20-29']

    positions=[1,1.6,3,3.6,5,5.6]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=300)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    colors = [cmap(0.0625 + 0.25), cmap(0.8125 - 0.125)]*3

    parts = plt.violinplot(data_combined, positions=positions, showmeans=False, showextrema=True, showmedians=True)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    print(parts.keys())

    for partname in ( 'cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')
    vp=parts['cmedians']
    vp.set_edgecolor('red')
    parts['cbars'].set_linewidth(0.8)


    def remove_outliers(datalist1,datalist2):
        Q11 = np.percentile(datalist1, 25)
        Q31 = np.percentile(datalist1, 75)
        IQR1 = Q31 - Q11

        lower_bound1 = Q11 - 1.5 * IQR1
        upper_bound1 = Q31 + 1.5 * IQR1

        Q12 = np.percentile(datalist2, 25)
        Q32 = np.percentile(datalist2, 75)
        IQR2 = Q32 - Q12

        lower_bound2 = Q12 - 1.5 * IQR2
        upper_bound2 = Q32 + 1.5 * IQR2

        z_scores1 = np.abs((datalist1 - np.mean(datalist1)) / np.std(datalist1))
        z_scores2 = np.abs((datalist2 - np.mean(datalist2)) / np.std(datalist2))
        threshold=3

        median1 = np.median(datalist1)
        mad1 = robust.mad(datalist1)

        median2 = np.median(datalist2)
        mad2 = robust.mad(datalist2)

        datalist1_cleaned=[]
        datalist2_cleaned=[]
        for i in range(len(datalist1)):
            if mad1!=0:
                robust_z_score1 = np.abs((datalist1[i] - median1) / mad1)
            else:
                robust_z_score1=0
            if mad2!=0:
                robust_z_score2 = np.abs((datalist2[i] - median2) / mad2)
            else:
                robust_z_score2=0
            # print(robust_z_score1,robust_z_score2)
            # if z_scores1[i] <= threshold and z_scores2[i] <= threshold:
            #     datalist1_cleaned.append(datalist1[i])
            #     datalist2_cleaned.append(datalist2[i])
            if robust_z_score1 <= threshold and robust_z_score2 <= threshold:
                datalist1_cleaned.append(datalist1[i])
                datalist2_cleaned.append(datalist2[i])
            # if datalist1[i]>lower_bound1 and datalist1[i]<upper_bound1 and datalist2[i]>lower_bound2 and datalist2[i]<upper_bound2:
            #     datalist1_cleaned.append(datalist1[i])
            #     datalist2_cleaned.append(datalist2[i])
            # else:
            #     datalist1_cleaned.append(np.median(datalist1))
            # if datalist2[i]>lower_bound2 and datalist2[i]<upper_bound2:
            #     datalist2_cleaned.append(datalist2[i])
            # else:
            #     datalist2_cleaned.append(np.median(datalist2))
        return datalist1_cleaned,datalist2_cleaned

    legend_patches = [mpatches.Patch(color=cmap(0.0625 + 0.25), label='w/o fNIRS'),
                      mpatches.Patch(color=cmap(0.8125-0.125), label='w/ fNIRS')]

    plt.legend(handles=legend_patches, loc='lower center',bbox_to_anchor=(0.5, -0.45), ncol=2)
    plt.grid()
    plt.xlabel('Episodes')
    plt.ylabel('Minumum TTC (s)')
    plt.xticks([1.3,3.3,5.3],['1-10','11-20','21-30'])



    list1a_cleaned, list2a_cleaned = remove_outliers(list1a, list2a)
    list1b_cleaned, list2b_cleaned = remove_outliers(list1b, list2b)
    list1c_cleaned, list2c_cleaned = remove_outliers(list1c, list2c)

    # plt.boxplot([list1a_cleaned, list2a_cleaned,list1b_cleaned, list2b_cleaned,list1c_cleaned, list2c_cleaned], boxprops=boxprops, labels=labels)

    print("Wilcoxon test for 0-9:", stats.wilcoxon(list1a_cleaned, list2a_cleaned, alternative='less'))
    print("Wilcoxon test for 10-19:", stats.wilcoxon(list1b_cleaned, list2b_cleaned, alternative='less'))
    print("Wilcoxon test for 20-29:", stats.wilcoxon(list1c_cleaned, list2c_cleaned, alternative='less'))
    wilconxon_test_p=[stats.wilcoxon(list1a_cleaned, list2a_cleaned, alternative='less').pvalue,stats.wilcoxon(list1b_cleaned, list2b_cleaned, alternative='less').pvalue,stats.wilcoxon(list1c_cleaned, list2c_cleaned, alternative='less').pvalue]
    for i in range(3):
        if wilconxon_test_p[i]<0.05 and wilconxon_test_p[i]>=0.01:
            plt.text(i*2+1.3, np.max(np.array(data_combined))*1.08,
                     '*', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
            plt.text(i*2+1.3, np.max(np.array(data_combined))*1.17,
                     f'$P$={wilconxon_test_p[i]:.3f}', ha='center', va='bottom', color='k', fontsize=15)
            plt.plot([i*2+1, i*2+1.6], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.1], linewidth=1, color='k')
            plt.plot([i*2+1, i*2+1], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.05], linewidth=1, color='k')
            plt.plot([i*2+1.6, i*2+1.6], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.05], linewidth=1, color='k')
        elif wilconxon_test_p[i]<0.01 and wilconxon_test_p[i]>=0.001:
            plt.text(i*2+1.3, np.max(np.array(data_combined))*1.08,
                     '**', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
            plt.text(i*2+1.3, np.max(np.array(data_combined))*1.17,
                     f'$P$={wilconxon_test_p[i]:.3f}', ha='center', va='bottom', color='k', fontsize=15)
            plt.plot([i*2+1, i*2+1.6], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.1], linewidth=1, color='k')
            plt.plot([i*2+1, i*2+1], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.05], linewidth=1, color='k')
            plt.plot([i*2+1.6, i*2+1.6], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.05], linewidth=1, color='k')
        elif wilconxon_test_p[i]<0.001:
            plt.text(i*2+1.3, np.max(np.array(data_combined))*1.08,
                     '***', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
            # plt.text(i*2+1.3, np.max(np.array(data_combined))*1.17,
            #          f'p = {wilconxon_test_p[i]:.2e}', ha='center', va='bottom', color='k', fontsize=15)
            p_value = wilconxon_test_p[i]
            formatted_p_value = f"{p_value:.2e}"
            base, exponent = formatted_p_value.split("e")
            exponent = int(exponent.lstrip("+"))

            plt.text(i * 2 + 1.3, np.max(np.array(data_combined)) * 1.17, 
                     '$P$='+str(base)+'×10'+rf'$^{{{exponent}}}$', 
                     ha='center', va='bottom', color='k', fontsize=15)
            plt.plot([i*2+1, i*2+1.6], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.1], linewidth=1, color='k')
            plt.plot([i*2+1, i*2+1], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.05], linewidth=1, color='k')
            plt.plot([i*2+1.6, i*2+1.6], [np.max(np.array(data_combined))*1.1, np.max(np.array(data_combined))*1.05], linewidth=1, color='k')
    if scene=='aeb':
        fig_no='a'
    elif scene=='cutin':
        fig_no='b'
    else:
        fig_no='c'
    plt.savefig('Figure8({}).pdf'.format(fig_no), format='pdf', bbox_inches='tight')
    plt.show()

def create_jerk_plot(jerkneg_RL,jerkneg_fNIRS,jerkpos_RL,jerkpos_fNIRS, scene):
    fig, ax = plt.subplots(figsize=(4.5, 4), dpi=300)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    colors = [cmap(0.0625 + 0.25), cmap(0.8125 - 0.125)]*2
    data_combined=[jerkneg_RL,jerkneg_fNIRS,jerkpos_RL,jerkpos_fNIRS]
    positions=[2,3,2,3]
    parts = plt.violinplot(data_combined, positions=positions, showmeans=False, showextrema=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
    for partname in ( 'cbars', 'cmins', 'cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('black')
    vp=parts['cmedians']
    vp.set_edgecolor('red')
    parts['cbars'].set_linewidth(0.8)
    ax.spines['bottom'].set_position(('data', 0))
    plt.grid()
    plt.xticks([2,3], ['', ''])
    plt.xlim((1.3,3.7))
    plt.ylabel('Average Jerk (m/s$^3$)')
    legend_patches = [mpatches.Patch(color=cmap(0.0625 + 0.25), label='w/o fNIRS'),
                      mpatches.Patch(color=cmap(0.8125-0.125), label='w/ fNIRS')]
    min_total=0
    max_total=0
    for i in range(4):
        min_total=min(min_total,min(data_combined[i]))
        max_total=max(max_total,max(data_combined[i]))
    size=[min_total-max_total,max_total-min_total]
    y_pos=[min_total+0.1*(min_total-max_total),max_total+0.1*(max_total-min_total)]

    plt.text(2.5, y_pos[0] + 0.3 * size[0], 'Average Negative Jerk', ha='center', va='center')
    plt.text(2.5, y_pos[1] + 0.3 * size[1], 'Average Positive Jerk', ha='center', va='center')
    for i in range(2):
        data1 = data_combined[i * 2]
        data2 = data_combined[i * 2 + 1]
        if i==0:
            test_side='less'
        else:
            test_side='greater'
        _, p_val = stats.wilcoxon(data1, data2, alternative=test_side)
        if p_val < 0.001:
            # plt.text(2.5, y_pos[i] + 0.05 * size[i]-1.2+offset[i],
            #          '***', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
            # if i == 0:
            #     plt.text(2.5, y_pos[i] + 0.15 * size[i],#-1 + offset[i],
            #          f'***\np = {p_val:.2e}', ha='center', va='center', color='k', fontsize=15)
            # else:
            #     plt.text(2.5, y_pos[i] + 0.15 * size[i],#-1 + offset[i],
            #          f'p = {p_val:.2e}\n***', ha='center', va='center', color='k', fontsize=15)
            formatted_p_value = f"{p_val:.2e}"
            base, exponent = formatted_p_value.split("e")
            exponent = int(exponent.lstrip("+"))

            if i == 0:
                plt.text(2.5, y_pos[i] + 0.15 * size[i],  # -1 + offset[i],
             f'***\n'+'$P$='+str(base)+'×10'+rf'$^{{{exponent}}}$',
             ha='center', va='center', color='k', fontsize=15)
            else:
                plt.text(2.5, y_pos[i] + 0.1 * size[i],  # -1 + offset[i],
             '$P$='+str(base)+'×10'+rf'$^{{{exponent}}}$' + f'\n***',
             ha='center', va='center', color='k', fontsize=15)
            plt.plot([2, 3], [y_pos[i], y_pos[i]], linewidth=1, color='k')
            plt.plot([2, 2], [y_pos[i], y_pos[i] - 0.04 * size[i]], linewidth=1, color='k')
            plt.plot([3, 3], [y_pos[i], y_pos[i] - 0.04 * size[i]], linewidth=1, color='k')
        elif p_val < 0.01:
            # plt.text(2.5, max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2)),
            #          '**', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
            if i == 0:
                plt.text(2.5, y_pos[i] + 0.15 * size[i],#-1 + offset[i],
                     f'**\np = {p_val:.3f}', ha='center', va='center', color='k', fontsize=15)
            else:
                plt.text(2.5, y_pos[i] + 0.15 * size[i],#-1 + offset[i],
                     f'p = {p_val:.3f}\n**', ha='center', va='center', color='k', fontsize=15)
            plt.plot([2, 3], [y_pos, y_pos], linewidth=1, color='k')
            plt.plot([2, 2], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
            plt.plot([3, 3], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
        elif p_val < 0.05:
            # plt.text(1.5, max(max(data1), max(data2)) + 0.1 * max(max(data1) - min(data1), max(data2) - min(data2)),
            #          '*', ha='center', va='bottom', color='k', fontsize=15, weight='bold')
            if i==0:
                plt.text(2.5, y_pos[i] + 0.15 * size[i],#-1 + offset[i],
                     f'*\np = {p_val:.3f}', ha='center', va='center', color='k', fontsize=15)
            else:
                plt.text(2.5, y_pos[i] + 0.15 * size[i],#-1 + offset[i],
                     f'p = {p_val:.3f}\n*', ha='center', va='center', color='k', fontsize=15)
            plt.plot([1, 2], [y_pos, y_pos], linewidth=1, color='k')
            plt.plot([2, 2], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')
            plt.plot([1, 1], [y_pos, y_pos - 0.04 * size], linewidth=1, color='k')

    plt.legend(handles=legend_patches, loc='lower center',bbox_to_anchor=(0.5, -0.4), ncol=2)
    if scene=='aeb':
        fig_no='a'
    elif scene=='cutin':
        fig_no='b'
    else:
        fig_no='c'
    plt.savefig('Figure9({}).pdf'.format(fig_no), format='pdf', bbox_inches='tight')
    plt.show()


for scene in ['aeb', 'cutin', 'ped']:
    risk_RL, risk_fNIRS, ttc_min_RL, ttc_min_fNIRS, jerkpos_RL, jerkpos_fNIRS, jerkneg_RL, jerkneg_fNIRS = process_data(scene)
    create_risk_field_plot(risk_RL, risk_fNIRS,scene)
    create_ttc_plot1(ttc_min_RL,ttc_min_fNIRS,scene)
    # create_violin_plot(ttc_min_RL,ttc_min_fNIRS,'Minimum TTC(s)','less')
    # create_violin_plot(jerkneg_RL,jerkneg_fNIRS,'Average Negative Jerk (m/s$^3$)','less')
    # create_violin_plot(jerkpos_RL,jerkpos_fNIRS,'Average Positive Jerk (m/s$^3$)','greater')
    create_jerk_plot(jerkneg_RL,jerkneg_fNIRS,jerkpos_RL,jerkpos_fNIRS,scene)

