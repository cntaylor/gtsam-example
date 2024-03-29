#%%
import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm
from gnss_est_utils import get_chemnitz_data, init_pos
import glob

rr = np.arange(100) # rr = reduced range

in_data = get_chemnitz_data() #('/c/Users/ctaylor/Downloads/Data_Chemnitz.csv')
times = in_data[:,0]
truth = np.array([in_data[i,1] for i in range(len(in_data))]) 
raw_GPS_process = np.array([init_pos(in_data[i,2]) for i in range(len(in_data))])
ecef0 = np.array(init_pos(in_data[0,2])[:3])
lla0 = pm.ecef2geodetic(ecef0[0], ecef0[1], ecef0[2])
ned_truth = np.array([pm.ecef2ned(x[0], x[1], x[2], lla0[0], lla0[1], lla0[2]) for x in truth])
raw_results = np.array([pm.ecef2ned(x[0], x[1], x[2], lla0[0], lla0[1], lla0[2]) for x in raw_GPS_process])

plt.plot(ned_truth[:,1], ned_truth[:,0], label='truth')
plt.scatter(raw_results[rr,1], raw_results[rr,0], color='r', label='raw GPS', s=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('North (m)', fontsize=15)
plt.ylabel('East (m)', fontsize=15)
plt.legend()
plt.title('Raw GPS and True Positions', fontsize=18)
plt.savefig('raw_gps_and_true_positions.pdf')
plt.show()


#%%
RMSEs = []
RMSE2d = np.sqrt(np.average(np.square(ned_truth[:,0:2] - raw_results[:,0:2])))
RMSE3d = np.sqrt(np.average(np.square(ned_truth[:,0:3] - raw_results[:,0:3])))
all_2d_RMSEs = np.sqrt(np.sum(np.square(ned_truth[:,:2] - raw_results[:,:2]),axis=1))
all_3d_RMSEs = np.sqrt(np.sum(np.square(ned_truth[:,:3] - raw_results[:,:3]),axis=1))

RMSEs.append(['raw', RMSE2d, RMSE3d, all_2d_RMSEs, all_3d_RMSEs])




# Get a list of all files that match the pattern
file_list = glob.glob('data_basic_graph_gnss_res*npz')
file_list += glob.glob('data_swichable_constraints_gnss_res*npz')
file_list += ['data_discrete_ind_gnss_res_DI.npz']

file_list = ['DEBUG_data_basic_graph_gnss_res_no_outlier.npz']
# Read in the data and store it in a list of numpy arrays
for f in file_list:
    if 'noBetween' in f:
        continue
    if 'discrete' in f:
        estimator = 'DI'
    elif 'constraints' in f:
        estimator = f.rsplit('.',1)[0].rsplit('_',1)[-1]
    else:
        estimator = f.split('.')[0].split('_')[5]

    # clean up some of the estimator names
    if estimator == 'no':
        estimator = 'Basic'
    if estimator == 'huber':
        estimator = 'Huber'
    if estimator == 'DCS-':
        estimator = 'DCS-.5'
    # if 'DCS' in estimator:
    #     if estimator == 'DCS-2':
    #         estimator = 'DCS'
    #     else:
    #         continue # go to the next estimator
    data = np.load(f)
    results = data['est_states']
    est_results = np.array([pm.ecef2ned(x[0], x[1], x[2], lla0[0], lla0[1], lla0[2]) for x in results])
    print(estimator)
    plt.figure()
    plt.plot(ned_truth[:,1], ned_truth[:,0], label='truth')
    plt.scatter(est_results[:,1], est_results[:,0], color='r', s=2, label='estimated')    
    plt.title(estimator)
    plt.legend()
    plt.savefig(f'res_GNSS_{estimator}.pdf')
    plt.show()
    plt.figure()
    plt.plot(np.sqrt(np.sum(np.square(ned_truth[rr,:2] - est_results[:,:2]),axis=1)))
    plt.show()
    # Let's compute 2D and 3D RMSE and store these values
    RMSE2d = np.sqrt(np.average(np.square(ned_truth[rr,0:2] - est_results[:,0:2])))
    RMSE3d = np.sqrt(np.average(np.square(ned_truth[rr,0:3] - est_results[:,0:3])))
    all_2d_RMSEs = np.sqrt(np.sum(np.square(ned_truth[rr,:2] - est_results[:,:2]),axis=1))
    all_3d_RMSEs = np.sqrt(np.sum(np.square(ned_truth[rr,:3] - est_results[:,:3]),axis=1))
    RMSEs.append([estimator, RMSE2d, RMSE3d, all_2d_RMSEs, all_3d_RMSEs])

#%%
est_desired = np.arange(len(RMSEs))
# est_desired = np.array([0,8,1,4,5,6,7,9,10,14,17])

print('\\hline')
print('Estimator & 2D RMSE (m) & 3D RMSE (m) \\\\')
print('\\hline')
for i in est_desired:
    row = RMSEs[i]
    print('{} & {:.2f} & {:.2f} \\\\'.format(row[0], row[1], row[2]))
    print('\\hline')
# create a violin plot for the 2D RMSE values
violin_data = [RMSEs[i][3] for i in est_desired]
violin_labels = [RMSEs[i][0] for i in est_desired]
fig, ax = plt.subplots()
ax.violinplot(violin_data, showmeans=False, showextrema=True, showmedians=False)
ax.set_xticks(range(1, len(violin_data)+1))
ax.set_xticklabels(violin_labels,fontsize=14)
ax.set_ylim([0,150])
ax.set_ylabel('2D RMSE (m)',fontsize=15)
ax.set_title('2D RMSE per Estimator', fontsize=18)
fig.set_figwidth(8.5)
plt.tight_layout()
plt.savefig('GNSS-2D-Violin.pdf')

# create a violin plot for the 3D RMSE values
violin_data = [RMSEs[i][4] for i in est_desired]
violin_labels = [RMSEs[i][0] for i in est_desired]
fig, ax = plt.subplots()
ax.violinplot(violin_data, showmeans=True, showextrema=False, showmedians=True)
ax.set_xticks(range(1, len(violin_data)+1))
ax.set_xticklabels(violin_labels)
ax.set_ylabel('3D RMSE (m)')
ax.set_title('3D RMSE per Estimator')



plt.figure()
# print(RMSEs)
RMSEs_num = np.array([float(RMSEs[i][1]) for i in est_desired])
RMSEs_name = np.array([RMSEs[i][0] for i in est_desired])
plt.plot(RMSEs_num)
plt.xlabel('Estimator')
plt.xticks(range(len(est_desired)), RMSEs_name)




# %%
file_list = glob.glob('data_new_switchable*npz')
for f in file_list:
    estimator = f.rsplit('.',1)[0].rsplit('_',1)[-1]

    # clean up some of the estimator names
    if estimator == 'no':
        estimator = 'Basic'
    if estimator == 'huber':
        estimator = 'Huber'
    # if 'DCS' in estimator:
    #     if estimator == 'DCS-2':
    #         estimator = 'DCS'
    #     else:
    #         continue # go to the next estimator
    data = np.load(f)
    results = data['est_states']
    est_results = np.array([pm.ecef2ned(x[0], x[1], x[2], lla0[0], lla0[1], lla0[2]) for x in results])
    print(estimator)
    # plt.figure()
    # plt.plot(ned_truth[:,1], ned_truth[:,0], label='truth')
    # plt.plot(est_results[:,1], est_results[:,0], label='estimated')
    # plt.title(estimator)
    # plt.legend()
    # plt.savefig(f'res_GNSS_{estimator}.pdf')
    # plt.show()
    # plt.figure()
    # plt.plot(np.sqrt(np.sum(np.square(ned_truth[:,:2] - est_results[:,:2]),axis=1)))
    # plt.show()
    # Let's compute 2D and 3D RMSE and store these values
    RMSE2d = np.sqrt(np.average(np.square(ned_truth[:,0:2] - est_results[:,0:2])))
    RMSE3d = np.sqrt(np.average(np.square(ned_truth[:,0:3] - est_results[:,0:3])))
    all_2d_RMSEs = np.sqrt(np.sum(np.square(ned_truth[:,:2] - est_results[:,:2]),axis=1))
    all_3d_RMSEs = np.sqrt(np.sum(np.square(ned_truth[:,:3] - est_results[:,:3]),axis=1))
    RMSEs.append([estimator, RMSE2d, RMSE3d, all_2d_RMSEs, all_3d_RMSEs])

# %%
