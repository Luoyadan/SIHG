import matplotlib.pyplot as plt
# plt.style.use(['science', 'no-latex'])
import pickle
import numpy as np
from matplotlib import rcParams
rcParams['axes.labelpad'] = 10.0


# with plt.style.context(['science']):

# with open("params_bitcoin_alpha_AUC.pkl", "rb") as f:
#     study = pickle.load(f)
# font = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 18,
#         }
# font_medium = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 16,
#         }
# font_small = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 12,
#         }
# x = []
# y = []
# z = []
#
# for i in range(3):
#     for trial in study.trials:
#         x.append(trial.params['alpha'])
#         y.append(trial.params['gamma'])
#         z.append(trial.value)
# x = np.array(x)
# y = np.array(y)
# z = np.array(z)
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.gca(projection='3d')
# # switch to left side
# # tmp_planes = ax.zaxis._PLANES
# # ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
# #                      tmp_planes[0], tmp_planes[1],
# #                      tmp_planes[4], tmp_planes[5])
# # view_1 = (25, -135)
# # view_2 = (25, -45)
# # init_view = view_2
# # ax.view_init(*init_view)
#
# ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
# ax.set_xlabel(r'$\alpha$', fontdict=font)
# ax.set_ylabel(r'$\gamma$', fontdict=font)
# ax.set_zlabel('AUC', fontdict=font_medium)
#
# best_x = study.best_trial.params['alpha']
# best_y = study.best_trial.params['gamma']
# best_z = study.best_trial.value
# ax.scatter(best_x, best_y, best_z, c="r")
# ax.text(best_x-0.4, best_y-0.4, best_z+0.002, r'$({0:.2f}, {1:.2f}, {2:.3f})$'.format(best_x, best_y, best_z), fontdict=font_small)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# for tick in ax.zaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# ax.autoscale(tight=True)
# # ax.scatter(x, y, z, c=z, cmap='viridis')
#
# plt.show()

# with open("params_bitcoin_alpha_F1.pkl", "rb") as f:
#     study = pickle.load(f)
# font = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 18,
#         }
# font_medium = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 16,
#         }
# font_small = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 12,
#         }
# x = []
# y = []
# z = []
#
# for i in range(3):
#     for trial in study.trials:
#         x.append(trial.params['alpha'])
#         y.append(trial.params['gamma'])
#         z.append(trial.value)
# x = np.array(x)
# y = np.array(y)
# z = np.array(z)
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.gca(projection='3d')
# # switch to left side
# # tmp_planes = ax.zaxis._PLANES
# # ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
# #                      tmp_planes[0], tmp_planes[1],
# #                      tmp_planes[4], tmp_planes[5])
# # view_1 = (25, -135)
# # view_2 = (25, -45)
# # init_view = view_1
# # ax.view_init(*init_view)
#
# ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
# ax.set_xlabel(r'$\alpha$', fontdict=font)
# ax.set_ylabel(r'$\gamma$', fontdict=font)
# ax.set_zlabel('micro-F1', fontdict=font_medium)
#
# best_x = study.best_trial.params['alpha']
# best_y = study.best_trial.params['gamma']
# best_z = study.best_trial.value
# ax.scatter(best_x, best_y, best_z+0.001, c="r", s=10)
# ax.text(best_x-0.4, best_y-0.4, best_z+0.01, r'$({0:.2f}, {1:.2f}, {2:.3f})$'.format(best_x, best_y, best_z), fontdict=font_small)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# for tick in ax.zaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# ax.autoscale(tight=True)
# # ax.scatter(x, y, z, c=z, cmap='viridis')
#
# plt.show()



# with open("params_bitcoin_OTC_F1.pkl", "rb") as f:
#     study = pickle.load(f)
# font = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 18,
#         }
# font_medium = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 16,
#         }
# font_small = {'family': 'sans-serif',
#         'color':  'black',
#         'weight': 'normal',
#         'size': 12,
#         }
# x = []
# y = []
# z = []
#
# for i in range(3):
#     for trial in study.trials:
#         x.append(trial.params['alpha'])
#         y.append(trial.params['gamma'])
#         z.append(trial.value)
# x = np.array(x)
# y = np.array(y)
# z = np.array(z)
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.gca(projection='3d')
# # switch to left side
# # tmp_planes = ax.zaxis._PLANES
# # ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
# #                      tmp_planes[0], tmp_planes[1],
# #                      tmp_planes[4], tmp_planes[5])
# # view_1 = (25, -135)
# # view_2 = (25, -45)
# # init_view = view_1
# # ax.view_init(*init_view)
#
# ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
#
# ax.set_xlabel(r'$\alpha$', fontdict=font)
# ax.set_ylabel(r'$\gamma$', fontdict=font)
# ax.set_zlabel('micro-F1', fontdict=font_medium)
#
# best_x = study.best_trial.params['alpha']
# best_y = study.best_trial.params['gamma']
# best_z = study.best_trial.value
# ax.scatter(best_x, best_y, best_z, c="r", s=20)
# ax.text(best_x-0.4, best_y-0.4, best_z+0.015, r'$({0:.2f}, {1:.2f}, {2:.3f})$'.format(best_x, best_y, best_z), fontdict=font_small)
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# for tick in ax.zaxis.get_major_ticks():
#     tick.label.set_fontsize(13)
#     tick.label.set_fontfamily('sans-serif')
# ax.autoscale(tight=True)
# # ax.scatter(x, y, z, c=z, cmap='viridis')
#
# plt.show()


with open("params_bitcoin_OTC_AUC.pkl", "rb") as f:
    study = pickle.load(f)
font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 18,
        }
font_medium = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
font_small = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
x = []
y = []
z = []

for i in range(3):
    for trial in study.trials:
        x.append(trial.params['alpha'])
        y.append(trial.params['gamma'])
        z.append(trial.value)
x = np.array(x)
y = np.array(y)
z = np.array(z)

fig = plt.figure(figsize=(10, 7))
ax = fig.gca(projection='3d')
# switch to left side
# tmp_planes = ax.zaxis._PLANES
# ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
#                      tmp_planes[0], tmp_planes[1],
#                      tmp_planes[4], tmp_planes[5])
# view_1 = (25, -135)
# view_2 = (25, -45)
# init_view = view_2
# ax.view_init(*init_view)

ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
ax.set_xlabel(r'$\alpha$', fontdict=font)
ax.set_ylabel(r'$\gamma$', fontdict=font)
ax.set_zlabel('AUC', fontdict=font_medium)

best_x = study.best_trial.params['alpha']
best_y = study.best_trial.params['gamma']
best_z = study.best_trial.value
ax.scatter(best_x, best_y, best_z, c="r")
ax.text(best_x-0.4, best_y-0.4, best_z+0.002, r'$({0:.2f}, {1:.2f}, {2:.3f})$'.format(best_x, best_y, best_z), fontdict=font_small)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(13)
    tick.label.set_fontfamily('sans-serif')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(13)
    tick.label.set_fontfamily('sans-serif')
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(13)
    tick.label.set_fontfamily('sans-serif')
ax.autoscale(tight=True)
# ax.scatter(x, y, z, c=z, cmap='viridis')

plt.show()
