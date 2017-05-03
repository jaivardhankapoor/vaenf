import pickle
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
markers = ['v','^','d','_','|','s','8','s','p','*']


filname = 'flow_samples_all.txt'
with open(filname,'r') as f:
    data = pickle.loads(f.read())
x = []
y = []
# print(len(data))
for j in range(len(data)-10000,len(data)):
    # print(j)
    x.append(data[j][0][0])
    y.append(data[j][0][1])
    # print(j)
# print(data)
# len(y)
xy =np.vstack([x,y])
z = (gaussian_kde(xy)(xy))
# x_m = sum(x) / float(len(x))
# y_m = sum(y) / float(len(x))
ax.scatter(x, y, c=z, s=10, edgecolor='')

print("main_done")
for i in range(0,10):
    filname = 'flow_samples_' + str(i) + '.txt'
    with open(filname,'r') as f:
        data = pickle.loads(f.read())
    x = []
    y = []
    # print(len(data))
    for j in range(len(data)-1000,len(data)):
        x.append(data[j][0][0])
        y.append(data[j][0][1])
    # print(data)
    # len(y)
    # xy =np.vstack([x,y])
    # z = (gaussian_kde(xy)(xy))*(i+.1)/10.0
    x_m = sum(x) / float(len(x))
    y_m = sum(y) / float(len(x))
    ax.scatter(x_m, y_m, c=1000 ,s=100, edgecolor='',marker = markers[i])
    print(i)
plt.savefig('plotasa' + '.png')
# + str(i)+ 
# import pickle
# import numpy as np
# from scipy.stats import gaussian_kde
# import matplotlib.pyplot as plt

# with open('flow_samples_1.txt','r') as f:
# 		data = pickle.loads(f.read())
# x = []
# y = []
# print(len(data))
# for i in range(0,len(data)):
# 	x.append(data[i][0][0])
# 	y.append(data[i][0][1])
# # print(data)
# # len(y)
# xy =np.vstack([x,y])
# z = gaussian_kde(xy)(xy)
# fig, ax = plt.subplots()
# ax.scatter(x, y, c=z, s=100, edgecolor='')
# plt.savefig('as')

# # import pickle
# # with open('samples.txt','r') as f:
# #     data = pickle.load(f)


# # import numpy as np
# # from scipy.stats import gaussian_kde

# # import matplotlib.pyplot as plt
# # for j in range(0,9):
# #     data = (distribution[j])[-5000:]
# # x= []
# # y= []
# # len(data)
# # x.append(data[i][0][0])
# # y.append(data[i][0][1])

# # len(x)
# # len(y)
# # # Calculate the point density
# # xy = np.vstack([x,y])
# # z = gaussian_kde(xy)(xy)

# # fig, ax = plt.subplots()
# # ax.scatter(x, y, c=z, s=100, edgecolor='')
# # plt.savefig()
# #     savefig('foo' + str(j)+ '.pdf')