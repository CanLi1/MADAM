import numpy as np
from pathlib import Path
from sklearn import svm
from sklearn.metrics import *
from sklearn.feature_selection import *
from sklearn import preprocessing
import os
from sklearn.model_selection import GridSearchCV
import re 
# def get_features(n_vertices, n_edges, n, nonzero, fixedvalue, basicSDP, admm, LBvalue, rootadmm, rootbasicSDP, rootLB):
#     # fixed_ratio = n / n_vertices
#     # basicSDP_ratio = basicSDP / LBvalue
#     #
#     depth =
#     return np.array([fixed_ratio, ])

samples_dir = './Instances/rudy/'
# samples_dir = "/local_workspace/canli/maxcut/instance/"
# files = ["g05_100.0","g05_100.1","g05_100.2","g05_100.3","g05_100.4","g05_100.5","g05_100.6","g05_100.7","g05_100.8","g05_100.9","g05_60.0","g05_60.1","g05_60.2","g05_60.3","g05_60.4","g05_60.5","g05_60.6","g05_60.7","g05_60.8","g05_60.9","g05_80.0","g05_80.1","g05_80.2","g05_80.3","g05_80.4","g05_80.5","g05_80.6","g05_80.7","g05_80.8","g05_80.9","pm1d_100.0","pm1d_100.1","pm1d_100.2","pm1d_100.3","pm1d_100.4","pm1d_100.5","pm1d_100.6","pm1d_100.7","pm1d_100.8","pm1d_100.9","pm1d_80.0","pm1d_80.1","pm1d_80.2","pm1d_80.3","pm1d_80.4","pm1d_80.5","pm1d_80.6","pm1d_80.7","pm1d_80.8","pm1d_80.9","pm1s_100.0","pm1s_100.1","pm1s_100.2","pm1s_100.3","pm1s_100.4","pm1s_100.5","pm1s_100.6","pm1s_100.7","pm1s_100.8","pm1s_100.9","pm1s_80.0","pm1s_80.1","pm1s_80.2","pm1s_80.3","pm1s_80.4","pm1s_80.5","pm1s_80.6","pm1s_80.7","pm1s_80.8","pm1s_80.9","pw01_100.0","pw01_100.1","pw01_100.2","pw01_100.3","pw01_100.4","pw01_100.5","pw01_100.6","pw01_100.7","pw01_100.8","pw01_100.9","pw05_100.0","pw05_100.1","pw05_100.2","pw05_100.3","pw05_100.4","pw05_100.5","pw05_100.6","pw05_100.7","pw05_100.8","pw05_100.9","pw09_100.0","pw09_100.1","pw09_100.2","pw09_100.3","pw09_100.4","pw09_100.5","pw09_100.6","pw09_100.7","pw09_100.8","pw09_100.9","w01_100.0","w01_100.1","w01_100.2","w01_100.3","w01_100.4","w01_100.5","w01_100.6","w01_100.7","w01_100.8","w01_100.9","w05_100.0","w05_100.1","w05_100.2","w05_100.3","w05_100.4","w05_100.5","w05_100.6","w05_100.7","w05_100.8","w05_100.9","w09_100.0","w09_100.1","w09_100.2","w09_100.3","w09_100.4","w09_100.5","w09_100.6","w09_100.7","w09_100.8","w09_100.9"]
files = ["g05_100.0","g05_100.1","g05_100.2","g05_100.3","g05_100.4","g05_100.5","g05_100.6","g05_100.7","g05_100.8","g05_100.9","g05_60.0","pm1d_100.0","pm1d_100.1","pm1d_100.2","pm1d_100.3","pm1d_100.4","pm1d_100.5","pm1d_100.6","pm1d_100.7","pm1d_100.8","pm1d_100.9","pm1s_100.0","pm1s_100.1","pm1s_100.2","pm1s_100.3","pm1s_100.4","pm1s_100.5","pm1s_100.6","pm1s_100.7","pm1s_100.8","pm1s_100.9","pw01_100.0","pw01_100.1","pw01_100.2","pw01_100.3","pw01_100.4","pw01_100.5","pw01_100.6","pw01_100.7","pw01_100.8","pw01_100.9","pw05_100.0","pw05_100.1","pw05_100.2","pw05_100.3","pw05_100.4","pw05_100.5","pw05_100.6","pw05_100.7","pw05_100.8","pw05_100.9","pw09_100.0","pw09_100.1","pw09_100.2","pw09_100.3","pw09_100.4","pw09_100.5","pw09_100.6","pw09_100.7","pw09_100.8","pw09_100.9","w01_100.0","w01_100.1","w01_100.2","w01_100.3","w01_100.4","w01_100.5","w01_100.6","w01_100.7","w01_100.8","w01_100.9","w05_100.0","w05_100.1","w05_100.2","w05_100.3","w05_100.4","w05_100.5","w05_100.6","w05_100.7","w05_100.8","w05_100.9","w09_100.0","w09_100.1","w09_100.2","w09_100.3","w09_100.4","w09_100.5","w09_100.6","w09_100.7","w09_100.8","w09_100.9"]
# files = ["unweighted_100_01_1","unweighted_100_01_2","unweighted_100_01_3","unweighted_100_01_4","unweighted_100_01_5","unweighted_100_02_1","unweighted_100_02_2","unweighted_100_02_3","unweighted_100_02_4","unweighted_100_02_5","unweighted_100_03_1","unweighted_100_03_2","unweighted_100_03_3","unweighted_100_03_4","unweighted_100_03_5","unweighted_100_04_1","unweighted_100_04_2","unweighted_100_04_3","unweighted_100_04_4","unweighted_100_04_5","unweighted_100_05_1","unweighted_100_05_2","unweighted_100_05_3","unweighted_100_05_4","unweighted_100_05_5","unweighted_100_06_1","unweighted_100_06_2","unweighted_100_06_3","unweighted_100_06_4","unweighted_100_06_5","unweighted_100_07_1","unweighted_100_07_2","unweighted_100_07_3","unweighted_100_07_4","unweighted_100_07_5","unweighted_100_08_1","unweighted_100_08_2","unweighted_100_08_3","unweighted_100_08_4","unweighted_100_08_5","unweighted_100_09_1","unweighted_100_09_2","unweighted_100_09_3","unweighted_100_09_4","unweighted_100_09_5"]

all_files = [samples_dir +  file + "_test" for file in files]



# Path(traindata_dir).mkdir(parents=True, exist_ok=True)
train_data = [file for file in all_files if bool(re.search("\.[0-6]", file))]
test_data = [file for file in all_files if bool(re.search("\.[7-9]", file))]
sample_counter = 1
X_data = []
X_train = []
y_train_fathom = []
y_train_time = []
y_train_bound_improve = []
X_test = []
y_test = []
y_test_fathom = []
x_ub_train_record = []
x_lb_train_record = []
x_ub_test_record = []
x_lb_test_record = []
instance_line_num_train = []
instance_line_num_test = []
#check if we can predict the MADAM relaxation from the basic SDP relaxation
# for a_file in all_files:
#     print(a_file)
for a_file in all_files:
    line = os.popen("grep edges " + a_file+".output").readlines()[0]
    n_vertices = int(line.split()[2])
    n_edges = int(line.split()[5])
    root = open(a_file + ".traindata0", "r" ).readlines()[-1].strip().split()
    density = float(root[2])
    rootadmm = float(root[3])
    rootbasicSDP = float(root[4])
    rootLB =  float(root[5])

    for i in range(9):
        lines = open(a_file + ".traindata" + str(i+1), "r" ).readlines()
        for line in lines:
            basicSDP = float(line.split()[1])
            nodeLB = float(line.split()[2])
            ncuts = int(line.split()[4]) /50
            time = float(line.split("}")[0].split()[-1])
            admmbound = float(line.split("}")[0].split()[-2])
            num_iter = int(line.split("{")[1].split()[0])
            nodedim = float(line.split()[0])
            new_data = np.array([n_vertices - nodedim, ncuts, time, basicSDP, nodeLB, admmbound, num_iter, density, rootadmm, rootbasicSDP, rootLB])
            X_data.append(new_data)

            new_feature = np.array([n_vertices - nodedim, ncuts, basicSDP - nodeLB, density, rootadmm - nodeLB, rootbasicSDP - nodeLB])
            if a_file in train_data:
                X_train.append(new_feature)
                y_train_fathom.append(1 if admmbound <= nodeLB + 1 else 0)
            else:
                X_test.append(new_feature)
                y_test_fathom.append(1 if admmbound <= nodeLB + 1 else 0)

            


plot_ncuts = [x[1] for x in X_data if x[6]>=6]
plot_time = [x[2] for x in X_data if x[6]>=6]


# for i in range(len(X_data)):
#     depth = X_data[i][0]
#     ncuts = X_data[i][1]
#     density = X_data[i][7]
#     root_admm_improve = -(X_data[i][8] - X_data[i][9]) / X_data[i][9]
#     time = X_data[i][2]
#     admm_improve_percent = (X_data[i][3] - X_data[i][5])/X_data[i][3] 
#     boundimprove_percent = (X_data[int(i/6)*6][5] - X_data[i][5])/X_data[int(i/6)*6][5]
#     basicSDP_root_diff = -(X_data[i][3] - X_data[i][9]) / X_data[i][9]
#     new_feature = [depth, ncuts, density, root_admm_improve, basicSDP_root_diff]
#     X_train.append(new_feature)
#     y_train_time.append(time)
#     y_train_bound_improve.append(admm_improve_percent)

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# # X_test_scaled = scaler.transform(X_test)
# f_static1, p_values1 = f_classif(X_train_scaled, y_train_time)

# f_static2, p_values2 = f_classif(X_train_scaled, y_train_bound_improve)


# # #==========regression ===============
# # regr1 = svm.SVR(epsilon=0.5, C=10000, gamma=0.001)
# regr1 = svm.LinearSVR()
# regr1.fit(X_train_scaled, y_train_time)
# fitted1 = regr1.predict(X_train_scaled)

# regr2 = svm.SVR(epsilon=0.5, C=10000, gamma=0.001)
# regr2.fit(X_train_scaled, y_train_bound_improve)
# fitted2 = regr2.predict(X_train_scaled)


# predicted = regr.predict(X_test_scaled)


with open('plot.txt', 'w') as fp:
    for i in range(len(plot_ncuts)):
        fp.write(str(plot_ncuts[i]) + " " + str(plot_time[i]) + "\n")
# #get some statistics of these data 
# time_mean = []
# time_std = []
# boundimprovement_mean = []
# boundimprovement_std = []

# for i in range(6):
#     time_mean.append(np.mean([plot_time[j] for j in range(len(plot_time)) if plot_ncuts[j] == i]))
#     time_std.append(np.std([plot_time[j] for j in range(len(plot_time)) if plot_ncuts[j] == i]))
#     boundimprovement_mean.append(np.mean([plot_boundimprove_percent[j] for j in range(len(plot_boundimprove_percent)) if plot_ncuts[j] == i]))
#     boundimprovement_std.append(np.std([plot_boundimprove_percent[j] for j in range(len(plot_boundimprove_percent)) if plot_ncuts[j] == i]))
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# feature selection
f_static, p_values = f_classif(X_train_scaled, y_train_fathom)
# selector = SelectKBest(k=10).fit(X_train_scaled, y_train)
# X_train_scaled = selector.transform(X_train_scaled)
# X_test_scaled = selector.transform(X_test_scaled)

# svc = svm.SVC()

# #cross validation
# # param_grid = [
# #   {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
# #   {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly'], 'degree':[2, 3, 4]},
# #   {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'scale', 'auto'], 'kernel': ['rbf']},
# #  ]
# # clf = GridSearchCV(svc, param_grid)

clf = svm.SVC(C=1000, gamma=0.001)

clf.fit(X_train_scaled, y_train_fathom)
fitted = clf.predict(X_train_scaled)
n_00 = sum([1 if (y_train_fathom[j] ==0 and fitted[j] == 0) else 0 for j in range(len(y_train_fathom))])
n_01 = sum([1 if (y_train_fathom[j] ==0 and fitted[j] == 1) else 0 for j in range(len(y_train_fathom))])
n_11 = sum([1 if (y_train_fathom[j] ==1 and fitted[j] == 1) else 0 for j in range(len(y_train_fathom))])
n_10 = sum([1 if (y_train_fathom[j] ==1 and fitted[j] == 0) else 0 for j in range(len(y_train_fathom))])
print("the overall training stats\n")
print("accuracy ", "{0:.0%}".format((n_00 + n_11)/ (n_00 + n_01 + n_11 + n_10)))
print("precision ", "{0:.0%}".format((n_11)/ (n_01 + n_11 )))
print("recall ", "{0:.0%}".format((n_11)/ (n_11 + n_10)))
print("n00 ", n_00, " n01 ", n_01, " n_11 ", n_11, " n_10 ", n_10, "\n")
for i in range(6):
    n_00 = sum([1 if (y_train_fathom[j] ==0 and fitted[j] == 0 and j % 6 == i) else 0 for j in range(len(y_train_fathom))])
    n_01 = sum([1 if (y_train_fathom[j] ==0 and fitted[j] == 1 and j % 6 == i) else 0 for j in range(len(y_train_fathom))])
    n_11 = sum([1 if (y_train_fathom[j] ==1 and fitted[j] == 1 and j % 6 == i) else 0 for j in range(len(y_train_fathom))])
    n_10 = sum([1 if (y_train_fathom[j] ==1 and fitted[j] == 0 and j % 6 == i) else 0 for j in range(len(y_train_fathom))])
    print("number of cuts = ", i, "\n ")
    print("accuracy ", "{0:.0%}".format((n_00 + n_11)/ (n_00 + n_01 + n_11 + n_10)))
    print("precision ", "{0:.0%}".format((n_11)/ (n_01 + n_11 )))
    print("recall ", "{0:.0%}".format((n_11)/ (n_11 + n_10)))
    print("n00 ", n_00, " n01 ", n_01, " n_11 ", n_11, " n_10 ", n_10, "\n")

predicted = clf.predict(X_test_scaled)
n_00 = sum([1 if (y_test_fathom[j] ==0 and predicted[j] == 0) else 0 for j in range(len(y_test_fathom))])
n_01 = sum([1 if (y_test_fathom[j] ==0 and predicted[j] == 1) else 0 for j in range(len(y_test_fathom))])
n_11 = sum([1 if (y_test_fathom[j] ==1 and predicted[j] == 1) else 0 for j in range(len(y_test_fathom))])
n_10 = sum([1 if (y_test_fathom[j] ==1 and predicted[j] == 0) else 0 for j in range(len(y_test_fathom))])
print("the overall test stats\n")
print("test accuracy ", "{0:.0%}".format((n_00 + n_11)/ (n_00 + n_01 + n_11 + n_10)))
print("test precision ", "{0:.0%}".format((n_11)/ (n_01 + n_11 )))
print("test recall ", "{0:.0%}".format((n_11)/ (n_11 + n_10)))
print("n00 ", n_00, " n01 ", n_01, " n_11 ", n_11, " n_10 ", n_10, "\n")
for i in range(6):
    n_00 = sum([1 if (y_test_fathom[j] ==0 and predicted[j] == 0 and j % 6 == i) else 0 for j in range(len(y_test_fathom))])
    n_01 = sum([1 if (y_test_fathom[j] ==0 and predicted[j] == 1 and j % 6 == i) else 0 for j in range(len(y_test_fathom))])
    n_11 = sum([1 if (y_test_fathom[j] ==1 and predicted[j] == 1 and j % 6 == i) else 0 for j in range(len(y_test_fathom))])
    n_10 = sum([1 if (y_test_fathom[j] ==1 and predicted[j] == 0 and j % 6 == i) else 0 for j in range(len(y_test_fathom))])
    print("number of cuts = ", i, "\n ")
    print("test accuracy ", "{0:.0%}".format((n_00 + n_11)/ (n_00 + n_01 + n_11 + n_10)))
    print("test precision ", "{0:.0%}".format((n_11)/ (n_01 + n_11 )))
    print("test recall ", "{0:.0%}".format((n_11)/ (n_11 + n_10)))
    print("n00 ", n_00, " n01 ", n_01, " n_11 ", n_11, " n_10 ", n_10, "\n")

# #define 6 different classifiers and train 
# for i in range(6):
#     X_train_i = []
#     y_train_fathom_i = []
#     for j in range(len(X_train)):
#         if j % 6 == i:
#             X_train_i.append(np.array([X_train[j][0], X_train[j][2], X_train[j][3], X_train[j][4], X_train[j][5]]))
#             y_train_fathom_i.append(y_train_fathom[j])
#     scaler = preprocessing.StandardScaler().fit(X_train_i)
#     X_train_scaled = scaler.transform(X_train_i)
#     f_static, p_values = f_classif(X_train_scaled, y_train_fathom_i)    
#     clf = svm.SVC(C=1000, gamma=0.001)
#     clf.fit(X_train_scaled, y_train_fathom_i)
#     fitted = clf.predict(X_train_scaled)     
#     n_00 = sum([1 if (y_train_fathom_i[j] ==0 and fitted[j] == 0) else 0 for j in range(len(y_train_fathom_i))])
#     n_01 = sum([1 if (y_train_fathom_i[j] ==0 and fitted[j] == 1 ) else 0 for j in range(len(y_train_fathom_i))])
#     n_11 = sum([1 if (y_train_fathom_i[j] ==1 and fitted[j] == 1 ) else 0 for j in range(len(y_train_fathom_i))])
#     n_10 = sum([1 if (y_train_fathom_i[j] ==1 and fitted[j] == 0 ) else 0 for j in range(len(y_train_fathom_i))])
#     print("number of cuts = ", i, "\n ")
#     print("accuracy ", "{0:.0%}".format((n_00 + n_11)/ (n_00 + n_01 + n_11 + n_10)))
#     if n_01 + n_11 != 0:
#         print("precision ", "{0:.0%}".format((n_11)/ (n_01 + n_11 )))
#     print("recall ", "{0:.0%}".format((n_11)/ (n_11 + n_10)))    
#     print("n00 ", n_00, " n01 ", n_01, " n_11 ", n_11, " n_10 ", n_10, "\n")

#write decision boundary to file 
f = open("svm.txt", "w")
#write scaler mean and std
f.write(' '.join([str(e) for e in scaler.mean_]) + '\n')
f.write(' '.join([str(np.sqrt(e)) for e in scaler.var_]) + '\n')
#write svm 
f.write(str(clf.intercept_[0]) + "\n")
f.write(str(clf.gamma) + "\n")
f.write(str(len(clf.support_vectors_)) + "\n")
for j in range(len(clf.support_vectors_)):
    f.write(str(clf.dual_coef_[0][j]) + " " + ' '.join([str(e) for e in clf.support_vectors_[j]]) + '\n')

f.close()
#
# # predicted = clf.predict(X_test_scaled)
# # print("test accuracy ", accuracy_score(y_test, predicted))



# #examine predicted results true label/ pred label
# n_00 = 0
# n_01 = 0
# n_11 = 0
# n_10 = 0
# n_1_to_0 = 0
# graphs = []
# for i in range(len(y_train)):
#     if y_train[i] == 0 and fitted[i] == 0:
#         n_00 += 1
#     elif y_train[i] == 0 and fitted[i] == 1:
#         n_01 += 1
#     elif y_train[i] == 1 and fitted[i] == 1:
#         n_11 += 1
#     else:
#         n_10 += 1
# # for i in range(len(y_test)):
# #     if y_test[i] == 0 and predicted[i] == 0:
# #         n_00 += 1
# #     elif y_test[i] == 0 and predicted[i] == 1:
# #         n_01 += 1
# #     elif y_test[i] == 1 and predicted[i] == 1:
# #         n_11 += 1
# #     else:
# #         n_10 += 1



# print("accuracy ", "{0:.0%}".format((n_00 + n_11)/ (n_00 + n_01 + n_11 + n_10)))
# print("precision ", "{0:.0%}".format((n_11)/ (n_01 + n_11 )))
# print("recall ", "{0:.0%}".format((n_11)/ (n_11 + n_10)))
# print("1 to 0 error", "{0:.2%}".format(n_1_to_0/ (n_00 + n_01 + n_11 + n_10)))









