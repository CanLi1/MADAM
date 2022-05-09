import numpy as np
from pathlib import Path
from sklearn import svm
from sklearn.metrics import *
from sklearn.feature_selection import *
from sklearn import preprocessing
import os
from sklearn.model_selection import GridSearchCV

# def get_features(n_vertices, n_edges, n, nonzero, fixedvalue, basicSDP, admm, LBvalue, rootadmm, rootbasicSDP, rootLB):
#     # fixed_ratio = n / n_vertices
#     # basicSDP_ratio = basicSDP / LBvalue
#     #
#     depth =
#     return np.array([fixed_ratio, ])

samples_dir = './Instances/rudy/'
# samples_dir = "/local_workspace/canli/maxcut/instance/"
files = ["g05_100.0","g05_100.1","g05_100.2","g05_100.3","g05_100.4","g05_100.5","g05_100.6","g05_100.7","g05_100.8","g05_100.9","g05_60.0","g05_60.1","g05_60.2","g05_60.3","g05_60.4","g05_60.5","g05_60.6","g05_60.7","g05_60.8","g05_60.9","g05_80.0","g05_80.1","g05_80.2","g05_80.3","g05_80.4","g05_80.5","g05_80.6","g05_80.7","g05_80.8","g05_80.9","pm1d_100.0","pm1d_100.1","pm1d_100.2","pm1d_100.3","pm1d_100.4","pm1d_100.5","pm1d_100.6","pm1d_100.7","pm1d_100.8","pm1d_100.9","pm1d_80.0","pm1d_80.1","pm1d_80.2","pm1d_80.3","pm1d_80.4","pm1d_80.5","pm1d_80.6","pm1d_80.7","pm1d_80.8","pm1d_80.9","pm1s_100.0","pm1s_100.1","pm1s_100.2","pm1s_100.3","pm1s_100.4","pm1s_100.5","pm1s_100.6","pm1s_100.7","pm1s_100.8","pm1s_100.9","pm1s_80.0","pm1s_80.1","pm1s_80.2","pm1s_80.3","pm1s_80.4","pm1s_80.5","pm1s_80.6","pm1s_80.7","pm1s_80.8","pm1s_80.9","pw01_100.0","pw01_100.1","pw01_100.2","pw01_100.3","pw01_100.4","pw01_100.5","pw01_100.6","pw01_100.7","pw01_100.8","pw01_100.9","pw05_100.0","pw05_100.1","pw05_100.2","pw05_100.3","pw05_100.4","pw05_100.5","pw05_100.6","pw05_100.7","pw05_100.8","pw05_100.9","pw09_100.0","pw09_100.1","pw09_100.2","pw09_100.3","pw09_100.4","pw09_100.5","pw09_100.6","pw09_100.7","pw09_100.8","pw09_100.9","w01_100.0","w01_100.1","w01_100.2","w01_100.3","w01_100.4","w01_100.5","w01_100.6","w01_100.7","w01_100.8","w01_100.9","w05_100.0","w05_100.1","w05_100.2","w05_100.3","w05_100.4","w05_100.5","w05_100.6","w05_100.7","w05_100.8","w05_100.9","w09_100.0","w09_100.1","w09_100.2","w09_100.3","w09_100.4","w09_100.5","w09_100.6","w09_100.7","w09_100.8","w09_100.9"]
# files = ["unweighted_100_01_1","unweighted_100_01_2","unweighted_100_01_3","unweighted_100_01_4","unweighted_100_01_5","unweighted_100_02_1","unweighted_100_02_2","unweighted_100_02_3","unweighted_100_02_4","unweighted_100_02_5","unweighted_100_03_1","unweighted_100_03_2","unweighted_100_03_3","unweighted_100_03_4","unweighted_100_03_5","unweighted_100_04_1","unweighted_100_04_2","unweighted_100_04_3","unweighted_100_04_4","unweighted_100_04_5","unweighted_100_05_1","unweighted_100_05_2","unweighted_100_05_3","unweighted_100_05_4","unweighted_100_05_5","unweighted_100_06_1","unweighted_100_06_2","unweighted_100_06_3","unweighted_100_06_4","unweighted_100_06_5","unweighted_100_07_1","unweighted_100_07_2","unweighted_100_07_3","unweighted_100_07_4","unweighted_100_07_5","unweighted_100_08_1","unweighted_100_08_2","unweighted_100_08_3","unweighted_100_08_4","unweighted_100_08_5","unweighted_100_09_1","unweighted_100_09_2","unweighted_100_09_3","unweighted_100_09_4","unweighted_100_09_5"]

all_files = [samples_dir +  file for file in files]



# Path(traindata_dir).mkdir(parents=True, exist_ok=True)
train_data = all_files
# test_data = all_files[80:100]
sample_counter = 1
X_train = []
y_train = []
X_test = []
y_test = []
x_ub_train_record = []
x_lb_train_record = []
x_ub_test_record = []
x_lb_test_record = []
instance_line_num_train = []
instance_line_num_test = []
#check if we can predict the MADAM relaxation from the basic SDP relaxation
for a_file in all_files:
    print(a_file)
    line = os.popen("grep edges " + a_file+".output").readlines()[0]
    n_vertices = int(line.split()[2])
    n_edges = int(line.split()[5])
    root = open(a_file + ".traindata0", "r" ).readlines()[0]
    rootadmm = float(root.strip().split("[")[-1].split()[1])
    rootbasicSDP = float(root.strip().split("[")[-1].split()[1])
    rootLB =  float(root.strip().split("[")[-1].split()[2])
    if len(root.strip().split("[")) >= 5:
        roottriag =  float(root.strip().split("[")[4].split()[1])
    else:
        roottriag = rootadmm
    for i in range(10):
        lines = open(a_file + ".traindata" + str(i), "r" ).readlines()
        for line in lines:
            #number of nonfixed variables
            n = int(line.strip().split("[")[0].split()[0])
            #nonzeros in the L matrix
            nonzero = int(line.strip().split("[")[0].split()[1])
            #fixed value
            fixedvalue = float(line.strip().split("[")[0].split()[2])
            #basic SDP relaxation value
            basicSDP = float(line.strip().split("[")[1].split()[1])
            #admm relaxation value
            admm =  float(line.strip().split("[")[-1].split()[1])
            #relaxation with only triag inequality
            if len(line.strip().split("[")) >= 5:
                triag =  float(line.strip().split("[")[4].split()[1])
            else:
                triag = admm
            #LB value
            LBvalue = float(line.strip().split("[")[-1].split()[2])
            new_feature = np.array([rootbasicSDP - LBvalue, basicSDP - LBvalue, roottriag - LBvalue, triag - LBvalue, n_vertices - n])
            X_train.append(new_feature)
            y_train.append(1 if admm< LBvalue + 1.0  else 0)





scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#feature selection
# f_static, p_values = f_classif(X_train_scaled, y_train)
# selector = SelectKBest(k=10).fit(X_train_scaled, y_train)
# X_train_scaled = selector.transform(X_train_scaled)
# X_test_scaled = selector.transform(X_test_scaled)

svc = svm.SVC()

#cross validation
# param_grid = [
#   {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
#   {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly'], 'degree':[2, 3, 4]},
#   {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'scale', 'auto'], 'kernel': ['rbf']},
#  ]
# clf = GridSearchCV(svc, param_grid)

clf = svm.SVC(C=1000, gamma=0.001)

clf.fit(X_train_scaled, y_train)
fitted = clf.predict(X_train_scaled)
print("training accuracy ", accuracy_score(y_train, fitted)  )
#
# predicted = clf.predict(X_test_scaled)
# print("test accuracy ", accuracy_score(y_test, predicted))



#examine predicted results true label/ pred label
n_00 = 0
n_01 = 0
n_11 = 0
n_10 = 0
n_1_to_0 = 0
graphs = []
for i in range(len(y_train)):
    if y_train[i] == 0 and fitted[i] == 0:
        n_00 += 1
    elif y_train[i] == 0 and fitted[i] == 1:
        n_01 += 1
    elif y_train[i] == 1 and fitted[i] == 1:
        n_11 += 1
    else:
        n_10 += 1
# for i in range(len(y_test)):
#     if y_test[i] == 0 and predicted[i] == 0:
#         n_00 += 1
#     elif y_test[i] == 0 and predicted[i] == 1:
#         n_01 += 1
#     elif y_test[i] == 1 and predicted[i] == 1:
#         n_11 += 1
#     else:
#         n_10 += 1



print("accuracy ", "{0:.0%}".format((n_00 + n_11)/ (n_00 + n_01 + n_11 + n_10)))
print("precision ", "{0:.0%}".format((n_11)/ (n_01 + n_11 )))
print("recall ", "{0:.0%}".format((n_11)/ (n_11 + n_10)))
print("1 to 0 error", "{0:.2%}".format(n_1_to_0/ (n_00 + n_01 + n_11 + n_10)))









