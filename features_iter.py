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

all_files = [samples_dir +  file + "_test" for file in files]



# Path(traindata_dir).mkdir(parents=True, exist_ok=True)
train_data = all_files
# test_data = all_files[80:100]
sample_counter = 1
X_data = []
X_train = []
y_train_fathom = []
y_train_time = []
y_train_bound_improve = []
X_test = []
y_test = []
x_ub_train_record = []
x_lb_train_record = []
x_ub_test_record = []
x_lb_test_record = []
instance_line_num_train = []
instance_line_num_test = []
#check if we can predict the MADAM relaxation from the basic SDP relaxation
# for a_file in all_files:
#     print(a_file)
import re 
for a_file in all_files:
    line = os.popen("grep edges " + a_file+".output").readlines()[0]
    n_vertices = int(line.split()[2])
    n_edges = int(line.split()[5])
    root = open(a_file + ".traindata0", "r" ).readlines()[-1].strip().split()
    density = float(root[2])
    rootadmm = float(root[3])
    rootbasicSDP = float(root[4])
    rootLB =  float(root[5])
    i = 0 
    lines = open(a_file + ".traindata" + str(i+1), "r" ).readlines()
    for line in lines:
        basicSDP = float(line.split()[1])
        nodeLB = float(line.split()[2])
        nodedim = float(line.split()[0])
        ncuts = int(line.split()[4]) /50
        admmbound = float(line.split("}")[0].split()[-2])
        iter_records = re.findall('\[.*?\]', line)
        for iter_record in iter_records:
            iter_num, bound, time, ntriag, npent, nhept = [float(rec) for rec in iter_record.split("[")[1].split("]")[0].split()]
            new_feature = np.array([n_vertices - nodedim, ncuts, basicSDP - nodeLB, density, iter_num, bound - nodeLB, ntriag, npent, nhept, rootadmm - rootLB, rootbasicSDP - rootLB])
            X_train.append(new_feature)
            y_train_fathom.append(1 if admmbound <= nodeLB + 1 else 0)


scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
# # X_test_scaled = scaler.transform(X_test)
# # feature selection
f_static, p_values = f_classif(X_train_scaled, y_train_fathom)
# # selector = SelectKBest(k=10).fit(X_train_scaled, y_train)
# # X_train_scaled = selector.transform(X_train_scaled)
# # X_test_scaled = selector.transform(X_test_scaled)

# # svc = svm.SVC()

# # #cross validation
# # # param_grid = [
# # #   {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
# # #   {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly'], 'degree':[2, 3, 4]},
# # #   {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'scale', 'auto'], 'kernel': ['rbf']},
# # #  ]
# # # clf = GridSearchCV(svc, param_grid)

clf = svm.SVC(C=1000, gamma=0.001)

clf.fit(X_train_scaled, y_train_fathom)
fitted = clf.predict(X_train_scaled)
n_00 = sum([1 if (y_train_fathom[j] ==0 and fitted[j] == 0) else 0 for j in range(len(y_train_fathom))])
n_01 = sum([1 if (y_train_fathom[j] ==0 and fitted[j] == 1) else 0 for j in range(len(y_train_fathom))])
n_11 = sum([1 if (y_train_fathom[j] ==1 and fitted[j] == 1) else 0 for j in range(len(y_train_fathom))])
n_10 = sum([1 if (y_train_fathom[j] ==1 and fitted[j] == 0) else 0 for j in range(len(y_train_fathom))])
print("the overall stats\n")
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

def svm_predict(x):
    val = clf.intercept_[0]
    gamma = clf.gamma
    for i in range(len(clf.support_vectors_)):
        supp_vec = clf.support_vectors_[i]
        dual_coeff = clf.dual_coef_[0][i]
        val += dual_coeff * np.exp(-gamma * np.dot(x-supp_vec, x-supp_vec))
    return val 

#write decision boundary to file 
f = open("svm_iter.txt", "w")
#write scaler mean and std
f.write(' '.join([str(e) for e in scaler.mean_]) + '\n')
f.write(' '.join([str(np.sqrt(e)) for e in scaler.var_]) + '\n')
#write svm 
f.write(str(clf.intercept_[0]) + "\n")
f.write(str(clf.gamma) + "\n")
f.write(str(len(clf.support_vectors_)) + "\n")
i = 0
for supp_vec in clf.support_vectors_:
    f.write(str(clf.dual_coef_[0][i]) + " " + ' '.join([str(e) for e in supp_vec]) + '\n')
    i += 1

f.close()


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











