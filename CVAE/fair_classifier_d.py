import numpy as np
import os

import statsmodels.api as sm
import torch
import torch.utils.data as utils

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error

from scipy.optimize import minimize

import pandas as pd

def custom_objective(params, X1, X2, y, alpha=1.0):
    w = params[:-1]
    b = params[-1]
    y1_pred = np.dot(X1, w) + b
    y2_pred = np.dot(X2, w) + b
    mse = mean_squared_error(y, y1_pred)
    reg_term = alpha * np.linalg.norm(y1_pred - y2_pred)
    objective = mse + reg_term
    return objective

class Reg_linearRegression():
    def __init__(self, size, alpha):
        self.w = np.zeros(size)
        self.b = np.zeros(1)
        self.alpha = alpha
    
    def fit(self, X1, X2, y):
        init_params = np.concatenate([self.w, self.b], axis=-1)
        results = minimize(custom_objective, init_params, args=(X1, X2, y, self.alpha), method="L-BFGS-B")
        self.w = results.x[:-1]
        self.b = results.x[-1]
    
    def predict(self, X):
        return np.dot(X, self.w) + self.b
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def custom_regularization(weights, X1, X2, y, alpha):
    y1_pred = sigmoid(X1 @ weights)[:, np.newaxis]
    y2_pred = sigmoid(X2 @ weights)[:, np.newaxis]
    log_loss = -np.mean(y * np.log(y1_pred + 1e-10) + (1 - y) * np.log(1 - y1_pred + 1e-10))
    reg_term = alpha * np.linalg.norm(y1_pred - y2_pred)
    objective = log_loss + reg_term
    return objective

class Reg_logisticRegression():
    def __init__(self, size, alpha):
        self.weights = np.random.randn(size + 1)
        self.alpha = alpha
    
    def fit(self, X1, X2, y):
        X1 = np.hstack((X1, np.ones((X1.shape[0], 1))))
        X2 = np.hstack((X2, np.ones((X2.shape[0], 1))))
        results = minimize(custom_regularization, self.weights, args=(X1, X2, y, self.alpha), method="BFGS", options={'maxiter': 500})
        self.weights = results.x
    
    def predict(self, X):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return (sigmoid(X @ self.weights) > 0.5).astype(int)

class SM_LinearRegression():
    def __init__(self):
        pass
        
    def fit(self, X, y):
        N = X.shape[0]
        self.LRFit = sm.OLS(y, np.hstack([X,np.ones(N).reshape(-1,1)]),hasconst=True).fit()
        
    def predict(self,X):
        N = X.shape[0]
        return self.LRFit.predict(np.hstack([X,np.ones(N).reshape(-1,1)]))

def cf_eval(r, y, y_cf, a, args):
    a = a.squeeze()
    mask1 = (a == 0)
    mask2 = (a == 1)
    
    cf_effect = np.abs(y_cf - y)
    o1 = cf_effect[mask1 == [True]]
    o2 = cf_effect[mask2 == [True]]
    print("*" * 20)
    print("causal effect of the classifier")
    print("cf = {:.4f}".format(np.sum(cf_effect) / cf_effect.shape[0]))
    print("o1 = {:.4f}".format(np.sum(o1) / o1.shape[0]))
    print("o2 = {:.4f}".format(np.sum(o2) / o2.shape[0]))
    """if args.dataset == "law":
        m = np.argmax(r, axis=1)
        print("m = {}".format(m))
        a = a.squeeze(1)

        mask1 = (m == 0)
        mask2 = (m == 1)
        mask3 = (m == 2)
        mask4 = (m == 3)
        mask5 = (m == 4)
        mask6 = (m == 5)
        mask7 = (m == 6)
        mask8 = (m == 7)

        cf_effect = np.abs(y_cf - y)
        
        o1 = cf_effect[mask1 == [True]]
        o2 = cf_effect[mask2 == [True]]
        o3 = cf_effect[mask3 == [True]]
        o4 = cf_effect[mask4 == [True]]
        o5 = cf_effect[mask5 == [True]]
        o6 = cf_effect[mask6 == [True]]
        o7 = cf_effect[mask7 == [True]]
        o8 = cf_effect[mask8 == [True]]

        print("*" * 20)
        print("causal effect of the classifier")
        print("cf = {:.4f}".format(np.sum(cf_effect) / cf_effect.shape[0]))
        print("o1 = {:.4f}".format(np.sum(o1) / o1.shape[0]))
        print("o2 = {:.4f}".format(np.sum(o2) / o2.shape[0]))
        print("o3 = {:.4f}".format(np.sum(o3) / o3.shape[0]))
        print("o4 = {:.4f}".format(np.sum(o4) / o4.shape[0]))
        print("o5 = {:.4f}".format(np.sum(o5) / o5.shape[0]))
        print("o6 = {:.4f}".format(np.sum(o6) / o6.shape[0]))
        print("o7 = {:.4f}".format(np.sum(o7) / o7.shape[0]))
        print("o8 = {:.4f}".format(np.sum(o8) / o8.shape[0]))
    else:    
        m = r[:, 1:]
        a = a.squeeze(1)
    
        mask1 = (m == [False, False]).all(axis=1)
        mask2 = (m == [False, True]).all(axis=1)
        mask3 = (m == [True, False]).all(axis=1)
        mask4 = (m == [True, True]).all(axis=1)
    
        mask_a = np.where(a == 1, -1, 1)
        #cf_effect = (y_cf - y) * mask_a
        cf_effect = np.abs(y_cf - y)
        cf_bin = (np.greater(y_cf, 0.5).astype(int) - np.greater(y, 0.5).astype(int)) * mask_a

        o1 = cf_effect[mask1 == [True]]
        o2 = cf_effect[mask2 == [True]]
        o3 = cf_effect[mask3 == [True]]
        o4 = cf_effect[mask4 == [True]]

        o1_bin = cf_bin[mask1 == [True]]
        o2_bin = cf_bin[mask2 == [True]]
        o3_bin = cf_bin[mask3 == [True]]
        o4_bin = cf_bin[mask4 == [True]]

        print("*" * 20)
        print("causal effect of the classifier")
        print("cf = {}".format(np.sum(cf_effect) / cf_effect.shape[0]))
        print("o1 = {}".format(np.sum(o1) / o1.shape[0]))
        print("o2 = {}".format(np.sum(o2) / o2.shape[0]))
        print("o3 = {}".format(np.sum(o3) / o3.shape[0]))
        print("o4 = {}".format(np.sum(o4) / o4.shape[0]))
        print("o1_bin = {}".format(np.sum(o1_bin) / o1_bin.shape[0]))
        print("o2_bin = {}".format(np.sum(o2_bin) / o2_bin.shape[0]))
        print("o3_bin = {}".format(np.sum(o3_bin) / o3_bin.shape[0]))
        print("o4_bin = {}".format(np.sum(o4_bin) / o4_bin.shape[0]))"""

def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b +1e-10) for (a, b) in zip(A, B)])

    return chi

def generated_dataset(datasets, args):
    factual_whole_data, counter_whole_data = None, None

    for dataset in datasets:
        npzfile = os.path.join(args.save_path, dataset + '.npz')
        dat = np.load(npzfile)
        
        if args.use_real:
            gen_input = np.concatenate((dat["input_real"], dat["a"]), 1)
            gen_y = dat["y_real"]
        else:
            gen_input = np.concatenate((dat['input'], dat['a']), 1)
            gen_y = dat['y']
        col = ['input' + str(i) for i in range(gen_input.shape[1])]
        col.append('y')
        factual_data = np.concatenate((gen_input, gen_y), 1)
        factual_whole_data = np.concatenate((factual_whole_data, factual_data), 0) \
            if dataset != 'train' else factual_data

        gen_input_cf = np.concatenate((dat['input_cf'], dat['a_cf']), 1)
        gen_y_cf = dat['y_cf']
        counter_data = np.concatenate((gen_input_cf, gen_y_cf), 1)
        counter_whole_data = np.concatenate((counter_whole_data, counter_data), 0) \
            if dataset != 'train' else counter_data

    factual_df = pd.DataFrame(data=factual_whole_data, columns=col)
    counter_df = pd.DataFrame(data=counter_whole_data, columns=col)
    return factual_df, counter_df, col

def original_dataset(loaders, col):
    i_all, a_all, y_all = None, None, None
    for idx1, loader in enumerate(loaders):
        for idx2, (r, d, a, y) in enumerate(loader):
            idx = idx1 + idx2
            i = torch.cat((r, d), 1)
            i_all = torch.cat((i_all, i), 0) if idx != 0 else i
            a_all = torch.cat((a_all, a), 0) if idx != 0 else a
            y_all = torch.cat((y_all, y), 0) if idx != 0 else y

    data = (i_all, a_all, y_all)
    data = torch.cat(data, 1)
    data = data.cpu().detach().numpy()
    org_df = pd.DataFrame(data=data, columns=col)
    return org_df


def baseline_classifier(train_loader, valid_loader, test_loader, args, logger):
    factual_df, counter_df, col = generated_dataset(['train'], args)
    train_df = original_dataset([train_loader, valid_loader], col)
    test_df = original_dataset([test_loader], col)

    # The input is [r, d, a]
    input = train_df[col[:-1]].values
    y = train_df[col[-1]].values
    
    clfs = []
    if args.dataset == "law":
        clf = SM_LinearRegression()
        clf.fit(input, y)
        test_pred = clf.predict(test_df[col[:-1]].values)

        mse = mean_squared_error(test_df[col[-1]].values, test_pred, squared=False)
        logger.info("Mean squared error of Linear Regression -->    {:.4f}".format(mse))
        clfs.append(clf)
    else:
        #for name, clf in zip(['LR', 'SVM'],
        #                     [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        for name, clf in zip(["LR"], [LogisticRegression(penalty="l2", solver="liblinear")]):
            clf.fit(input, y)
            # predict y from x
            factual_df[name] = clf.predict(factual_df[col[:-1]].values)
            test_df[name] = clf.predict(test_df[col[:-1]].values)

            # compare real y and predicted y
            acc = accuracy_score(test_df[col[-1]].values, test_df[name])
            logger.info('accuracy of {:3s} train -->    test: {:.4f}'.format(name, acc))
            clfs.append(clf)
    
    for clf in clfs:
        test_dat = np.load(os.path.join(args.save_path, "test.npz"))
        input_factual = np.concatenate([test_dat["input_real"], test_dat["a"]], axis=1)
        y_factual = clf.predict(input_factual)
        input_counter = np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
        y_counter = clf.predict(input_counter)
        a = test_dat["a"]
        if args.dataset == "law":
            r = test_dat["input_real"][:, :8]
        else:
            r = test_dat["input_real"][:, :3]
        cf_eval(r, y_factual, y_counter, a, args)
        
        test_dat = np.load(os.path.join(args.save_path, "test_curve.npz"))
        curve_factual = np.concatenate([test_dat["input"], test_dat["a"]], axis=-1)
        curve_counter = np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=-1)
        y_factual = clf.predict(curve_factual)
        y_counter = clf.predict(curve_counter)
        np.save(os.path.join(args.save_path, "{}_baseline_factual".format(args.use_real)), y_factual)
        np.save(os.path.join(args.save_path, "{}_baseline_counter".format(args.use_real)), y_counter)
        

def fair_whole_classifier(train_loader, valid_loader, test_loader, args, logger):
    line = ''
    #datasets = ['train', 'valid', 'test']
    #loaders = [train_loader, valid_loader, test_loader]
    datasets = ["train", "valid"]
    loaders = [test_loader]
    factual_df, counter_df, col = generated_dataset(datasets, args)
    test_df = original_dataset(loaders, col)

    input = factual_df[col[:-1]].values
    y = factual_df[col[-1]].values

    f_data = np.asarray(factual_df[col].values).astype(int)
    original = np.asarray(test_df[col].values).astype(int)

    f_data = np.sum(f_data, 0)
    original = np.sum(original, 0)

    original = original / np.sum(original)
    f_data = f_data / np.sum(f_data)
    chi2 = chi2_distance(original, f_data)
    logger.info('(factual) chi squared distance is: {:.4f}'.format(chi2))

    cf_data = np.asarray(counter_df[col].values).astype(int)
    cf_data = np.sum(cf_data, 0)

    cf_data = cf_data / np.sum(cf_data)
    chi2_cf = chi2_distance(original, cf_data)
    logger.info('(counter) chi squared distance is: {:.4f}'.format(chi2_cf))

    whole_df = [factual_df, counter_df]
    whole_df = pd.concat(whole_df)
    whole_data = np.asarray(whole_df[col].values).astype(int)
    whole_data = np.sum(whole_data, 0)

    cf_data = whole_data / np.sum(whole_data)
    chi2_whole = chi2_distance(original, cf_data)
    logger.info('(whole) chi squared distance is: {:.4f}'.format(chi2_whole))

    """#for name, clf in zip(['LR', 'SVM'],
    #                     [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
    for name, clf in zip(["LR"], [LogisticRegression(penalty="l2", solver="liblinear")]):
        clf.fit(input, y)
        # predict y from x
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} factual -->    real: {:.4f}'.format(name, acc))
        line += str(acc)
        line += '\t'
    line += str(chi2) + '\t'

    input = counter_df[col[:-1]].values
    y = counter_df[col[-1]].values

    #for name, clf in zip(['LR', 'SVM'],
    #                     [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
    for name, clf in zip(["LR"], [LogisticRegression(penalty="l2", solver="liblinear")]):
        clf.fit(input, y)
        # predict y from input
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} counter -->    real: {:.4f}'.format(name, acc))"""
    
    input = whole_df[col[:-1]].values
    y = whole_df[col[-1]].values
    
    clfs_whole = []
    if args.dataset == "law":
        clf = SM_LinearRegression()
        clf.fit(input, y)
        test_pred = clf.predict(test_df[col[:-1]].values)

        mse = mean_squared_error(test_df[col[-1]].values, test_pred, squared=False)
        logger.info("mean squared error of Linear Regression -->    {:.4f}".format(mse))
        line += str(mse)
        line += "\t"
        clfs_whole.append(clf)
        line += str(chi2_whole) + "\n"
    else:
        #for name, clf in zip(['LR', 'SVM'],
        #                     [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        for name, clf in zip(["LR"], [LogisticRegression(penalty="l2", solver="liblinear")]):
            clf.fit(input, y)
            # predict y from input
            factual_df[name] = clf.predict(factual_df[col[:-1]].values)
            test_df[name] = clf.predict(test_df[col[:-1]].values)

            # compare real y and predicted y
            acc = accuracy_score(test_df[col[-1]].values, test_df[name])
            logger.info('accuracy of {:3s} whole -->    real: {:.4f}'.format(name, acc))
            line += str(acc)
            line += '\t'
            clfs_whole.append(clf)
        line += str(chi2_whole) + '\n'
   
    for clf in clfs_whole:
        test_dat = np.load(os.path.join(args.save_path, "test.npz"))
        input_factual = np.concatenate([test_dat["input_real"], test_dat["a"]], axis=-1)
        y_factual = clf.predict(input_factual)
        input_counter = np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=-1)
        y_counter = clf.predict(input_counter)
        a = test_dat["a"]
        if args.dataset == "law":
            r = test_dat["input_real"][:, :8]
        else:
            r = test_dat["input_real"][:, :3]
        cf_eval(r, y_factual, y_counter, a, args)
        
        test_dat = np.load(os.path.join(args.save_path, "test_curve.npz"))
        curve_factual = np.concatenate([test_dat["input"], test_dat["a"]], axis=-1)
        curve_counter = np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=-1)
        y_factual = clf.predict(curve_factual)
        y_counter = clf.predict(curve_counter)
        np.save(os.path.join(args.save_path, "{}_cvae_factual".format(args.use_real)), y_factual)
        np.save(os.path.join(args.save_path, "{}_cvae_counter".format(args.use_real)), y_counter)

    file_dir = os.path.abspath(os.path.join(args.save_path, os.pardir))
    file_dir = os.path.join(file_dir, 'whole_log.txt')
    if not os.path.exists(file_dir):
        print('file not exist!')
    else:
        f = open(file_dir, 'a')
    f.write(line)
    f.close()

"""def l2_classifier(args, logger):
    train_dat = np.load(os.path.join(args.save_path, "train.npz"))
    valid_dat = np.load(os.path.join(args.save_path, "valid.npz"))
    test_dat = np.load(os.path.join(args.save_path, "test.npz"))
    
    f_data = np.concatenate([
        np.concatenate([train_dat["input"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input"], test_dat["a"]], axis=1)
    ], axis=0)
    original = np.concatenate([
        np.concatenate([train_dat["input_real"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input_real"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input_real"], test_dat["a"]], axis=1)
    ], axis=0)

    f_data = np.sum(f_data, 0)
    original = np.sum(original, 0)

    original = original / np.sum(original)
    f_data = f_data / np.sum(f_data)
    chi2 = chi2_distance(original, f_data)
    logger.info("(factual) chi squared distance is: {:.4f}".format(chi2))

    cf_data = np.concatenate([
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1),
        np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    ], axis=0)
    cf_data = np.sum(cf_data, 0)

    cf_data = cf_data / np.sum(cf_data)
    chi2_cf = chi2_distance(original, cf_data)
    logger.info("(counter) chi squared distance is: {:.4f}".format(chi2_cf))

    whole_data = np.concatenate([
        np.concatenate([train_dat["input"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input"], test_dat["a"]], axis=1),
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1),
        np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    ], axis=0)
    whole_data = np.sum(whole_data, 0)

    whole_data = whole_data / np.sum(whole_data)
    chi2_whole = chi2_distance(original, whole_data)
    logger.info("(whole) chi squared distance is: {:.4f}".format(chi2_whole))

    #input = np.concatenate([train_dat["u"], valid_dat["u"]], axis=0)
    if args.dataset == "law":
        input = np.concatenate([
            np.concatenate([train_dat["u"], train_dat["input_real"][:, :8]], axis=1),
            np.concatenate([valid_dat["u"], valid_dat["input_real"][:, :8]], axis=1),
        ])
    else:
        input = np.concatenate([
            np.concatenate([train_dat["u"], train_dat["input_real"][:, :3]], axis=1),
            np.concatenate([valid_dat["u"], valid_dat["input_real"][:, :3]], axis=1),
        ])
        print("r = {}".format(train_dat["input_real"][1, :3]))
    y = np.concatenate([train_dat["y_real"], valid_dat["y_real"]], axis=0)
    
    if args.dataset == "law":
        clf = SM_LinearRegression()
        clf.fit(input, y)
        test_input = np.concatenate([test_dat["u"], test_dat["input_real"][:, :8]], axis=1)
        test_pred = clf.predict(test_input)
        
        mse = mean_squared_error(test_dat["y_real"], test_pred, squared=False)
        logger.info("mean squared error of Linear Regression -->    {:.4f}".format(mse))
    else:
        #for name, clf in zip(['LR', 'SVM'],
        #                     [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        for name, clf in zip(["LR"], [LogisticRegression(penalty="l2", solver="liblinear")]):
            clf.fit(input, y)
            test_input = np.concatenate([test_dat["u"], test_dat["input_real"][:, :3]], axis=1)
            test_pred = clf.predict(test_input)

            acc = accuracy_score(test_dat["y_real"].ravel(), test_pred.ravel())
            logger.info("accuracy of {:3s} whole -->     real: {:.4f}".format(name, acc))
    y_factual = test_pred
    y_counter = test_pred
    print("save path = {}".format(args.save_path))
    np.save(os.path.join(args.save_path, "{}_l2_factual".format(args.use_real)), y_factual)
    np.save(os.path.join(args.save_path, "{}_l2_counter".format(args.use_real)), y_counter)"""

def l2_classifier(args, logger):
    train_dat = np.load(os.path.join(args.save_path, "train.npz"))
    valid_dat = np.load(os.path.join(args.save_path, "valid.npz"))
    test_dat = np.load(os.path.join(args.save_path, "test.npz"))
    
    f_data = np.concatenate([
        np.concatenate([train_dat["input"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input"], test_dat["a"]], axis=1)
    ], axis=0)
    original = np.concatenate([
        np.concatenate([train_dat["input_real"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input_real"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input_real"], test_dat["a"]], axis=1)
    ], axis=0)

    f_data = np.sum(f_data, 0)
    original = np.sum(original, 0)

    original = original / np.sum(original)
    f_data = f_data / np.sum(f_data)
    chi2 = chi2_distance(original, f_data)
    logger.info("(factual) chi squared distance is: {:.4f}".format(chi2))

    cf_data = np.concatenate([
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1),
        np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    ], axis=0)
    cf_data = np.sum(cf_data, 0)

    cf_data = cf_data / np.sum(cf_data)
    chi2_cf = chi2_distance(original, cf_data)
    logger.info("(counter) chi squared distance is: {:.4f}".format(chi2_cf))

    whole_data = np.concatenate([
        np.concatenate([train_dat["input"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input"], test_dat["a"]], axis=1),
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1),
        np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    ], axis=0)
    whole_data = np.sum(whole_data, 0)

    whole_data = whole_data / np.sum(whole_data)
    chi2_whole = chi2_distance(original, whole_data)
    logger.info("(whole) chi squared distance is: {:.4f}".format(chi2_whole))

    #input = np.concatenate([train_dat["u"], valid_dat["u"]], axis=0)
    if args.dataset == "law":
        input = np.concatenate([
            np.concatenate([train_dat["u"], train_dat["input_real"][:, :8]], axis=1),
            np.concatenate([valid_dat["u"], valid_dat["input_real"][:, :8]], axis=1),
        ])
    else:
        input = np.concatenate([
            np.concatenate([train_dat["u"], train_dat["input_real"][:, :3]], axis=1),
            np.concatenate([valid_dat["u"], valid_dat["input_real"][:, :3]], axis=1),
        ])
    y = np.concatenate([train_dat["y_real"], valid_dat["y_real"]], axis=0)
    
    if args.dataset == "law":
        clf = SM_LinearRegression()
        #clf = MLPRegressor(random_state=1, max_iter=1500)
        clf.fit(input, y)
        test_input = np.concatenate([test_dat["u"], test_dat["input_real"][:, :8]], axis=1)
        test_pred = clf.predict(test_input)
        
        mse = mean_squared_error(test_dat["y_real"], test_pred, squared=False)
        logger.info("mean squared error of Linear Regression -->    {:.4f}".format(mse))
    else:
        #for name, clf in zip(['LR', 'SVM'],
        #                     [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        for name, clf in zip(["LR"], [LogisticRegression(penalty="l2", solver="liblinear")]):
        #for name, clf in zip(["MLP"], [MLPClassifier(random_state=1, max_iter=500)]):
            clf.fit(input, y)
            test_input = np.concatenate([test_dat["u"], test_dat["input_real"][:, :3]], axis=1)
            test_pred = clf.predict(test_input)

            acc = accuracy_score(test_dat["y_real"].ravel(), test_pred.ravel())
            logger.info("accuracy of {:3s} whole -->     real: {:.4f}".format(name, acc))
    #y_factual = test_pred
    #y_counter = test_pred
    test_dat = np.load(os.path.join(args.save_path, "test_curve.npz"))
    if args.dataset == "law":
        curve_factual = np.concatenate([test_dat["u"], test_dat["input"][:, :3]], 1)
        curve_counter = np.concatenate([test_dat["u"], test_dat["input"][:, :3]], 1)
    else:
        curve_factual = np.concatenate([test_dat["u"], test_dat["input"][:, :8]], 1)
        curve_counter = np.concatenate([test_dat["u"], test_dat["input"][:, :8]], 1)
    y_factual = clf.predict(curve_factual)
    y_counter = clf.predict(curve_counter)
    print("save path = {}".format(args.save_path))
    np.save(os.path.join(args.save_path, "{}_l2_factual".format(args.use_real)), y_factual)
    np.save(os.path.join(args.save_path, "{}_l2_counter".format(args.use_real)), y_counter)

def avg_classifier(args, logger):
    train_dat = np.load(os.path.join(args.save_path, "train.npz"))
    valid_dat = np.load(os.path.join(args.save_path, "valid.npz"))
    test_dat = np.load(os.path.join(args.save_path, "test.npz"))
    
    f_data = np.concatenate([
        np.concatenate([train_dat["input"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input"], test_dat["a"]], axis=1)
    ], axis=0)
    original = np.concatenate([
        np.concatenate([train_dat["input_real"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input_real"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input_real"], test_dat["a"]], axis=1)
    ], axis=0)

    f_data = np.sum(f_data, 0)
    original = np.sum(original, 0)

    original = original / np.sum(original)
    f_data = f_data / np.sum(f_data)
    chi2 = chi2_distance(original, f_data)
    logger.info("(factual) chi squared distance is: {:.4f}".format(chi2))

    cf_data = np.concatenate([
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1),
        np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    ], axis=0)
    cf_data = np.sum(cf_data, 0)

    cf_data = cf_data / np.sum(cf_data)
    chi2_cf = chi2_distance(original, cf_data)
    logger.info("(counter) chi squared distance is: {:.4f}".format(chi2_cf))

    whole_data = np.concatenate([
        np.concatenate([train_dat["input"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input"], test_dat["a"]], axis=1),
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1),
        np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    ], axis=0)
    whole_data = np.sum(whole_data, 0)

    whole_data = whole_data / np.sum(whole_data)
    chi2_whole = chi2_distance(original, whole_data)
    logger.info("(whole) chi squared distance is: {:.4f}".format(chi2_whole))
    
    if args.dataset == "law":
        input = np.concatenate([
            np.concatenate([train_dat["u"], train_dat["input_real"][:, :8], (train_dat["input_real"][:, 8:] + train_dat["input_cf"][:, 8:]) / 2], axis=1),
            np.concatenate([valid_dat["u"], valid_dat["input_real"][:, :8], (valid_dat["input_real"][:, 8:] + valid_dat["input_cf"][:, 8:]) / 2], axis=1)
            #np.concatenate([train_dat["u"], (train_dat["input_real"] + train_dat["input_cf"]) / 2], axis=1),
            #np.concatenate([valid_dat["u"], (valid_dat["input_real"] + valid_dat["input_cf"]) / 2], axis=1),
            #(train_dat["input_real"] + train_dat["input_cf"]) / 2,
            #(valid_dat["input_real"] + valid_dat["input_cf"]) / 2
        ])
    else:
        input = np.concatenate([
            #np.concatenate([train_dat["u"], train_dat["input_real"][:, :8], (train_dat["input_real"][:, 8:] + train_dat["input_cf"][:, 8:]) / 2], axis=1),
            #np.concatenate([valid_dat["u"], valid_dat["input_real"][:, :8], (valid_dat["input_real"][:, 8:] + valid_dat["input_cf"][:, 8:]) / 2], axis=1),
            np.concatenate([train_dat["u"], train_dat["input_real"][:, :3], (train_dat["input_real"][:, 3:] + train_dat["input_cf"][:, 3:]) / 2], axis=1),
            np.concatenate([valid_dat["u"], valid_dat["input_real"][:, :3], (valid_dat["input_real"][:, 3:] + valid_dat["input_cf"][:, 3:]) / 2], axis=1)
            #(train_dat["input_real"] + train_dat["input_cf"]) / 2,
            #(valid_dat["input_real"] + valid_dat["input_cf"]) / 2
        ], axis=0)
    print("obseved data = {}".format(test_dat["input_real"][2]))
    print("factual data = {}".format(test_dat["input"][2]))
    print("counter factual data = {}".format(test_dat["input_cf"][2]))
    #for i in range(test_dat["input_real"].shape[0]):
    #    if (test_dat["input_real"][i][:8] != test_dat["input"][i][:8]).any():
    #        print("i= {}, not equal".format(i))
    #        print("real data = {}".format(test_dat["input_real"][i]))
    #        print("cf data = {}".format(test_dat["input"][i]))
    #input = np.concatenate([
    #    (train_dat["input_real"] + train_dat["input_cf"]) / 2,
    #    (valid_dat["input_real"] + valid_dat["input_cf"]) / 2
    #], axis=0)
    y = np.concatenate([train_dat["y_real"], valid_dat["y_real"]], axis=0)
    
    clfs_whole = []
    if args.dataset == "law":
        clf = SM_LinearRegression()
        #clf = MLPRegressor(random_state=1, max_iter=1500)
        clf.fit(input, y)
        test_input = np.concatenate([
            test_dat["u"],
            test_dat["input_real"][:, :8],
            (test_dat["input_real"][:, 8:] + test_dat["input_cf"][:, 8:]) / 2
            #(test_dat["input_real"] + test_dat["input_cf"]) / 2
        ], axis=1)
        #test_input = (test_dat["input_real"] + test_dat["input_cf"]) / 2
        test_pred = clf.predict(test_input)

        mse = mean_squared_error(test_dat["y_real"], test_pred, squared=False)
        logger.info("mean squared error of Linear Regression -->    {:.4f}".format(mse))
        clfs_whole.append(clf)
    else:
        #for name, clf in zip(['LR', 'SVM'],
        #                     [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        for name, clf in zip(["LR"], [LogisticRegression(penalty="l2", solver="liblinear")]):
        #for name, clf in zip(["MLP"], [MLPClassifier(random_state=1, max_iter=500)]):
            clf.fit(input, y)
            test_input = np.concatenate([
                test_dat["u"],
                test_dat["input_real"][:, :3],
                (test_dat["input_real"][:, 3:] + test_dat["input_cf"][:, 3:]) / 2
            ], axis=1)
            #test_input = (test_dat["input_real"] + test_dat["input_cf"]) / 2
            test_pred = clf.predict(test_input)

            acc = accuracy_score(test_dat["y_real"].ravel(), test_pred.ravel())
            logger.info("accuracy of {:3s} whole -->     real: {:.4f}".format(name, acc))
            clfs_whole.append(clf)
    
    for clf in clfs_whole:
        if args.dataset == "law":
            input_factual = np.concatenate([
                test_dat["u"],
                test_dat["input_real"][:, :8],
                (test_dat["input_real"][:, 8:] + test_dat["input_cf"][:, 8:]) / 2
                #(test_dat["input_real"] + test_dat["input_cf"]) / 2
            ], axis=1)
            #input_factual = (test_dat["input_real"] + test_dat["input_cf"]) / 2
            y_factual = clf.predict(input_factual)
            input_counter = np.concatenate([
                test_dat["u"],
                test_dat["input_real"][:, :8],
                (test_dat["input_cf"][:, 8:] + test_dat["input"][:, 8:]) / 2
                #(test_dat["input_cf"] + test_dat["input"]) / 2
            ], axis=1)
            #input_counter = (test_dat["input_cf"] + test_dat["input"]) / 2
            y_counter = clf.predict(input_counter)
        else:
            input_factual = np.concatenate([
                test_dat["u"],
                test_dat["input_real"][:, :3],
                (test_dat["input_real"][:, 3:] + test_dat["input_cf"][:, 3:]) / 2
                #(test_dat["input_real"] + test_dat["input_cf"]) / 2
            ], axis=1)
            #input_factual = (test_dat["input_real"] + test_dat["input_cf"]) / 2
            y_factual = clf.predict(input_factual)
            input_counter = np.concatenate([
                test_dat["u"],
                test_dat["input_real"][:, :3],
                (test_dat["input_cf"][:, 3:] + test_dat["input"][:, 3:]) / 2
                #(test_dat["input_cf"] + test_dat["input"]) / 2
            ], axis=1)
            #input_counter = (test_dat["input_cf"] + test_dat["input"]) / 2
            y_counter = clf.predict(input_counter)
        a = test_dat["a"]
        if args.dataset == "law":
            r = test_dat["input_real"][:, :8]
        else:
            r = test_dat["input_real"][:, :3]
        cf_eval(r, y_factual, y_counter, a, args)

        test_dat = np.load(os.path.join(args.save_path, "test_curve.npz"))
        if args.dataset == "law":
            curve_factual = np.concatenate([test_dat["u"], test_dat["input"][:, :8], (test_dat["input"][:, 8:] + test_dat["input_cf"][:, 8:]) / 2], 1)
            curve_counter = np.concatenate([test_dat["u"], test_dat["input"][:, :8], (test_dat["input_cf"][:, 8:] + test_dat["input"][:, 8:]) / 2], 1)
        else:
            curve_factual = np.concatenate([test_dat["u"], test_dat["input"][:, :3], (test_dat["input"][:, 3:] + test_dat["input_cf"][:, 3:]) / 2], 1)
            curve_counter = np.concatenate([test_dat["u"], test_dat["input"][:, :3], (test_dat["input_cf"][:, 3:] + test_dat["input"][:, 3:]) / 2], 1)
        y_factual = clf.predict(curve_factual)
        y_counter = clf.predict(curve_counter)
        np.save(os.path.join(args.save_path, "{}_ours_factual".format(args.use_real)), y_factual)
        np.save(os.path.join(args.save_path, "{}_ours_counter".format(args.use_real)), y_counter)
    


"""def avg_classifier(args, logger):
    train_dat = np.load(os.path.join(args.save_path, "train.npz"))
    valid_dat = np.load(os.path.join(args.save_path, "valid.npz"))
    test_dat = np.load(os.path.join(args.save_path, "test.npz"))
    
    f_data = np.concatenate([
        np.concatenate([train_dat["input"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input"], test_dat["a"]], axis=1)
    ], axis=0)
    original = np.concatenate([
        np.concatenate([train_dat["input_real"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input_real"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input_real"], test_dat["a"]], axis=1)
    ], axis=0)

    f_data = np.sum(f_data, 0)
    original = np.sum(original, 0)

    original = original / np.sum(original)
    f_data = f_data / np.sum(f_data)
    chi2 = chi2_distance(original, f_data)
    logger.info("(factual) chi squared distance is: {:.4f}".format(chi2))

    cf_data = np.concatenate([
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1),
        np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    ], axis=0)
    cf_data = np.sum(cf_data, 0)

    cf_data = cf_data / np.sum(cf_data)
    chi2_cf = chi2_distance(original, cf_data)
    logger.info("(counter) chi squared distance is: {:.4f}".format(chi2_cf))

    whole_data = np.concatenate([
        np.concatenate([train_dat["input"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input"], valid_dat["a"]], axis=1),
        np.concatenate([test_dat["input"], test_dat["a"]], axis=1),
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1),
        np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    ], axis=0)
    whole_data = np.sum(whole_data, 0)

    whole_data = whole_data / np.sum(whole_data)
    chi2_whole = chi2_distance(original, whole_data)
    logger.info("(whole) chi squared distance is: {:.4f}".format(chi2_whole))
    
    input = np.concatenate([
        np.concatenate([train_dat["u"], (train_dat["input_real"] + train_dat["input_cf"]) / 2], axis=1),
        np.concatenate([valid_dat["u"], (valid_dat["input_real"] + valid_dat["input_cf"]) / 2], axis=1)
    ], axis=0)
    y = np.concatenate([train_dat["y_real"], valid_dat["y_real"]], axis=0)
    print("observed data = {}".format(train_dat["input_real"][1]))
    print("factual data = {}".format(train_dat["input"][1]))
    print("counterfactual data = {}".format(train_dat["input_cf"][1]))
    
    clfs_whole = []
    if args.dataset == "law":
        clf = SM_LinearRegression()
        clf.fit(input, y)
        test_input = np.concatenate([
            test_dat["u"],
            (test_dat["input_real"] + test_dat["input_cf"]) / 2
        ], axis=1)
        test_pred = clf.predict(test_input)

        mse = mean_squared_error(test_dat["y_real"], test_pred, squared=False)
        logger.info("mean squared error of Linear Regression -->    {:.4f}".format(mse))
        clfs_whole.append(clf)
    else:
        #for name, clf in zip(['LR', 'SVM'],
        #                     [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        for name, clf in zip(["LR"], [LogisticRegression(penalty="l2", solver="liblinear")]):
            clf.fit(input, y)
            test_input = np.concatenate([
                test_dat["u"],
                (test_dat["input_real"] + test_dat["input_cf"]) / 2
            ], axis=1)
            test_pred = clf.predict(test_input)

            acc = accuracy_score(test_dat["y_real"].ravel(), test_pred.ravel())
            logger.info("accuracy of {:3s} whole -->     real: {:.4f}".format(name, acc))
            clfs_whole.append(clf)
    
    for clf in clfs_whole:
        input_factual = np.concatenate([
            test_dat["u"],
            (test_dat["input_real"] + test_dat["input_cf"]) / 2
        ], axis=1)
        y_factual = clf.predict(input_factual)
        input_counter = np.concatenate([
            test_dat["u"],
            (test_dat["input_cf"] + test_dat["input"]) / 2
        ], axis=1)
        y_counter = clf.predict(input_counter)
        a = test_dat["a"]
        if args.dataset == "law":
            r = test_dat["input_real"][:, :8]
        else:
            r = test_dat["input_real"][:, :3]
        cf_eval(r, y_factual, y_counter, a, args)
        np.save(os.path.join(args.save_path, "{}_ours_factual".format(args.use_real)), y_factual)
        np.save(os.path.join(args.save_path, "{}_ours_counter".format(args.use_real)), y_counter)"""

def reg_classifier(args, logger):
    train_dat = np.load(os.path.join(args.save_path, "train.npz"))
    valid_dat = np.load(os.path.join(args.save_path, "valid.npz"))
    test_dat = np.load(os.path.join(args.save_path, "test.npz"))

    factual_data = np.concatenate([
        np.concatenate([train_dat["input_real"], train_dat["a"]], axis=1),
        np.concatenate([valid_dat["input_real"], valid_dat["a"]], axis=1)
    ], axis=0)
    train_Y = np.concatenate([train_dat["y_real"], valid_dat["y_real"]], axis=0)

    counter_data = np.concatenate([
        np.concatenate([train_dat["input_cf"], train_dat["a_cf"]], axis=1),
        np.concatenate([valid_dat["input_cf"], valid_dat["a_cf"]], axis=1)
    ], axis=0)

    test_factual = np.concatenate([test_dat["input_real"], test_dat["a"]], axis=1)
    test_counter = np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=1)
    test_Y = test_dat["y_real"]
    
    print("obseved data = {}".format(test_dat["input_real"][2]))
    print("factual data = {}".format(test_dat["input"][2]))
    print("counter factual data = {}".format(test_dat["input_cf"][2]))

    clfs = []
    if args.dataset == "law":
        clf = Reg_linearRegression(size=factual_data.shape[1], alpha=0.002)
        clf.fit(factual_data, counter_data, train_Y)
        test_pred = clf.predict(test_factual)

        mse = mean_squared_error(test_Y, test_pred, squared=False)
        logger.info("mean squared error of Linear Regression -->    {:.4f}".format(mse))
        clfs.append(clf)
    else:
        clf = Reg_logisticRegression(size=factual_data.shape[1], alpha=0.002)
        clf.fit(factual_data, counter_data, train_Y)
        test_pred = clf.predict(test_factual)

        acc = accuracy_score(test_Y, test_pred)
        logger.info("accuracy of logistic regression -->    {:.4f}".format(acc))
        clfs.append(clf)
        
    for clf in clfs:
        a = test_dat["a"]
        if args.dataset == "law":
            r = test_dat["input_real"][:, :8]
        else:
            r = test_dat["input_real"][:, :3]
        y_factual = clf.predict(test_factual)
        y_counter = clf.predict(test_counter)
        cf_eval(r, y_factual, y_counter, a, args)
        
        test_dat = np.load(os.path.join(args.save_path, "test_curve.npz"))
        curve_factual = np.concatenate([test_dat["input"], test_dat["a"]], axis=1)
        curve_counter = np.concatenate([test_dat["input_cf"], test_dat["a_cf"]], axis=-1)
        y_factual = clf.predict(curve_factual)
        y_counter = clf.predict(curve_counter)
        np.save(os.path.join(args.save_path, "{}_reg_factual".format(args.use_real)), y_factual)
        np.save(os.path.join(args.save_path, "{}_reg_counter".format(args.use_real)), y_counter)

def fair_seperate_classifier(loader, args, logger, dataset='train'):
    factual_df, counter_df, col = generated_dataset([dataset], args)
    test_df = original_dataset([loader], col)

    input = factual_df[col[:-1]].values
    y = factual_df[col[-1]].values

    cf_data = np.asarray(factual_df[col].values).astype(int)
    original = np.asarray(test_df[col].values).astype(int)

    cf_data = np.sum(cf_data, 0)
    original = np.sum(original, 0)

    original = original / np.sum(original)
    cf_data = cf_data / np.sum(cf_data)
    chi2 = chi2_distance(original, cf_data)
    logger.info('chi squared distance is: {:.4f}'.format(chi2))

    for name, clf in zip(['LR', 'SVM'],
                         [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        clf.fit(input, y)
        # predict y from input
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} factual -->    real: {:.4f}'.format(name, acc))

    input = counter_df[col[:-1]].values
    y = counter_df[col[-1]].values

    for name, clf in zip(['LR', 'SVM'],
                         [LogisticRegression(penalty='l2', solver='liblinear'), SVC(kernel='poly', gamma='auto')]):
        clf.fit(input, y)
        # predict y from input
        factual_df[name] = clf.predict(factual_df[col[:-1]].values)
        test_df[name] = clf.predict(test_df[col[:-1]].values)

        # compare real y and predicted y
        acc = accuracy_score(test_df[col[-1]].values, test_df[name])
        logger.info('accuracy of {:3s} counter -->    real: {:.4f}'.format(name, acc))
