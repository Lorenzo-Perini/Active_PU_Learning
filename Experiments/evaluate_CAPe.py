import numpy as np
import pandas as pd
from multiprocessing import Pool, freeze_support, cpu_count
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import StratifiedKFold
from anomatools.models import SSDO
from sklearn.ensemble import IsolationForest
import collections, functools
from anomatools.models import SSDO

import pyximport                        #import cython such that CAPe can work!
pyximport.install()

from CAPe import *                      #this imports CAPe functions
from TIcE import *                      #this imports TIcE functions
from Kernel_MPE_grad_threshold import * #this imports km1 and km2 functions

def get_f1scores_wdiff_priors(data, y, real_cont, tmp_cont = 0.1, k = 5, ntimes = 10, name_ds = '', case = 2,
                                   n_splits = 5, n_iter = 3, n_jobs = cpu_count()):
    
    """ This function gets the F1 scores of SSDO when using different contamination factors according to all the methods used in
        the paper (CAPe). This function builds a dataframe with the final results and saves it on a csv file (remark: change the 
        path at the bottom or remove the line). The 5 methods for estimating the class prior in a PU dataset compared are: CAPe,
        TIcE, km1, km2 and the naive baseline with the real contamination factor.
        
        Parameters
        ----------
        data          : np.array of shape (n_samples, n_features). It is the entire dataset.
        y             : np.array of shape (n_samples,) containing all the labels.
        real_cont     : float regarding the REAL expected percentage of anomalies in the training set.
        tmp_cont      : float regarding the starting expected percentage of anomalies in the training set. Default=0.1.
        k             : int regarding the number of new labels required. Default = 5.
        ntimes        : int regarding the number of iterations for getting new k labels. Default = 10 (at least 50 examples).
        name_ds       : string containing the name of the dataset to output a meaningful csv file. Default empty string.
        case          : int getting 0 when the user's uncertainty is not present (PO), 2 when dealing with it (IO). Default=0.
        n_splits      : int regarding the number of splits inside the crossvalidation. Default = 5.
        n_iter        : int regarding the number of iterations of the whole method. Default = 3.
        n_jobs        : int regarding the number of jobs to run in parallel. Default = maximum number of jobs.

        Returns
        ----------
        F1_results    : dataframe containing the F1 results. The first columns is about the labels acquired (k) while the others
                        contain the F1 score for each method and each number of labels.
        prior_results : dataframe containing the prior results. The first columns is about the labels acquired (k) while the
                        others contain the prior estimates for each method and each number of labels.
    """

    F1_results = pd.DataFrame(data = k*np.arange(1, ntimes+1,1), columns = ['k'])
    F1_results['F1_CAPe'] = np.zeros(ntimes, dtype = float)
    F1_results['F1_TIcE'] = np.zeros(ntimes, dtype = float)
    F1_results['F1_km1'] = np.zeros(ntimes, dtype = float)
    F1_results['F1_km2'] = np.zeros(ntimes, dtype = float)
    F1_results['F1_real'] = np.zeros(ntimes, dtype = float)
    CAPe_prior = np.zeros(ntimes, dtype = float)
    TIcE_prior = np.zeros(ntimes, dtype = float)
    km1_prior = np.zeros(ntimes, dtype = float)
    km2_prior = np.zeros(ntimes, dtype = float)
    real_prior = np.zeros(ntimes, dtype = float)
    
    for num in np.arange(1,n_iter+1):

        skf = StratifiedKFold(n_splits=n_splits, random_state=331, shuffle=True)
        F1_CAPe = []
        F1_TIcE = []
        F1_km1 = []
        F1_km2 = []
        F1_real = []

        for train_index, test_index in skf.split(data, y):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = y[train_index], y[test_index]
            real_anomalies = np.where(y_train == 1)[0]
            
            f1cape, prior_cape = evaluate_CAPe(X_train, X_test, y_test, real_anomalies, k, ntimes, tmp_cont, case)
            F1_CAPe.append(f1cape)
            CAPe_prior += prior_cape
            
            f1tice, prior_tice = evaluate_TIcE(X_train, X_test, y_test, real_anomalies, k, ntimes, tmp_cont, case)
            F1_TIcE.append(f1tice)
            TIcE_prior += prior_tice

            f1km1, f1km2, prior_km1, prior_km2 = evaluate_km1km2(X_train, X_test, y_test, real_anomalies, k, ntimes,
                                                                 tmp_cont, case)
            F1_km1.append(f1km1)
            F1_km2.append(f1km2)
            km1_prior += prior_km1
            km2_prior += prior_km2

            f1real, prior_real = evaluate_realF1(X_train, X_test, y_test, real_cont, real_anomalies, k, ntimes, case)
            F1_real.append(f1real)
            real_prior += prior_real
            print('Done crossval for iter num:', num)

        FinalF1CAPe = dict(functools.reduce(lambda x, y: x.update(y) or x, F1_CAPe, collections.Counter()))
        FinalF1TIcE = dict(functools.reduce(lambda x, y: x.update(y) or x, F1_TIcE, collections.Counter()))
        FinalF1km1 = dict(functools.reduce(lambda x, y: x.update(y) or x, F1_km1, collections.Counter()))
        FinalF1km2 = dict(functools.reduce(lambda x, y: x.update(y) or x, F1_km2, collections.Counter()))
        FinalF1real = dict(functools.reduce(lambda x, y: x.update(y) or x, F1_real, collections.Counter()))

        for j in range(ntimes):
            tnfpfntp = FinalF1CAPe[int(k*(j+1))]
            FinalF1CAPe[int(k*(j+1))] = (2 * tnfpfntp[3]) / (2 * tnfpfntp[3] + tnfpfntp[1] + tnfpfntp[2])

            tnfpfntp = FinalF1TIcE[int(k*(j+1))]
            FinalF1TIcE[int(k*(j+1))] = (2 * tnfpfntp[3]) / (2 * tnfpfntp[3] + tnfpfntp[1] + tnfpfntp[2])

            tnfpfntp = FinalF1km1[int(k*(j+1))]
            FinalF1km1[int(k*(j+1))] = (2 * tnfpfntp[3]) / (2 * tnfpfntp[3] + tnfpfntp[1] + tnfpfntp[2])

            tnfpfntp = FinalF1km2[int(k*(j+1))]
            FinalF1km2[int(k*(j+1))] = (2 * tnfpfntp[3]) / (2 * tnfpfntp[3] + tnfpfntp[1] + tnfpfntp[2])

            tnfpfntp = FinalF1real[int(k*(j+1))]
            FinalF1real[int(k*(j+1))] = (2 * tnfpfntp[3]) / (2 * tnfpfntp[3] + tnfpfntp[1] + tnfpfntp[2])

        F1_results['F1_CAPe'] += list(FinalF1CAPe.values())
        F1_results['F1_TIcE'] += list(FinalF1TIcE.values())
        F1_results['F1_km1'] += list(FinalF1km1.values())
        F1_results['F1_km2'] += list(FinalF1km2.values())
        F1_results['F1_real'] += list(FinalF1real.values())
        print('Finished Iteration number', num,'out of', n_iter)
        
    prior_results = pd.DataFrame(data = k*np.arange(1, ntimes+1,1), columns = ['k'])
    prior_results['CAPe_prior'] = CAPe_prior/(ntimes*n_splits)
    prior_results['TIcE_prior'] = TIcE_prior/(ntimes*n_splits)
    prior_results['km1_prior'] = km1_prior/(ntimes*n_splits)
    prior_results['km2_prior'] = km2_prior/(ntimes*n_splits)
    prior_results['real_prior'] = real_prior/(ntimes*n_splits)
    
    F1_columns = ['F1_CAPe', 'F1_TIcE', 'F1_km1', 'F1_km2', 'F1_real']
    F1_results[F1_columns] = F1_results[F1_columns]/n_iter
    
    F1_results.to_csv('F1score_case_'+str(case)+name_ds+'.csv')
    prior_results.to_csv('prior_case_'+str(case)+name_ds+'.csv')
    
    return F1_results, prior_results


def evaluate_CAPe(X_train, X_test, y_test, real_anomalies = [], k = 5, ntimes = 10, tmp_cont = 0.1, 
                  case = 0, n_jobs = cpu_count()):
    
    """ Evaluating CAPe as provided in the paper. This function 1) gets CAPe's estimation of the class prior and 2) saves the
        confusion matrix's cell values (tn, fp, fn, tp) in order to compute after all the interations a unique F1 score.
        
        Parameters
        ----------
        X_train        : np.array of shape (n_samples, n_features). It is the training set.
        X_test         : np.array of shape (m_samples, n_features). It is the test set.
        y_test         : np.array of shape (m_samples,). It contains the test labels.
        real_anomalies : list of shape (n_samples,) containing the index of the real training anomalies. Only needed if case=2.
        k              : int regarding the number of new labels required.
        ntimes         : int regarding the number of iterations for getting new k labels.
        tmp_cont       : float regarding the starting expected percentage of anomalies in the training set. Default=0.1.
        case           : int getting 0 when the user's uncertainty is not present (PO), 2 when dealing with it (IO). Default=0.
        n_jobs         : int regarding the number of jobs to run in parallel. Default = maximum number of jobs.
        
        Returns
        ----------
        F1_CAPe        : dict containing for each key multiple of k (k, 2*k, 3*k,...,ntimes*k) the array [tn,fp,fn,tp] obtained
                         with the estimate of the prior in such case.
        class_priors   : array of shape (ntimes,) containing every k new labels the estimate of the class prior.
        
    """

    n = np.shape(X_train)[0]
    tmp_cont = 0.1
    query_list = []
    labeled_ex = np.zeros(n, dtype=np.int)
    ker = KernelDensity().fit(X_train)
    dmu = [np.exp(ker.score(X_train[i:i+1])) for i in range(n)]
    mean_prob_term = math.log(np.mean(dmu),10) #Take the log density
    F1_CAPe = {}
    class_priors = np.zeros(ntimes, dtype = float)
    
    for j in range(ntimes):
        
        prior, labeled_ex, query_list = CAPe(X_train, labeled_ex, query_list, k, real_anomalies, tmp_cont, mean_prob_term,
                                             case, n_jobs)
        
        class_priors[j] = prior
        
        tmp_cont = 1 - min(prior,0.9999) #update the contamination factor
        
        F1_CAPe[int(k*(j+1))] = get_tnfpfntp(X_train, labeled_ex, X_test, y_test, tmp_cont) #compute the performance of the 
                                                                                         #classifier
        
    return F1_CAPe, class_priors


def evaluate_TIcE(X_train, X_test, y_test, real_anomalies = [], k = 5, ntimes = 10, tmp_cont = 0.1, case = 0):
    
    """ This function for evaluating TIcE does 1) query examples until k new labels are acquired, 2) get
        TIcE's estimation of the class prior and 3) save the confusion matrix's cell values (tn, fp, fn, tp) in order to
        compute after all the interations a unique F1 score.
        
        Parameters
        ----------
        X_train        : np.array of shape (n_samples, n_features). It is the training set.
        X_test         : np.array of shape (m_samples, n_features). It is the test set.
        y_test         : np.array of shape (m_samples,). It contains the test labels.
        real_anomalies : list of shape (n_samples,) containing the index of the real training anomalies. Only needed if case=2.
        k              : int regarding the number of new labels required.
        ntimes         : int regarding the number of iterations for getting new k labels.
        tmp_cont       : float regarding the starting expected percentage of anomalies in the training set. Default=0.1.
        case           : int getting 0 when the user's uncertainty is not present (PO), 2 when dealing with it (IO). Default=0.
        
        Returns
        ----------
        F1_TIcE        : dict containing for each key multiple of k (k, 2*k, 3*k,...,ntimes*k) the array [tn,fp,fn,tp] obtained
                         with the estimate of the prior in such case.
        class_priors   : array of shape (ntimes,) containing every k new labels the estimate of the class prior.
    """

    n = np.shape(X_train)[0]
    tmp_cont = 0.1
    query_list = []
    labeled_ex = np.zeros(n, dtype=np.int)
    F1_TIcE = {}
    class_priors = np.zeros(ntimes, dtype = float)
    scaler = MinMaxScaler()
    
    for j in range(ntimes):
        
        labeled_ex, query_list = query_at_least_k_points(X_train, labeled_ex, real_anomalies, query_list, k, tmp_cont,\
                                                       case)

        _, prior = run_from_elsewhere(data = scaler.fit_transform(X_train), labels = labeled_ex) #run TIcE algo and find c
        
        class_priors[j] = prior

        tmp_cont = 1 - min(prior, 0.9999)
        
        F1_TIcE[int(k*(j+1))] = get_tnfpfntp(X_train, labeled_ex, X_test, y_test, tmp_cont) #compute the performance of the 
        
    return F1_TIcE, class_priors

def evaluate_km1km2(X_train, X_test, y_test, real_anomalies = [], k = 5, ntimes = 10, tmp_cont = 0.1, case = 0):
    
    """ This function for evaluating km1 and km2 does 1) query examples until k new labels are acquired, 2) get
        km1's (km2's) estimation of the class prior and 3) save the confusion matrix's cell values (tn, fp, fn, tp) in order to
        compute after all the interations a unique F1 score. It repeats the whole process twice, once for km1 and once for km2.
        
        Parameters
        ----------
        X_train         : np.array of shape (n_samples, n_features). It is the training set.
        X_test          : np.array of shape (m_samples, n_features). It is the test set.
        y_test          : np.array of shape (m_samples,). It contains the test labels.
        real_anomalies  : list of shape (n_samples,) containing the index of the real training anomalies. Only needed if case=2.
        k               : int regarding the number of new labels required.
        ntimes          : int regarding the number of iterations for getting new k labels.
        tmp_cont        : float regarding the starting expected percentage of anomalies in the training set. Default=0.1.
        case            : int getting 0 when the user's uncertainty is not present (PO), 2 when dealing with it (IO). Default=0.
        
        Returns
        ----------
        F1_km1          : dict containing for each key multiple of k (k, 2*k, 3*k,...,ntimes*k) the array [tn,fp,fn,tp] obtained
                          with the estimate of the prior in such case (km1).
        F1_km2          : dict containing for each key multiple of k (k, 2*k, 3*k,...,ntimes*k) the array [tn,fp,fn,tp] obtained
                          with the estimate of the prior in such case (km2).
        class_priors_km1: array of shape (ntimes,) containing every k new labels the estimate of the class prior for km1.
        class_priors_km2: array of shape (ntimes,) containing every k new labels the estimate of the class prior for km2.
    """
    
    n = np.shape(X_train)[0]
    query_list_km1 = []
    query_list_km2 = []
    labeled_ex_km1 = np.zeros(n, dtype=np.int)
    labeled_ex_km2 = np.zeros(n, dtype=np.int)
    F1_km1 = {}
    F1_km2 = {}
    class_priors_km1 = np.zeros(ntimes, dtype = float)
    class_priors_km2 = np.zeros(ntimes, dtype = float)
    km1_tmp_cont = 0.1
    km2_tmp_cont = 0.1
    km1_query_list = []
    km2_query_list = []

    for j in range(ntimes):
        
        labeled_ex_km1, query_list_km1 = query_at_least_k_points(X_train, labeled_ex_km1, real_anomalies, query_list_km1, k,\
                                                                 tmp_cont, case)
        
        X_component = np.where(labeled_ex_km1 == 1)[0]
        X_component = X_train[X_component]
        X_mixture = np.where(labeled_ex_km1 == 0)[0]
        X_mixture = X_train[X_mixture]
        prior_km1,_ = wrapper(X_mixture, X_component)
        class_priors_km1[j] = prior_km1
        
        km1_tmp_cont = 1-prior_km1
        F1_km1[int(k*(j+1))] = get_tnfpfntp(X_train, labeled_ex_km1, X_test, y_test, km1_tmp_cont) 
        
        #---------------------------------------------------------------------------------------------------
        labeled_ex_km2, query_list_km2 = query_at_least_k_points(X_train, labeled_ex_km2, real_anomalies, query_list_km2, k,\
                                                                 tmp_cont, case)
        
        X_component = np.where(labeled_ex_km2 == 1)[0]
        X_component = X_train[X_component]
        X_mixture = np.where(labeled_ex_km2 == 0)[0]
        X_mixture = X_train[X_mixture]
        _, prior_km2 = wrapper(X_mixture, X_component)
        class_priors_km2[j] = prior_km2
        
        km1_tmp_cont = 1-prior_km2
        F1_km2[int(k*(j+1))] = get_tnfpfntp(X_train, labeled_ex_km2, X_test, y_test, km2_tmp_cont) 

    return F1_km1, F1_km2, class_priors_km1, class_priors_km2

def evaluate_realF1(X_train, X_test, y_test, real_cont, real_anomalies = [], k = 5, ntimes = 10, case = 0):
    
    """ This function for evaluating the performance of SSDO with real contamination factor does 1) query examples until k new
        labels are acquired and 2) save the confusion matrix's cell values (tn, fp, fn, tp) in order to compute after all the 
        interations a unique F1 score. It never updates the contamination factor, as it is the true one.
        
        Parameters
        ----------
        X_train        : np.array of shape (n_samples, n_features). It is the training set.
        X_test         : np.array of shape (m_samples, n_features). It is the test set.
        y_test         : np.array of shape (m_samples,). It contains the test labels.
        real_cont      : float regarding the REAL expected percentage of anomalies in the training set.
        real_anomalies : list of shape (n_samples,) containing the index of the real training anomalies. Only needed if case=2.
        k              : int regarding the number of new labels required.
        ntimes         : int regarding the number of iterations for getting new k labels.
        case           : int getting 0 when the user's uncertainty is not present (PO), 2 when dealing with it (IO). Default=0.
        
        Returns
        ----------
        F1_real        : dict containing for each key multiple of k (k, 2*k, 3*k,...,ntimes*k) the array [tn,fp,fn,tp] obtained
                         with the real prior.
        class_priors   : array of shape (ntimes,) containing every k new labels the real class prior.
    """
    
    n = np.shape(X_train)[0]
    query_list = []
    labeled_ex = np.zeros(n, dtype=np.int)
    F1_real = {}
    class_priors = np.zeros(ntimes, dtype = float)

    for j in range(ntimes):
        
        labeled_ex, query_list = query_at_least_k_points(X_train, labeled_ex, real_anomalies, query_list, k, real_cont, case)
        
        class_priors[j] = 1 - real_cont
        
        F1_real[int(k*(j+1))] = get_tnfpfntp(X_train, labeled_ex, X_test, y_test, real_cont) 
        
    return F1_real, class_priors

def get_tnfpfntp(X_train, y_train, X_test, y_test, contamination):
    
    """ This function for evaluating the performance of SSDO gets as input the contamination factor (in addition to training and
        test sets) in order to train the classifier with all the information available at the considered step (labels and prior
        estimate) and then to compute the confusion matrix's cells values (tn fp fn tp).
        
        Parameters
        ----------
        X_train        : np.array of shape (n_samples, n_features). It is the training set.
        X_test         : np.array of shape (m_samples, n_features). It is the test set.
        y_train        : np.array of shape (n_samples,). It contains the training labels (at the considered step).
        y_test         : np.array of shape (m_samples,). It contains the test labels.
        contamination  : float regarding the REAL expected percentage of anomalies in the training set.
        
        Returns
        ----------
        a numpy array containing [tn,fp,fn,tp] computed for SSDO with y_train labels and contamination equal 1-prior.
    """
    
    prior_detector = IsolationForest(contamination = contamination, behaviour='new').fit(X_train)
    train_prior = prior_detector.decision_function(X_train) * -1
    train_prior = train_prior + abs(min(train_prior))
    detector = SSDO(k=3, alpha=2.3, unsupervised_prior='other', contamination = contamination)
    
    detector.fit(X_train, y_train, prior = train_prior)
    prediction = detector.predict(X_test, prior = train_prior)
    
    y_pred = [1 if prediction[i] == 1 else 0 for i in range(len(prediction))]
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    return np.array([tn, fp, fn, tp], dtype = int)

def query_at_least_k_points(data, labeled_ex, real_anomalies, query_list, k, tmp_cont, case):
    
    """ This function queries at least k examples in order to acquire new k labels. The new labels depend on the case and on the
        user's uncertainty. This function does 1) compute the user's uncertainty with respect to the case, 2) as long as k new
        labels are not acquired, train SSDO and query the most informative example to get labeled by the oracle (PO or IO).
        
        Parameters
        ----------
        data           : np.array of shape (n_samples, n_features). It is the data set.
        labeled_ex     : list of shape (n_samples,) assuming 1 if the example is labeled, 0 otherwise.
        real_anomalies : list of shape (n_samples,) containing the index of the real anomalies. Only needed if case=2.
        query_list     : list of shape (n_samples,) assuming 1 if the example has been queried, 0 otherwise.
        k              : int regarding the number of new labels required.
        tmp_cont       : float regarding the starting expected percentage of anomalies in the training set. Default=0.1.
        case           : int getting 0 when the user's uncertainty is not present (PO), 2 when dealing with it (IO). Default=0.
        
        Returns
        ----------
        labeled_ex     : list of shape (n_samples,) containing both the already labeled examples and the new ones.
        query_list     : list of shape (n_samples,) containing both the already queried examples and the new ones.
    """
    
    n = np.shape(data)[0]
    user_uncertainty = compute_user_uncertainty(data, real_anomalies, tmp_cont, case)
    
    prior_detector = IsolationForest(contamination = tmp_cont, behaviour='new', random_state = 331).fit(data)
    train_prior = prior_detector.decision_function(data) * -1
    train_prior = train_prior + abs(min(train_prior))
    detector = SSDO(k=3, alpha=2.3, unsupervised_prior='other', contamination = tmp_cont)

    while int(sum(labeled_ex)) < k and len(query_list) < n:
        detector.fit(data, np.negative(labeled_ex), prior = train_prior)
        score = detector.predict_proba(data, prior = train_prior, method='squash')[:, 0]
        score = [abs(x-0.5) for x in score]
        #Sort the data according to their uncertainty score
        index = sorted([[x,i] for i,x in enumerate(score) if i not in query_list], reverse = False)
        idx_query_point = index[0][1] #choose the first most uncertain example and query it
        query_list.append(idx_query_point)
        uncertainty_score = user_uncertainty[idx_query_point]
        reply = np.random.binomial(1, uncertainty_score)

        if reply: 
            labeled_ex[idx_query_point] = +1 #if the user says that it's normal, than update the label
                                     #otherwise put it in the ranking but the model will be trained on the same dataset
    return labeled_ex, query_list

