import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import matplotlib
from SSAD_functions import *
from sklearn.neighbors import KernelDensity
from multiprocessing import Pool, freeze_support, cpu_count
from anomatools.models import SSDO
from sklearn.ensemble import IsolationForest
from libc.math cimport log10, exp

#-----------------------------------------------------------------------------------------------------------------------
cpdef CAPe(data, labeled_ex, list query_list = [], int k = 5, real_anomalies = [], float tmp_cont = 0.1,\
             float mean_prob_term = -1.7, int case = 0, n_jobs = cpu_count()):
    
    """ Estimate the class prior according to the model CAPe provided in the paper. First, this method creates the global
        ranking, where higher position means higher probability to be queried. Second, it estimates the propensity scores
        using the counting techniques as explained in the paper. Third, from the propensity scores the class prior is computed. 
        
        Parameters
        ----------
        data           : np.array of shape (n_samples, n_features). It is the input data set.
        labeled_ex     : list of shape (n_samples,) assuming 1 if the example is (already) labeled, 0 otherwise.
        query_list     : list of shape (n_samples,) assuming 1 if the example has been queried, 0 otherwise.
        k              : int regarding the number of new labels required. Default = 5.
        real_anomalies : list of shape (n_samples,) containing the index of the real anomalies. Only needed when case = 2.
        tmp_cont       : float regarding the temporary expected percentage of anomalies in the training set. Default = 0.1.
        mean_prob_term : float with the log10 mean probability to draw an example from the distribution mu. Default = -1.7.
        case           : int assuming 0 when the user's uncertainty is not present, 2 when dealing with it. Default = 0.
        n_jobs         : int regarding the number of jobs to run in parallel. Default = maximum number of jobs.
        
        Returns
        -------
        prior          : float with the estimated class prior value.
        labeled_ex     : updated list with the new labeled examples after the querying phase based on the AL strategy.
        query_list     : updated list with the new queried examples after the querying phase based on the AL strategy.
    """

    cdef int N = data.shape[0], h, i, idx, n, eightpct, fifteenpct, steps, n_norm, total_labels
    cdef float prior
    cdef list user_uncertainty, all_args, gr
    cdef object pool = Pool(n_jobs)
    cdef dict old_to_new = {}
    if mean_prob_term < -1.7:
        mean_prob_term = -1.7
        
    propensity_score = np.zeros(N, dtype=np.float)
    gr, labeled_ex, query_list, normal_prob, user_uncertainty = make_a_global_ranking(data, labeled_ex, query_list, k,\
                                                                                      real_anomalies, tmp_cont, case)
    n_norm = len(gr)
    
    for i from 0 <= i < n_norm by 1:
        idx = gr[i]
        old_to_new[idx] = i
        
    eightpct = int(max(0.08*n_norm,1)) #Here you can change the granularity of the subsets.
    fifteenpct = int(0.15*n_norm)

    steps = int(min(eightpct,5))

    cardinalities = np.arange(eightpct,min(fifteenpct, eightpct+10*steps)+steps, steps)

    total_labels = np.count_nonzero(labeled_ex)
    all_args = [(propensity_score, n, gr, old_to_new, user_uncertainty, mean_prob_term, total_labels, tmp_cont, case)\
                for n in cardinalities]
    
    propensity_score = pool.map(compute_prop_score, all_args)
    propensity_score = [sum(t) for t in zip(*propensity_score)] 
    #train the model and compute the prior
    prior = from_propensity_score_to_class_prior(propensity_score, normal_prob, labeled_ex)

    return prior, labeled_ex, query_list

#-----------------------------------------------------------------------------------------------------------------------
cpdef make_a_global_ranking(data, labeled_ex, list query_list = [], int k = 5, real_anomalies = [],\
                            float tmp_cont = 0.1, int case = 0):
    
    """ Make a global ranking according to the Uncertainty Sampling (AL strategy) and SSDO (Anomaly Detector).
        The higher is the position in the ranking, the more informative is the point (and then the higher will 
        be probably the prop score).
        Assumption: we can reasonably assume that the ranking is preserved for the examples in any subset of the dataset.
        
        Parameters
        ----------
        data           : np.array of shape (n_samples, n_features). It is the input data set.
        labeled_ex     : list of shape (n_samples,) assuming 1 if the example is labeled, 0 otherwise.
        query_list     : list of shape (n_samples,) assuming 1 if the example has been queried, 0 otherwise.
        k              : int regarding the number of new labels required.
        real_anomalies : list of shape (n_samples,) containing the index of the real anomalies. Only needed if case = 2.
        tmp_cont       : float regarding the temporary expected percentage of anomalies in the training set.
        case           : int assuming 0 when the user's uncertainty is not present, 2 when dealing with it.
        
        Returns
        -------
        gr          : list of shape (n_samples-n_anom,) containing the global ranking or positive examples (the indexes).
        labeled_ex  : list of shape (n_samples,) with the new labeled examples (the input amount + k).
        query_list  : list of shape (n_samples,) with the new queried examples.
        pred_anom   : list of shape (n_samples,) with the expected anomalies according to the classifier.
        normal_prob : list of shape (n_samples,) with the probabilities that the examples are positive (normal).
    """

    cdef int N = np.shape(data)[0], starting_labels = sum(labeled_ex), lenquerylist = len(query_list), n_labels, i
    cdef int one_new_label, number_queried_points, idx_query_point
    cdef list user_uncertainty, Dy, pred_anom, index, prediction, ranking, not_ranked_points
    cdef object prior_detector, detector
    cdef long[:] already_queried_idxs = np.where(labeled_ex == 1)[0]
    cdef double[:] train_prior
    
    if not set(already_queried_idxs).issubset(set(query_list)):
        print('Copying existing labels into query list.')
        query_list = already_queried_idxs
    
    user_uncertainty = compute_user_uncertainty(data, real_anomalies, tmp_cont, case)
    #train the model:
    prior_detector = IsolationForest(contamination = tmp_cont, behaviour='new', random_state = 331).fit(data)
    train_prior = prior_detector.decision_function(data) * -1
    train_prior = np.add(train_prior, abs(min(train_prior)))
    detector = SSDO(k=3, alpha=2.3, unsupervised_prior='other', contamination = tmp_cont)
    n_labels = starting_labels
    #Let's make t times the ranking labeling one point at the time in order to have a more precise ranking!
    while n_labels < starting_labels + k and lenquerylist < N:
        detector.fit(data, np.negative(labeled_ex), prior = train_prior)
        score = np.absolute(np.subtract(detector.predict_proba(data, prior = train_prior, method='unify')[:, 0], 0.5))
        #Sort the data according to their uncertainty score
        index = sorted([[x,i] for i,x in enumerate(score) if i not in query_list], reverse = False)
        
        one_new_label = 0
        number_queried_points = 0
        while one_new_label == 0 and lenquerylist < N:
            idx_query_point = index[number_queried_points][1] #choose the first most uncertain example and query it
            query_list.append(idx_query_point)
            lenquerylist = len(query_list)

            if label_or_dont_know(idx_query_point, user_uncertainty):
                labeled_ex[idx_query_point] = 1
                one_new_label = 1
                n_labels = n_labels+1
            else:
                number_queried_points = number_queried_points +1
    
    detector = SSDO(k=3, alpha=2.3, unsupervised_prior='other', contamination = tmp_cont)
    detector.fit(data, np.negative(labeled_ex), prior = train_prior)
    prediction = list(detector.predict(data, prior = train_prior))
    pred_anom = [i for i in range(N) if prediction[i] == +1]

    cdef double [:] normal_prob = detector.predict_proba(data, prior=train_prior, method='unify')[:, 0]
    
    if lenquerylist >= N:
        print('Queried all the examples in the data set. The user should still label', N-k,'points.')
        gr = [x for x in query_list if x not in pred_anom]
        return gr, labeled_ex, query_list, pred_anom, normal_prob
    
    ranking = [x for x in query_list]
    not_ranked_points = [t for [x,t] in index[1:]]
    ranking.extend(not_ranked_points)
    gr = [x for x in ranking if x not in pred_anom]
    user_uncertainty = compute_user_uncertainty(data, pred_anom, tmp_cont, case)
    #print('Global Ranking done.')
    return gr, labeled_ex, query_list, normal_prob, user_uncertainty

#-------------------------------------------------------------------------------------------------
cpdef prop_score(propensity_score, int n, list gr, dict old_to_new, list user_uncertainty, float mean_prob_term, int k = 5,\
                 float tmp_cont = 0.1, int case = 0):
    
    """ Compute the propensity score for each example according to its position in the global ranking. 
        According to the case we consider the Perfect Oracle (case 0) or the Imperfect Oracle (case 2).
        Assumptions: We can reasonably approximate the binomial factors with log10 and than compute the exponentials.
        In addition, we consider the predicted anomalies as real anomalies and we don't compute the propensity score for them.
        
        Parameters
        ----------
        propensity_score : np.array of shape (n_samples,). It is an empty array at the beginning.
        n                : int representing the cardinality of the theoretically drawn subset of the data set.
        gr               : list of unfixed shape (n_samples-n_anomalies,) containing the GR with no anomalies inside.
        old_to_new       : dictionary that associates the position in the ranking to the previous index.
        user_uncertainty : list of shape (n_samples,) containing the probability that the example is positive for each data points.
        mean_prob_term   : float with the mean probability to draw a single example from the distribution mu.
        k                : int regarding the number of new labels required.
        tmp_cont         : float regarding the temporary expected percentage of anomalies in the training set.
        case             : int assuming 0 when the user's uncertainty is not present, 2 when dealing with it.
        
        Returns
        -------
        propensity_score : list of shape (n_samples,) containing the propensity score for each example. It is 0 for anomalies.
    """
    
    cdef int N = len(gr), idx, start_loop = 0, end_loop, new_idx
    cdef float prob_select_apoint, tot_cases, mean_user_uncertainty, prob_be_queried_in_topk, prob_be_queried_not_topk
    
    tot_cases = discretize_a_binomial(N-1,n-1) #int(n*1.7)
    
    mean_prob_term = n*mean_prob_term
    mean_user_uncertainty = np.mean(user_uncertainty)
    
    for idx in gr:
        new_idx = old_to_new[idx]
        if n <= k:
            prob_be_queried_in_topk = 1
            prob_be_queried_not_topk = 0
        elif new_idx < k:
            prob_be_queried_in_topk = 1
            prob_be_queried_not_topk = 0
        elif case == 0:
            if n + new_idx - N > 0:
                start_loop = n + new_idx - N
            end_loop = k
            prob_be_queried_in_topk = compute_prob_be_queried_in_topk(new_idx, N, n, tot_cases, start_loop, end_loop, k)
            if prob_be_queried_in_topk > 1:
                prob_be_queried_in_topk = 1
            prob_be_queried_not_topk = 0
            
        elif case == 2:
            if n + new_idx - N > 0:
                start_loop = n + new_idx - N
            end_loop = k
            prob_be_queried_in_topk = compute_prob_be_queried_in_topk(new_idx, N, n, tot_cases, start_loop, end_loop, k)
            if prob_be_queried_in_topk > 1:
                prob_be_queried_in_topk = 1
                
            if n - N + new_idx > k:
                start_loop = n - N + new_idx
            else:
                start_loop = k
            if new_idx < n-1:
                end_loop = new_idx + 1
            else:
                end_loop = n
                
            prob_be_queried_not_topk = compute_prob_be_queried_not_topk(new_idx, N, n, tot_cases, start_loop, end_loop,\
                                                                        k, mean_user_uncertainty)
        else:
            raise ValueError(case, 'The case must be 0 or 2.')
        propensity_score[idx] = 10**(tot_cases+mean_prob_term) * user_uncertainty[idx]*\
                                (prob_be_queried_in_topk + (1-prob_be_queried_in_topk)*prob_be_queried_not_topk)
    return propensity_score

#-------------------------------------------------------------------------------------------------    
def compute_prop_score(args): 
    #It's useful just for notation
    return prop_score(*args)

#-------------------------------------------------------------------------------------------------
cdef double from_propensity_score_to_class_prior(propensity_score, normal_prob, labeled_ex):
    
    """ Estimate the class prior from propensity scores according to the provided formula in Section 3.1.
        
        Parameters
        ----------
        propensity_score : np.array of shape (n_samples,) containing the propensity scores.
        normal_prob      : list of shape (n_samples,) with the probabilities that the examples are positive (normal).
        labeled_ex       : list of shape (n_samples,) assuming 1 if the example is labeled, 0 otherwise.
        
        Returns
        -------
        prior : float with the estimated class prior value.
    """
    
    cdef int N = len(labeled_ex), k = reduce(lambda x,y: x+y, labeled_ex)
    cdef float labeled_contribution, unlabeled_contribution, prior, norm_prop_score
    
    norm_prop_score = reduce(lambda x,y: x+y, propensity_score)
    if norm_prop_score >= 0:
        propensity_score = np.divide(propensity_score, norm_prop_score) #restrict the probability to the available dataset
    labeled_contribution = float(k)/float(N)
    unlabeled_contribution = np.mean(np.divide(np.multiply(np.multiply(1-labeled_ex,normal_prob), 1-propensity_score),\
                                       1 - np.multiply(normal_prob,propensity_score)))
    prior = labeled_contribution + unlabeled_contribution
    return prior

#-----------------------------------------------------------------------------------------------------------------------
cdef int label_or_dont_know(int index_point, list user_uncertainty):
    
    """ Label the queried example according to the user's uncertainty. We flip a coin with probability p equal to the 
        user's uncertainty for the example: if the user is secure that the point is normal it returns 1, otherwise 0.
        
        Parameters
        ----------
        index_point      : int representing the index of the queried example.
        user_uncertainty : list of shape (n_samples,) containing the user's uncertainty for all the examples.
        
        Returns
        -------
        reply : int assuming 1 if the user is secure of its positive class, 0 if the user doesn't know.
    """
    
    cdef float uncertainty_score = user_uncertainty[index_point]
    cdef int reply = np.random.binomial(1, uncertainty_score)
    return reply

#-------------------------------------------------------------------------------------------------
cpdef list compute_user_uncertainty(data, anomalies, float tmp_cont = 0.1, int case = 0):
    
    """ Compute the user's uncertainty according to the provided case (0 or 2).
        
        Parameters
        ----------
        data           : np.array of shape (n_samples, n_features). It is the input data set.
        real_anomalies : list of shape (n_samples,) containing the index of the real anomalies. Only needed if case = 2.
        tmp_cont       : float regarding the temporary expected percentage of anomalies in the training set.
        case           : int assuming 0 when the user's uncertainty is not present, 2 when dealing with it.
        
        Returns
        -------
        user_uncertainty : list of shape (n_samples,) containing the probability that the example is positive for each 
        data point.
    """
    
    cdef int N = np.shape(data)[0], i, j, flag, n_anom = len(anomalies)
    cdef list user_uncertainty = N*[0], dmu = N*[0], tmp_score 
    cdef float gamma
    cdef double score
    if case == 0:
        for i from 0 <= i < N by 1:
            flag = 0
            for 0 <= j < n_anom:
                if i == anomalies[j]:
                    flag = 1
                    break;
            user_uncertainty[i] = 1 - flag
    elif case == 2:
        ker = KernelDensity().fit(data)
        for i from 0 <= i < N by 1:
            score = ker.score(data[i:i+1])
            dmu[i] = exp(score)
        tmp_score = sorted(dmu, reverse = False)
        gamma = tmp_score[int(tmp_cont*N)]
        user_uncertainty = [1 - 2**(-(x/gamma)**2) for x in dmu]
    return user_uncertainty

#-------------------------------------------------------------------------------------------------
cdef double compute_prob_be_queried_in_topk(int idx, int N, int n, double tot_cases, int start_loop, int end_loop, int k):
    
    """ Compute the probability that an example with index = idx will be queried, given that the data set has size N,
        the drawn subsamples have size n and that the example ~IS~ in the top k of the Global Ranking.
    """

    cdef double tmp_sum = 0
    cdef int t
    for t from start_loop <= t < k by 1:
        tmp_sum += 10**(discretize_a_binomial(idx, t) + discretize_a_binomial(N-idx-1, n-t-1) - tot_cases)
    return tmp_sum

#-------------------------------------------------------------------------------------------------
cdef double compute_prob_be_queried_not_topk(int idx, int N, int n, double tot_cases, int start_loop, int end_loop,\
                                             int k, double mean_us_un):
    
    """ Compute the probability that an example with index = idx will be queried, given that the data set has size N,
        the drawn subsamples have size n and that the example is ~NOT~ in the top k of the Global Ranking.
    """
    
    cdef double tmp_sum = 0
    cdef double tmp_sum_s
    cdef int t, s
    for t from start_loop <= t < end_loop by 1:
        tmp_sum_s = 0
        for s from t-k+1 <= s < t by 1:
            tmp_sum_s += (1-mean_us_un)**s*mean_us_un**(t-s-1)
        tmp_sum += 10**(discretize_a_binomial(idx, t) + discretize_a_binomial(N-idx-1, n-t-1) - tot_cases) * tmp_sum_s
    return tmp_sum

#-------------------------------------------------------------------------------------------------
cdef double discretize_a_binomial(int up, int down):
    
    """ Compute the log10 of binomial. It's useful for approximating extremely high or low numbers.
        
        Parameters
        ----------
        up   : int, the upper element of a binomial.
        down : int, the lower element of a binomial.
        
        Returns
        -------
        num - den_1 - den_2 : float containing the log10 of the binomial.
    """
    
    cdef float log10bin
    try:
        log10bin = logfactorial(up) - logfactorial(down) - logfactorial(up-down)
    except:
        raise ValueError('At least one between',up,',',down,'and',up-down,'is not an integer > 0')
    return log10bin

#-------------------------------------------------------------------------------------------------
cdef double logfactorial(int x):
    
    """ Compute the logarithm of the factorial of x.
    """

    cdef double y = 0.0
    cdef int i
    for i from 2 <= i < x+1 by 1:
        y += log10(i)
    return y
