import numpy as np
import math
from bitarray import bitarray
import time
import heapq

import argparse



def run_from_elsewhere(data, labels, k=5, its=100):
    
    _max_bepp = 5
    _delta = None
    _nbIts = its
    _maxSplits = 500
    _promis = False
    _minT = 10
    
    labels = bitarray(list(labels==1)) #from [0],[1],[1],[0] to 0110.
    folds = np.random.randint(5, size=len(data)) #random assignment of fold -> not equally distributed
    
    ti = time.time()    
    (c_estimate, c_its_estimates) = tice(data, labels, _max_bepp, folds, _delta, nbIterations=_nbIts,\
                                         maxSplits=_maxSplits, useMostPromisingOnly=_promis,minT=_minT)
    ti = time.time() - ti
    
    
    alpha = 1.0
    if c_estimate > 0:
        pos = float(labels.count())/c_estimate
        tot = len(data)
        #print("POS:",pos,"and TOT:",tot)
        alpha=max(0.0,min(1.0,pos/tot))

    return c_estimate, alpha


def pick_delta(T):
    return max(0.025, 1/(1+0.004*T))
  
def low_c(data, label, delta, minT,c=0.1):
    T = float(data.count())
    if T<minT:
        return 0.0
    L = float((data&label).count())
    clow = L/T - math.sqrt(c*(1-c)*(1-delta)/(delta*T))
    #print("L/T = ", L/T,"correction =", math.sqrt(c*(1-c)*(1-delta)/(delta*T)))
    return clow


def max_bepp(k):
    def fun(counts):
        #listcount = list(counts)
        maximum = max(map(lambda TP : (0 if TP[0]==0 else float(TP[1])/(TP[0]+5)), counts)) # <------ I PUT 5 INSTEAD OF K!
        #print("Lambda_Function:", maximum)
        #print(listcount)
        return maximum
    return fun


def generate_folds(folds):
    for fold in range(max(folds)+1):
        tree_train = bitarray(list(folds==fold))
        estimate = ~tree_train 
        yield (tree_train, estimate)

        
def tice(data, labels, k, folds, delta = None, nbIterations = 2, maxSplits = 500, useMostPromisingOnly = False, minT = 10, ):
    c_its_ests = []
    c_estimate = 0.1
    j = 0 #amount of useless corrections
    k = 0 #amount of nice corrections
    
    for it in range(nbIterations):
        
        c_estimates = []
    
        global c_cur_best # global so that it can be used for optimizing queue.
        
        for (tree_train, estimate) in generate_folds(folds): #Loop of 5 iterations as Default value -> it is like a cross-validation
            #print("Tree_train:", len(tree_train),"estimate:",len(estimate)) #->they are bitarray, i.e. 0010100, of length 2000 (len(data))
            c_cur_best = low_c(estimate, labels, 1.0, minT, c = c_estimate) #first estimation of c = L/T
            #c_cur_best = float(sum(labels))/data.shape[0]
            cur_delta = delta if delta else pick_delta(estimate.count())
            if useMostPromisingOnly:
                #print("Enter here?") -> no!
                c_tree_best=0.0
                most_promising = estimate
                for tree_subset, estimate_subset in subsetsThroughDT(data, tree_train, estimate, labels,\
                                                                     splitCrit=max_bepp(k), minExamples=minT,\
                                                                     maxSplits=maxSplits, c_prior=c_estimate,\
                                                                     delta=cur_delta):
                    tree_est_here = low_c(tree_subset, labels, cur_delta, 1, c = c_estimate)
                    if tree_est_here > c_tree_best:
                        c_tree_best = tree_est_here
                        most_promising = estimate_subset
                
                c_estimates.append(max(c_cur_best, low_c(most_promising, labels, cur_delta, minT, c = c_estimate)))
                
            else:
                for tree_subset, estimate_subset in subsetsThroughDT(data, tree_train, estimate, labels, \
                                                                     splitCrit=max_bepp(k), minExamples=minT,\
                                                                     maxSplits=maxSplits, c_prior=c_estimate,\
                                                                     delta=cur_delta):
                    #print("Tree train:", tree_subset, "\nEstimate:", estimate_subset)
                    #print("SUBSETS:",len(tree_subset),len(estimate_subset))#->tree_subset and estimate_subset lenght = 2000 (len(data))
                    est_here = low_c(estimate_subset, labels, cur_delta, minT, c = c_estimate)
                    #print("Labels:", len(labels), sum(labels)) # -> always labels length = 2000 and  num_pos_labeled = n active points
                    if c_cur_best > est_here:
                        #print("Useless correction!")
                        j+=1
                    else:
                        #print("Nice correction!")
                        k+=1
                        
                    #print("Difference between c_best and c_low:", c_cur_best - est_here)
                    c_cur_best = max(c_cur_best, est_here)
                    #print("Low c:", c_cur_best)
                c_estimates.append(c_cur_best)

        #print("Vector of c:", c_estimates)       
        c_estimate = sum(c_estimates)/float(len(c_estimates))
        #c_estimate = max(c_estimates) #-> it doesn't work -> too high
        c_its_ests.append(c_estimates)
        #print("c_its_ests:", c_its_ests) #-> c_its_ests is useless (never used)
    #print("Number of useless corrections:",j)
    #print("Number of nice corrections:",k)
    #print("-------------------------------------------------")
    return c_estimate, c_its_ests

def subsetsThroughDT(data, tree_train, estimate, labels, splitCrit=max_bepp(5), minExamples=10, maxSplits=500,\
                     c_prior=0.1, delta=0.0):
    
  # This learns a decision tree and updates the label frequency lower bound for every tried split.
  # It splits every variable into 4 pieces: [0,.25[ , [.25, .5[ , [.5,.75[ , [.75,1]
  # The input data is expected to have only binary or continues variables with values between 0 and 1. 
  # To achieve this, the multivalued variables should be binarized and the continuous variables should be normalized
  
  # Max: Return all the subsets encountered
  
    all_data=tree_train|estimate
    
    borders=[.25, .5, .75]
    
    def makeSubsets(a):
        subsets = []
        options=bitarray(all_data)
        for b in borders:
            X_cond = bitarray(list((data[:,a]<b)))&options
            options&=~X_cond
            subsets.append(X_cond)
        subsets.append(options)
        return subsets
    
    conditionSets = [makeSubsets(a) for a in range(data.shape[1])] # -> data.shape[1] = 2 columns
    #print("Shape data:",data.shape[1])
    priorityq = []
    heapq.heappush(priorityq,(-low_c(tree_train, labels, delta, 0, c=c_prior),-(tree_train&labels).count(),\
                              tree_train, estimate, set(range(data.shape[1])), 0))
    yield (tree_train, estimate)
    
    n=0
    minimumLabeled = 1
    while n<maxSplits and len(priorityq)>0:
        n+=1
        (ppos, neg_lab_count, subset_train, subset_estimate, available, depth) = heapq.heappop(priorityq)
        lab_count= -neg_lab_count
        
        best_a=-1
        best_score=-1
        best_subsets_train=[]
        best_subsets_estimate=[]
        best_subsets_estimate_fake_split = []
        best_lab_counts=[]
        uselessAs=set()

        for a in available:
            subsets_train=map(lambda X_cond:X_cond&subset_train, conditionSets[a])
            subsets_train_score = map(lambda X_cond:X_cond&subset_train, conditionSets[a])
            subsets_estimate=map(lambda X_cond:X_cond&subset_train, conditionSets[a])
            subsets_estimate_lab_counts = map(lambda X_cond:X_cond&subset_train, conditionSets[a])
            estimate_lab_counts = map(lambda subset: (subset&labels).count(), subsets_estimate_lab_counts)
            if max(estimate_lab_counts) < minimumLabeled:
                uselessAs.add(a)
            else:
                score = splitCrit(map(lambda subsub: (subsub.count(), (subsub&labels).count()), subsets_train_score))
                #print("Score:", score, "Best score:", best_score, "subsets_train:",list(subsets_train))
                if score>best_score:
                    #print("Update best subset!")
                    best_score=score
                    best_a=a
                    best_subsets_train=subsets_train
                    best_subsets_estimate=subsets_estimate
                    best_subsets_estimate_fake_split = map(lambda X_cond:X_cond&subset_train, conditionSets[a])
                    best_lab_counts = estimate_lab_counts

        fake_split = len(list(filter(lambda subset: subset.count()>0, best_subsets_estimate_fake_split)))==1
        #print("Fake split?", fake_split) #it is always false

        if best_score > 0 and not fake_split:
            newAvailable = available-set([best_a])-uselessAs
            for subsub_train,subsub_estimate in zip(best_subsets_train, best_subsets_estimate):
                yield (subsub_train,subsub_estimate)
            minimumLabeled = c_prior*(1-c_prior)*(1-delta)/(delta*(1-c_cur_best)**2)

            for (subsub_lab_count, subsub_train, subsub_estimate) in zip(best_lab_counts, best_subsets_train,\
                                                                         best_subsets_estimate):
                if subsub_lab_count>minimumLabeled:
                    total = subsub_train.count()
                    if total>minExamples: #stop criterion: minimum size for splitting 
                        train_lab_count = (subsub_train&labels).count()
                        if lab_count!=0 and lab_count!=total: #stop criterion: purity
                            heapq.heappush(priorityq,(-low_c(subsub_train, labels, delta, 0, c=c_prior), \
                                                      -train_lab_count, subsub_train, subsub_estimate, newAvailable,\
                                                      depth+1))