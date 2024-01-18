import numpy as np
import matplotlib.pyplot as plt

#A dictionary for computing PIP values
PIP = {'NONFOREX': 10000, 'FOREX': 10000} #unused


def cluster_info(classifier, info):
    """
    Classe les données (date + symbole) des patterns selon le cluster auquel ils appertiennent
    """
    labs = set(classifier.labels_)
    clust_info =[[] for i in range(len(labs))]
    for i in range(len(classifier.labels_)):
        clust_info[classifier.labels_[i]].append(info[i])
    return clust_info


def track_cluster(clusters_info, history, nb_cluster, t_tracking, inde):
    """
    Average PIP between the end of a pattern and end+t_tracking
    :param clusters_dates: end dates of patterns in respect to clusters
    :param history: pandas' dataframe
    :param nb_cluster: the number of the cluster we are tracking
    :param t_tracking:
    :return: a table of PIP values in each periode of time
    """
    pip = 0
    count = 0
    for date,symbol,id in clusters_info[nb_cluster]:
        id = int(id)
        end_price = history.iloc[id]['open']
        further_price = history.iloc[id+t_tracking]['open']
        pip += further_price/end_price - 1
        count += 1
    return pip / count


def predictive_index_1(classifier,pips,min_pip):
    """
    Finds the predictive clusters with respect only to their PIPs
    :return: a list of the indexes of the predictive clusters
    """
    indexes = []
    for i in set(classifier.labels_):
        if np.abs(pips[i])>min_pip:
            if pips[i]>0 :
                indexes.append(i)
            else:
                indexes.append(-i)
    return indexes


def predictive_indexe_2(clusters_dates, pred_ind, min_element):
    """
    Computes the predictive clusters with respect to their PIPs & their the number of elements
    :param pred_ind:
    :param min_element:
    :param clusters_dates:
    :return:
    """
    indexes_2 = [k for k,v  in clusters_dates.items() if len(v)>min_element]
    return [ind for ind in pred_ind if np.abs(ind) in indexes_2]

########################## Not USED FUNCTION IN THE MODEL###########################################


def plot_pips(clusters_dates, history, t_trucking, classifier):
    """
    Plots the PIP variation for different periods of time up to t_tracking
    """
    plt.figure(figsize=(10,6))
    plt.grid()
    for i in range(max(classifier.labels_)+1):
        pip = track_cluster(clusters_dates, history, i, t_trucking)
        plt.plot(pip)

def top3_abs_pips(classifier, pips):
    """
    Returns the index of the 3 clusters of biggest absolute PIPs
    """
    fst,snd,trd = -np.inf,-np.inf,-np.inf
    i1,i2,i3 = 0,0,0
    for i in set(classifier.labels_):
        if np.abs(pips[i])>fst:
            trd,i3 = snd,i2
            snd,i2 = fst,i1
            fst,i1 = np.abs(pips[i]),i
        else :
            if np.abs(pips[i])>snd:
                trd,i3 = snd,i2
                snd,i2 = np.abs(pips[i]),i
            else :
                if np.abs(pips[i])>trd:
                    trd,i3 = np.abs(pips[i]),i
    return i1,i2,i3