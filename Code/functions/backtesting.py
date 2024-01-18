import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def back_testing(classifier, t_tracking, spread, testing_set, info_set, trade_init, history, predictive_clust,longueur):
    """
    Runs a backtestinf of the classification model during the testing period
    :param classifier: the clustering model
    :param t_tracking: the time of tracking if a cluster once it's recognized
    :param testing_set: the dataset used for testing(normalized and encoded)
    :param indo_set : metadata of the 
    :param trade_init: number of initial trades
    :param history: pandas' dataframe containing the period of testing
    :param predictive_clust: the numbers of predictive clusters
    :return: variation of equity, leverage buy & sell, sortino ratio and a pandas' dataframe 'briefing' containing the orders made
    """
    labs = set(classifier.labels_)
    N = testing_set.shape[0]*longueur
    briefing = pd.DataFrame(columns = ['date', 'position', 'buy price', 'sell price', 'PnL' ])
    equity =[history.iloc[0]['open'] * trade_init]*N
    cluster_PnLs = [[] for i in labs]
    for i in range(len(testing_set)):
        date,symbol,cluster_end = info_set[i]
        nb_cluster = classifier.predict(np.array(testing_set[i]).reshape(1,-1))
        if nb_cluster in predictive_clust:
            pip = history.iloc[cluster_end+t_tracking]['open']/history.iloc[cluster_end]['open'] -1
            time = (i+1)*longueur-1
            PnL = equity[i*longueur]*(pip-spread)
            equity[i*longueur] += PnL
            equity = equity[:time+1] + [e + PnL for e in equity[time+1:]]
            briefing.loc[len(briefing)] = [history.iloc[cluster_end]['date'], 'buy', history.iloc[cluster_end]['open'], history.iloc[cluster_end+t_tracking]['close'], PnL]
            cluster_PnLs[int(nb_cluster)].append(PnL)
        elif -nb_cluster in predictive_clust:
            pip = history.iloc[cluster_end+t_tracking]['open']/history.iloc[cluster_end]['open'] -1
            time = (i+1)*longueur-1
            PnL = equity[i*longueur]*(-pip-spread)
            equity[i*longueur] += PnL
            equity = equity[:time+1] + [e + PnL for e in equity[time+1:]]
            briefing.loc[len(briefing)] = [history['date'].iloc[cluster_end], 'sell', history['open'].iloc[cluster_end], history['close'].iloc[cluster_end+t_tracking], PnL]
            cluster_PnLs[int(-nb_cluster)].append(PnL)

    return equity, briefing, cluster_PnLs

def max_drawdown(equity:list):
    """
    Mesure le max drawdown, la perte maximale encaissée, du portefeuille
    """
    maxi = 0
    j=1
    while j<len(equity):
        if equity[j]<equity[j-1]:
            i = j
            while  i<len(equity) and equity[i] < equity[i-1]:
                i+=1
            maxi = max(maxi, (equity[j-1]-equity[i-1])/equity[j-1])
            j = i
        else:
            j+=1
    return maxi


########################## Not USED FUNCTION IN THE MODEL###########################################


def plot_backtest(equity, leverage_b, leverage_s):
    """
    Plots the backtesting results
    """
    plt.subplot(1, 2, 1)
    plt.plot(equity, color='r')
    plt.xlabel("time")
    plt.ylabel("Equity")
    plt.grid()
    plt.title('Equity overtime')

    plt.subplot(1, 2, 2)
    leverage = [sum(x) for x in zip(leverage_b, leverage_s)]
    plt.plot(leverage, color='r')
    plt.xlabel("time")
    plt.ylabel("Leverage")
    plt.grid()
    plt.title('Leverage overtime')

    plt.savefig("Backtesting_plot")

