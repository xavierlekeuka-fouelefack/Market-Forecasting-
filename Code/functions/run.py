from preprocessing import *
from encoder import *
from clusters import *
from prediction import *
from backtesting import *
from sklearn.metrics import r2_score
import datetime

def run(path, index0, index, date_start, date_end_train, date_end_test, nb_clusters, longueur, echantillon, columns, n_in, latent_dim, input_dim,
         seuil, nb_iter, t_tracking, min_pip, min_element, spread, int_rate=0.1, trade_init=10):
    print("#################PREPOCESSING DATA...#####################")
    his = History(path+'marketdata.db', index, date_start, date_end_train)
    info_set, data_set = process_data(his, longueur, echantillon, t_tracking, input_dim, columns)
    # Encoding data
    encoder_model, decoder_model, model = encoder(n_in, latent_dim, input_dim)
    model_history = model.fit(data_set, data_set, validation_split=0.05, epochs=40, batch_size=130, verbose=0, shuffle=True)
    data_set_encoded = encoder_model.predict(data_set)
    #print("#################PREPOCESSING DONE#####################")


    print("#################BUILDING THE CLASSIFIER...#####################")

    Kmeans = nRaffinements(5, nb_clusters, seuil, data_set_encoded)
    Kmeans.fit(data_set_encoded)
    #print("#################CLASSIFIER IS READY#####################")
    labs = set(Kmeans.labels_)

    print("#################FILTERING CLUSTERS...#####################")
    clusters_info = cluster_info(Kmeans, np.array(info_set))
    pips = [track_cluster(clusters_info, his, i, t_tracking, inde=index0) for i in labs]

    pred_indexes_beta = predictive_index_1(Kmeans,pips, min_pip=min_pip)
    indexes_2 = [i for i in range(len(labs)) if len(clusters_info[i]) > min_element]
    pred_indexes = [ind for ind in pred_indexes_beta if np.abs(ind) in indexes_2]
    #pred_indexes = predictive_indexe_2(Kmeans,  pred_indexes_beta, min_element=min_element)
    #print("#################FILTERATION DONE#####################")

    print("#################RUNNING BACKTEST...#####################")
    history_test = History(path+'marketdata.db', [index[0]], date_end_train, date_end_test)

    info_set, data_test = process_data(history_test, longueur, echantillon,t_tracking, input_dim, columns)
    data_test_encoded = encoder_model.predict(data_test)
    equity, briefing, cluster_PnLs = back_testing(Kmeans,
                                                       t_tracking=t_tracking,
                                                       spread=spread,
                                                       testing_set=data_test_encoded,
                                                       info_set = info_set,
                                                       trade_init=trade_init,
                                                       history=history_test,
                                                       predictive_clust=pred_indexes,
                                                       longueur = longueur)
    maxdrawdown = max_drawdown(equity)
    bench_return = (history_test['open'].iloc[len(history_test) - 1] / history_test['open'].iloc[0] - 1)

    output = {}
    output["SYMBOL"] = ', '.join(index)
    output["START"] = date_start
    output["END_TRAIN"] = date_end_train
    output["END"] = date_end_test
    output["SPREAD"] = spread
    output["N_CLUSTERS"] = nb_clusters
    output["PREDICTIVE_CLUSTERS"] = len(pred_indexes)
    output["RETURN"] = (equity[-1] / equity[0] - 1) * 100
    output["MIN_PIPS"] = min_pip
    output["N_TRADE"] = len(briefing)
    output["WIN_RATE"] = len(briefing.loc[briefing["PnL"] > 0]) / len(briefing)
    output["MAX_DRAWDOWN"] = maxdrawdown
    s = datetime.datetime.strptime(date_end_train, '%Y-%m-%d %H:%M:%S')
    e = datetime.datetime.strptime(date_end_test, '%Y-%m-%d %H:%M:%S')

    output["SHARPE"] = (equity[-1] - equity[0] - (1.01 ** ((e - s).days // 365) - 1) * equity[0]) / np.array(equity).std()
    #output["PATH"] = path
    output["BENCHMARK_RETURN"] = bench_return * 100
    spr = getSpread(data_set_encoded)

    output["DATA_SPREAD"] = spr
    output["PIPS"] = ';'.join([str(pip) for pip in pips])
    sizes = [len(getClustInd(i,Kmeans.labels_)) for i in set(Kmeans.labels_)]
    output["SIZES"] = ';'.join([str(size) for size in sizes])
    spreads = []
    for i in set(Kmeans.labels_):
       clust = data_set_encoded[getClustInd(i,Kmeans.labels_)]
       spreads.append(getSpread(clust)/spr)
    output["SPREADS"] = ';'.join([str(spread) for spread in spreads])
    output["CLUSTER_PnLs"] = ''.join(['|'+';'.join([str(w) for w in l]) for l in cluster_PnLs])
    output["ECHANTILLON"] = echantillon
    output["LATENT_DIM"] = latent_dim
    output["T_TRACKING"] = t_tracking
    output["COLUMNS"] = ';'.join(columns)
    output["EQUITY"] = ";".join([str(val) for val in equity])
    return output

def norm2(l):
     return sum([x**2 for x in l])

def getSpread(l):
     avg = sum(l)/len(l)
     return np.sqrt(sum([norm2(x-avg) for x in l])/len(l))
