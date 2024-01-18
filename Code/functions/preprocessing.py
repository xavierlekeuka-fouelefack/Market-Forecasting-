import numpy as np
import sqlite3
import pandas as pd
from sklearn import preprocessing


def History(path_db, symbol, start, end):
    """
    Return a pandas' dataframe containing the index we are looking for
    :param path_db: system path where the data base file is stored
    :param symbol: the symbols of the indices
    :param start: starting date with the format "%Y-%M-%D %H-%M-%S"
    :param end: ending date with the format "%Y-%M-%D %H-%M-%S"
    :return: a pandas' dataframe
    """
    connection = sqlite3.connect(path_db)
    df = []
    for i in range(0,len(symbol)):
        query = f"""SELECT * FROM {symbol[i]}"""
        indexdata = pd.read_sql(query, connection)
        indexdata['symbol'] = symbol[i] 
        # The 'symbol' value is necessary to uniquely identify a given market price
        
        index_start = indexdata.index[(indexdata['date']==start)]
        if len(index_start)==0:
            index_start = 0
        else:
            index_start = index_start[0]
        index_end = indexdata.index[(indexdata['date']==end)]
        if len(index_end)==0:
            index_end = -1
        else:
            index_end = index_end[0]
        indexdata = indexdata.iloc[index_start:index_end,:]
        # Automates the selection of the timeframe

        df.append(indexdata)
    df = pd.concat(df)
    return df.reset_index(drop=True)


def process_data(history, longueur, echantillon, t_tracking, input_dim, columns):
    """
    Do the preprocessing step: normalize data & make it in a series of longueur-points
    :param history: dataframe
    :param longueur: length of the series of points
    :param echantillon: the number of samples taken
    :param columns: columns of the database taken into account
    :param input_dim
    :return: dates_set & data_set
    """
    # data = np.array([data_set[i:i+longueur:].tolist() for i in range(0,len(his)-longueur, longueur//echantillon)])
    total_info = []
    symbols = set(history['symbol'])
    total_data = []
    cur_id = 0
    for s in symbols:
        his = history.loc[history['symbol']==s] 
        # Choosing only the part of our data regarding each specific index

        info = [(his['date'].iloc[i + longueur - 1], s, cur_id+i+longueur-1) for i in range(0, len(his) - longueur - t_tracking, longueur)]
        # info contains the end dates, symbol and end id of the patterns of the current symbol (in order)
        cur_id+=his.shape[0]

        data = []
        L = len(his[columns[0]].to_list())
        for i in range(0, L - longueur - t_tracking, longueur):
            lComp = []
            h = his.iloc[i:i + longueur]
            for column in columns: 
                l = h[column].to_list()
                lComp.append(preprocessing.scale(np.array(l)))
            data.append((np.array(lComp)).flatten())
        data = np.array(data)
        # Finished partitioning the data into patterns

        data = data.reshape((len(data), -1, input_dim))
        total_info.extend(info)
        total_data.extend(data)
    total_data = np.array(total_data)
    return total_info, total_data