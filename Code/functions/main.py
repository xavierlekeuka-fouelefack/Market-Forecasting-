import sys
sys.path.append('./functions')

from run import *
from pdf import *
import sqlite3

nb_iter = 20
args = {}
args["path"] =".\\"

def draw(results):
    plt.scatter([line["PREDICTIVE_CLUSTERS"] for line in results],[line["RETURN"] for line in results], color = 'green')
    plt.ylabel("Return")
    plt.xlabel("Nb of predictive clusters")
    plt.grid()
    plt.savefig(args["path"] + "\\RETvsCLUSTERS")
    plt.close()

args["index0"] = 'FOREX'
args['index'] = ['EURUSD']#, 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCNH', 'USDCHF', 'USDHKD']


# List of all Forex symbols :
# ['EURUSD','AUDCHF','AUDJPY','AUDNZD','AUDUSD','CADCHF','CADJPY','CHFJPY','EURAUD','EURCAD','EURCHF','EURGBP','EURJPY','EURNOK','EURNZD','EURSEK','EURTRY','EURUSD','GBPAUD','GBPCAD','GBPCHF','GBPJPY','GBPNZD','GBPUSD','NZDCAD','NZDCHF','NZDJPY','NZDUSD','USDCAD','USDCHF','USDCNH','USDHKD','USDILS','USDJPY','USDMXN','USDNOK','USDOLLAR','USDSEK','USDTRY','USDZAR','XAGUSD','ZARGPY']
# A lot of them are repetitive

args["date_start"] = '2010-01-03 22:00:00'
args["date_end_train"] = '2016-01-06 07:00:00'
args["date_end_test"] = '2017-01-06 07:00:00'
args["nb_clusters"] = 500
args["longueur"] = 8
args["echantillon"] = args["longueur"]

args["columns"] = ['open'] #,'high','low','volume']

# Our model can use all columns except 'close', which equals 'open' of the next entry
# That process means that the real dimension manipulated by the preprocesser is len(columns)*echantillon
args["n_in"] = len(args["columns"])*args["echantillon"]


args["latent_dim"] = 3
args["input_dim"] = 1

args["seuil"]=0.4
args["nb_iter"] = 25

args["t_tracking"]=20
args["min_pip"]=0.0
args["min_element"]=2
args["spread"]=0.0
args["int_rate"]=0.1
args["trade_init"]=10

output = run(**args)

# Inserts the run data in the database
db = sqlite3.connect('run_data.db')
cur = db.cursor()
columns = ', '.join(output.keys())
placeholders = ':'+', :'.join(output.keys())
query = 'INSERT INTO X20 (%s) VALUES (%s)' % (columns, placeholders)
cur.execute(query, output)
db.commit()

# Generates the report
BACKTEST_REPORT(output)






