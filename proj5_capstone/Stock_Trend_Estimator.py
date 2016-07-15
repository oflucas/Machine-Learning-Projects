import argparse
from datetime import datetime
from Estimator_Core import predictStock

parser = argparse.ArgumentParser(description='Predict stocks trend in a number of days')
parser.add_argument('SymbList', metavar='Ticker Symbol List', nargs='+', help='Ticker Symbol List For Prediction')
parser.add_argument('-n', dest='forcast', type=int, default=5, help='No. of days of stock prediction (default: 5 days)')
parser.add_argument('-e', dest='endt', default='today', help='end time of model training (default: today), YYYY/MM/DD')
parser.add_argument('-s', dest='startt', default='2000/01/01', help='end time of model training (default: 2000/01/01)')
parser.add_argument('-debug', dest='debug', action='store_const', const=True, default=False, help='turn on debug mode')

args = parser.parse_args()
# print args.SymbList
# print args.forcast
# print args.startt
# print args.endt
# print args.debug

if args.endt == 'today':
	endTime = None
else:
	endTime = datetime.strptime(args.endt, '%Y/%m/%d')

startTime = datetime.strptime(args.startt, '%Y/%m/%d')

#print startTime, endTime

trend = [] 
confd = []
score = []

for symbol in args.SymbList:
	tr, cf, sc = predictStock(symbol, forcast = args.forcast, endTime = endTime, startTime = startTime, debug = args.debug)
	trend.append(tr)
	confd.append(cf)
	score.append(sc)

nn = len(args.SymbList)
print "\n\n\n\n///////////////////////////////////////////////////"
print "         PREDICTION SUMMARY"
print "///////////////////////////////////////////////////"
print "FORCAST [days]:", args.forcast
print "END TIME:", endTime
print "\nSYMBOL\t\tTREND\t\tMODEL FA SCORE\t\tFALL/RISE CONFIDENCE"
for i in range(nn):
	print args.SymbList[i], '\t\t', trend[i], '\t\t', score[i], '\t\t', confd[i]
