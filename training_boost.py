import xgboost as xgb
from sklearn.linear_model import LinearRegression
import json
import numpy as np
missing = 74122.387

with open('data/processed_xgb.json') as f:
    t = json.loads( f.read() )
    #clf.fit(t['x'], t['y'] )
for ind in range( len(t['y']) ):
    t['y'][ind] = (t['y'][ind] -1)/7
dtrain = xgb.DMatrix( t['x'], label = t['y'], missing = missing )

param = {'bst:max_depth':6, 'bst:eta':0.15, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 8
plst = list( param.items() )

num_round = 450

test = []

dtest = xgb.DMatrix( t['x'], missing = missing  )
    
bst = xgb.train( plst, dtrain, num_round)
bst.save_model('trained_for_linear.model')
res = list( bst.predict(data = dtest) )

LR = LinearRegression()
LR.fit( np.array(res)[:,None] , t['y'])    
with open('data/test_data_xgb.json') as f:
    test = json.loads( f.read() )
    dtest = xgb.DMatrix( test, missing = missing  )
    res_mid = list( bst.predict(data = dtest) )
    
    res = LR.predict( np.array(res_mid)[:,None] )

    with open('result3.csv','w') as R:
        R.write('Id,Response')
        count = 20000
        #print(res)
        #if(i>1)
        for i in res:
            if(i>1): i=1
            if(i<0): i=0
            R.write('\n' + str(count) + ',' + str( int(i*7 + 0.5) + 1 ) )
            count += 1 
'''            
with open('data/test_data_xgb.json') as f:
    test = json.loads( f.read() )
    dtest = xgb.DMatrix( test, missing = missing  )
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    
    bst = xgb.train( plst, dtrain, num_round)
    bst.save_model('trained.model')
    
    res = bst.predict(data = dtest)
    
    with open('result3.csv','w') as R:
        R.write('Id,Response')
        count = 20000
        #print(res)
        #if(i>1)
        for i in res:
            if(i>1): i=1
            if(i<0): i=0
            R.write('\n' + str(count) + ',' + str( int(i*7 + 0.5) + 1 ) )
            count += 1 
        
'''    