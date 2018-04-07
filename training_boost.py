import xgboost as xgb

import json

missing = 74122.387

with open('data/processed_xgb.json') as f:
    t = json.loads( f.read() )
    #clf.fit(t['x'], t['y'] )
for ind in range( len(t['y']) ):
    t['y'][ind] = (t['y'][ind] -1)/7
dtrain = xgb.DMatrix( t['x'], label = t['y'], missing = missing )

param = {'bst:max_depth':6, 'bst:eta':0.15, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
plst = list( param.items() )
plst += [('eval_metric', 'auc')] 
plst += [('eval_metric', 'ams@0')]

num_round = 50

test = []
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
        
        for i in res:
            R.write('\n' + str(count) + ',' + str( int(i*7 + 0.5) + 1 ) )
            count += 1 
        
    