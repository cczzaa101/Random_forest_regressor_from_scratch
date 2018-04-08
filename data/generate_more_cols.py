import xgboost as xgb

import json
import copy
missing = 74122.387

new_x = []
new_y = []

new_test = []
new_test_c = []
with open('processed_xgb.json') as f:
    new_t = json.loads( f.read() )
    new_x = copy.deepcopy( new_t['x'] )
    new_y = copy.deepcopy( new_t['y'] )
    
with open('test_data_xgb.json') as f:
    new_test = json.loads( f.read() ) 
    new_test_c = copy.deepcopy( new_test )
    #print(len(new_x))
for i in range(7):
    #if( i!=6 ):continue
    with open('special_xgb_data/' + str(i+1) + '.json') as f:
        t = json.loads( f.read() )
        
    dtrain = xgb.DMatrix( t['x'], label = t['y'], missing = missing )

    param = {'bst:max_depth':6, 'bst:eta':0.15, 'silent':1, 'objective':'binary:logistic' }
    param['nthread'] = 4
    plst = list( param.items() )
    plst += [('eval_metric', 'auc')] 
    plst += [('eval_metric', 'ams@0')]

    num_round = 300

    test = []

    dtest = xgb.DMatrix( t['x'], missing = missing  )
    #evallist  = [(dtest,'eval'), (dtrain,'train')]
    
    bst = xgb.train( plst, dtrain, num_round)
    #bst.save_model('trained.model')
    print(i)
    res = bst.predict(data = dtest)
    
    dtest = xgb.DMatrix( new_test, missing = missing  )
    res2 = bst.predict(data = dtest)
    print(i)
    
    for ind in range( len(res) ):
        #print(i, len(new_x))
        if( i==0 ):
            new_x[ind].append( float(res[ind]) )
        else:
            new_x[ind][-1] += float(res[ind])
    
    for ind in range( len(res2) ):
        if(i ==0):
        #print( len(new_test_c), len(res2) )
            new_test_c[ind].append( float(res2[ind]) )
        else:
            new_test_c[ind][-1] += float( res[ind] )
with open('processed_xgb_mc.json', 'w') as f:
    #print( new_x[0] )
    f.write( json.dumps({'x':new_x, 'y': new_y }) )
with open('test_data_xgb_mc.json', 'w') as f:
    f.write( json.dumps(new_test_c) )
        
    