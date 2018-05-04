from random import randrange,seed
import numpy as np
import pandas as pd
from copy import deepcopy
from numpy import array,var,mean
from numpy import polyfit
import json

#parameter set up
start_ind = 0
forest_num = 150
D_array = []
tree_array = []
second_fit_model = []
original_D = []
oobCheckResult = []
test = []
'''
    get_value function, see random forest new.py
'''
def get_value(node, input):
    #print(node)
    if(type(node) is dict):
        res = 0
        if( input[ node['F'] ] < node['V']):
            res = get_value( node['l_child'] , input)
            if( res == None ): res = get_value( node['r_child'] , input)
            return res
        else:
            res = get_value( node['r_child'] , input)
            if( res == None ): res = get_value( node['l_child'] , input)
            return res
    else:
        return node

'''
    do prediction function, see random forest new.py
'''        
def do_prediction(P, is_polyfit = False):
    res = []
    global start_ind
    global second_fit_model
    global oobCheckResult
    
    if(is_polyfit):
        for tree in tree_array:
            oobCheckResult.append(0)
        
    pred_ind = 0
    for i in P:
        sum = 0
        tree_ind = 0
        for tree in tree_array:
            temp = get_value( tree, i)
            
            if(is_polyfit): 
                oobCheckResult[tree_ind] += np.abs( original_D['y'][pred_ind] - temp )
            else:
                temp *= oobCheckResult[tree_ind]
            sum += temp
            tree_ind += 1
        pred_ind += 1
        if( is_polyfit):
            res.append( sum / forest_num )
        else:
            res.append(sum)
    
    if(is_polyfit): 
        for i in range( len(oobCheckResult) ):
            oobCheckResult[i]/=len(original_D['x'])
            oobCheckResult[i] = 1/oobCheckResult[i]
    
        sum_of_oob = np.sum(oobCheckResult)
    
        for i in range( len(oobCheckResult) ):
            oobCheckResult[i]/=sum_of_oob
        
    if( not is_polyfit):
        #print(oobCheckResult)
        f = open('res.csv', 'w')
        f.write('Id,Response\n')
        for i in res:
            #f.write( str(start_ind) + ',' + str( int(i+0.5) ) + '\n' )
            #print(i)
            
            temp = 0
            for ind in range( len(second_fit_model) ):
                temp += (i** (len(second_fit_model) - ind - 1) ) * second_fit_model[ind]
            if( str(test['Id'][start_ind]) == '27799'): print(temp)
            if( temp < 0.5 ): temp = 0.5
            if(temp > 8): temp = 7.9
            f.write( str(test['Id'][start_ind])  + ',' + str( int(temp+0.5) ) + '\n' )
            start_ind += 1
        f.close()
    else:
        
        second_fit_model = polyfit(res, original_D['y'], 1)
        #print(oobCheckResult)
        print( second_fit_model )

    
#loads trained model    
with open('model_best_sofar.json') as f:
    tree_array = json.loads(f.read())
    print('model_loading finished')

    
#preprocessing and predicting
if(True):
    train = pd.read_csv("data/training.csv")
    test = pd.read_csv("data/testing.csv")
    banned_key = ["Id", "Response"]
    original_D = train.append(test)
    original_D[ 'Product_Info_2' ] = pd.factorize(original_D["Product_Info_2"])[0]
    original_D.fillna(-1, inplace=True)
    original_D['Response'] = original_D['Response'].astype(int)
    train = original_D[ original_D['Response']>0 ].copy()
    test = original_D[original_D['Response']<0].copy()
    target_vars = [col for col in train.columns if col not in banned_key]    
    #print( train['response'])
    row_sample_lim = max( int(len(train["Response"])/5), 800 )
    original_D = {'x': array(train[target_vars]) , 'y':array(train["Response"])}
    do_prediction(  array(train[target_vars]), True) 
    do_prediction( array(test[target_vars]) )  