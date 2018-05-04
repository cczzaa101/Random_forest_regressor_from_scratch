from random import randrange,seed
import numpy as np
import pandas as pd
from copy import deepcopy
from numpy import array,var,mean
from numpy import polyfit
import json

#parameter set up
start_ind = 20000
col_sample_lim = 10
row_sample_lim = 800
err_thres = 1e-7
forest_num = 150
max_depth = 100
D_array = []
tree_array = []
second_fit_model = []
original_D = []
oobCheckResult = []  

'''
    take mean of label subset
'''
def avg(D):
    res = mean(D)
    if( res == np.nan ): return 0
    return res
    
'''
    calc reg_error by var*size
'''
def reg_error(D):
    res = var(D['y'])*len(D['y'])
    if( res == np.nan ): return 0
    return res

'''
    split the subset into two subsets of given feature
    < split point, first subset
    > split point, second subset
'''    
def do_split(D, feature, split_point):
    div0 = []
    div0_y = []
    div1 = []
    div1_y = []
    #print( len(D['x'])
    for i in range( len( D['x'] ) ):
        if( D['x'][i][feature] < split_point ):
            div0.append( D['x'][i] )
            div0_y.append( D['y'][i] )
        else:
            div1.append( D['x'][i] )
            div1_y.append( D['y'][i] )
            
    return {'x': array(div0), 'y':div0_y}, {'x': array(div1), 'y':div1_y}

'''
    find the best split point by trying all the values of randomly selected features
'''    
def find_split_point(D):
    selected_feature = set([])
    while(len(selected_feature)< col_sample_lim):
        selected_feature.add( randrange( len(D['x'][0] ) ) ) #pick up some features
    now_error = reg_error( D )
    sub_D_0 = []
    sub_D_1 = []
    best_feature = None
    best_split_point = None
    best_reg_error = 2147483647
    for F in selected_feature:
        split_value = set( D['x'][ :, F] ) #all unique value for this feature
        for V in split_value:
            sub_D_0, sub_D_1 = do_split( D, F, V )
            new_eval = reg_error(sub_D_0) + reg_error(sub_D_1)
            if( new_eval < best_reg_error ):
                best_reg_error = new_eval
                best_feature = F
                best_split_point = V
                
    if( np.abs(best_reg_error - now_error) <= err_thres ): #no improvement, no need for split
        return None, mean(D['y'])
    else:
        return best_feature, best_split_point

'''
    create a single decision tree traversely
'''        
def create_tree(D, dep):
    F,V = find_split_point(D)
    if( F == None ): return V
    if( (dep > max_depth) or (len(D['y'])<=2) ): return mean(D['y'])
    node = {}
    sub_D_0, sub_D_1 = do_split( D, F, V )
    node[ 'l_child' ] = create_tree(sub_D_0, dep+1)
    node[ 'r_child' ] = create_tree(sub_D_1, dep+1)
    node[ 'F' ] = F
    node[ 'V' ] = V
    #print(node)
    return node

'''
    get the predicted value from a single decision tree
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
    randomly selected samples and build different decision trees
'''        
def build_forest(D):
    global D_array
    global tree_array
    D_array = []
    tree_array = []
    
    for i in range( forest_num ):
        print(i)
        D_array.append({'x':[], 'y': []})
        for j in range( row_sample_lim ):
            ind = randrange( len(D['x'] ) )
            D_array[i]['x'].append( D['x'][ind] )
            D_array[i]['y'].append( deepcopy(D['y'][ind]) )
        D_array[i]['x'] = array( deepcopy(D_array[i]['x']) )
        #print(D_array[i])
        tree_array.append( create_tree(D_array[i], 0) )

'''
    predict values of the given P input
    if is_polyfit is true,
    we are using training input to find the linear relation ship
    between predicted value and actual value
'''        
def do_prediction(P, is_polyfit = False):
    res = []
    global start_ind
    global second_fit_model
    global oobCheckResult
    
    if(is_polyfit):
        for tree in tree_array:
            oobCheckResult.append(0)
    with open('model_new.json','w') as f:
        f.write( json.dumps(tree_array) )
        
    pred_ind = 0
    for i in P:
        sum = 0
        tree_ind = 0
        for tree in tree_array:
            temp = get_value( tree, i)
            
            if(is_polyfit): 
                oobCheckResult[tree_ind] += np.abs( original_D['y'][pred_ind] - temp )
            else:
                temp *= 1 #oobCheckResult[tree_ind]
            sum += temp
            tree_ind += 1
        pred_ind += 1
        if( is_polyfit or True):
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
        print(oobCheckResult)
        f = open('res_new.csv', 'w')
        f.write('Id,Response\n')
        for i in res:
            #f.write( str(start_ind) + ',' + str( int(i+0.5) ) + '\n' )
            #print(i)
            
            temp = 0
            for ind in range( len(second_fit_model) ):
                temp += (i** (len(second_fit_model) - ind - 1) ) * second_fit_model[ind]
            #print(temp)
            f.write( str(start_ind) + ',' + str( int(temp+0.5) ) + '\n' )
            start_ind += 1
        f.close()
    else:
        
        second_fit_model = polyfit(res, original_D['y'], 1)
        print(oobCheckResult)
        print( second_fit_model )
    start_ind = 0

'''
    get the start ID
'''
with open('data/testing.csv') as f:
    l = f.readline()
    l = f.readline()
    start_ind = int(l.split(',')[0])
    
#preprocessing, see readme.txt
with open('data/processed.json') as f:
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
    build_forest( {'x': array(train[target_vars]) , 'y':array(train["Response"])} )
    do_prediction(  array(train[target_vars]), True) 
    do_prediction( array(test[target_vars]) )  

      