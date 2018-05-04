from random import randrange,seed
import numpy as np
from copy import deepcopy
from numpy import array,var,mean
from numpy import polyfit
import json

#seed()
start_ind = 20000
col_sample_lim = 10
row_sample_lim = 500
err_thres = 0.001
forest_num = 40
max_depth = 15
D_array = []
tree_array = []
second_fit_model = []
original_D = []
oobCheckResult = []

def avg(D):
    res = mean(D)
    if( res == np.nan ): return 0
    return res
def reg_error(D):
    res = var(D['y'])*len(D['y'])
    if( res == np.nan ): return 0
    return res
    
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
        
def create_tree(D, dep):
    F,V = find_split_point(D)
    if( F == None ): return V
    if( dep > max_depth ): return mean(D['y'])
    node = {}
    sub_D_0, sub_D_1 = do_split( D, F, V )
    node[ 'l_child' ] = create_tree(sub_D_0, dep+1)
    node[ 'r_child' ] = create_tree(sub_D_1, dep+1)
    node[ 'F' ] = F
    node[ 'V' ] = V
    #print(node)
    return node

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
        
def build_forest(D):
    global D_array
    global tree_array
    D_array = []
    tree_array = []
    
    for i in range( forest_num ):
        D_array.append({'x':[], 'y': []})
        for j in range( row_sample_lim ):
            ind = randrange( len(D['x'] ) )
            D_array[i]['x'].append( D['x'][ind] )
            D_array[i]['y'].append( deepcopy(D['y'][ind]) )
        D_array[i]['x'] = array( deepcopy(D_array[i]['x']) )
        #print(D_array[i])
        tree_array.append( create_tree(D_array[i], 0) )
        
def do_prediction(P, is_polyfit = False):
    res = []
    global start_ind
    global second_fit_model
    global oobCheckResult
    
    if(is_polyfit):
        for tree in tree_array:
            oobCheckResult.append(0)
    with open('model.json','w') as f:
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
        f = open('res.csv', 'w')
        f.write('Id,Response\n')
        for i in res:
            #f.write( str(start_ind) + ',' + str( int(i+0.5) ) + '\n' )
            #print(i)
            
            temp = 0
            for ind in range( len(second_fit_model) ):
                temp += (i** (len(second_fit_model) - ind - 1) ) * second_fit_model[ind]
            #print(temp)
            f.write( str(start_ind) + ',' + str( int(i+0.5) ) + '\n' )
            start_ind += 1
        f.close()
    else:
        
        second_fit_model = polyfit(res, original_D['y'], 1)
        print(oobCheckResult)
        print( second_fit_model )
    start_ind = 0

    
with open('data/processed.json') as f:
    t = json.loads( f.read() )
    original_D = t
    row_sample_lim = max( int(len(t['x'])/40), 500 )
    build_forest(t)
    do_prediction( t['x'] , True) 
    
with open('data/testing.csv') as f:
    l = f.readline()
    l = f.readline()
    start_ind = int(l.split(',')[0])
    
with open('data/test_data.json') as f:
    test = json.loads( f.read() )
    do_prediction( test )    