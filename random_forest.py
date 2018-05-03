from random import randrange
import numpy as np
from copy import deepcopy
from numpy import array,var,mean
import json

start_ind = 20000
col_sample_lim = 10
row_sample_lim = 500
err_thres = 0.001
forest_num = 40
max_depth = 15
D_array = []
tree_array = []

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
        
def do_prediction(P):
    res = []
    global start_ind
    for i in P:
        sum = 0
        for tree in tree_array:
            sum = sum + get_value( tree, i)
        res.append( sum / forest_num )
        
    f = open('res.csv', 'w')
    f.write('Id,Response\n')
    for i in res:
        f.write( str(start_ind) + ',' + str( int(i+0.5) ) + '\n' )
        start_ind += 1
    f.close()
    
with open('data/processed.json') as f:
    t = json.loads( f.read() )
    
    build_forest(t)
    
with open('data/testing.csv') as f:
    l = f.readline()
    l = f.readline()
    start_ind = int(l.split(',')[0])
    
with open('data/test_data.json') as f:
    test = json.loads( f.read() )
    do_prediction( test )    