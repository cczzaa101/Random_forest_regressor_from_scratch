import json
import copy
product_to_ind = {"D4": 0, "A8": 1, "D3": 2, "A2": 3, "D2": 4, "A1": 5, "A7": 6, "B2": 7, "D1": 8, "A3": 9, "E1": 10, "A6": 11, "A5": 12, "C1": 13, "C4": 14, "C3": 15, "C2": 16, "B1": 17, "A4": 18}
banned_keys = [ 'Product_Info_2' ]
global_stat = {}
res = []
with open( 'global_stat.json', 'r') as f:
    global_stat = json.loads( f.read() )
    
with open('testing.csv','r') as f:
    #f = open('training.csv', 'r')
    attributes = f.readline().replace('\n','').split(',') 
    print(attributes)
    
    for line in f:
        l = line.replace('\n', '').split(',')
        for i in range( len(l) ):
            if( i==0 ): continue
            #print( l[i] )
            #break
            if( l[i] == '' ):
                if( attributes[i] == 'Product_Info_2' ):
                    l[i]  = 2
                else:
                    l[i] = global_stat[ attributes[i] ] [ 'sum' ] / global_stat[ attributes[i] ] [ 'count' ]
                    
            if( l[i] in product_to_ind ): l[i] = product_to_ind[ l[i] ]
            else:
                if( attributes[i] == 'Product_Info_2' ): continue
                if( global_stat[ attributes[i] ] [ 'sum' ] - int(global_stat[ attributes[i] ] [ 'sum' ]) == 0):
                    l[i] = int( float(l[i]) + 0.5 )    
                else:
                    l[i] = float( l[i] )
            
        temp = []
        for i in range(len(l)) :
            if(i==0): continue
            if( attributes[i] in banned_keys): continue
            if( attributes[i] == 'Product_Info_2' ): 
                temp.append( l[i] )
                continue
            if( global_stat[ attributes[i] ] [ 'count' ] >= 10000 ):
                temp.append( l[i] )
                
        res.append( copy.deepcopy( temp ) )
        #break   

with open('test_data.json','w') as f:
    f.write( json.dumps(res) )

                    
