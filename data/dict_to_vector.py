import json
import copy


product_to_ind = {"D4": 0, "A8": 1, "D3": 2, "A2": 3, "D2": 4, "A1": 5, "A7": 6, "B2": 7, "D1": 8, "A3": 9, "E1": 10, "A6": 11, "A5": 12, "C1": 13, "C4": 14, "C3": 15, "C2": 16, "B1": 17, "A4": 18}
f = open('processed.json', 'r')
l = json.loads(f.read())

f = open('global_stat.json', 'r')
global_stat = json.loads(f.read())

x = []
y = []
for i in l:
    temp = []
    for key in i:
        if(key == 'Response'):
            y.append( i[key] )
            continue
            
        if( i[key] in product_to_ind ):
            temp.append( product_to_ind[ i[key] ])
        else:
            if( global_stat[key]['count']>10000 ):
                temp.append( float( i[ key ] ) )
    
    x.append( copy.deepcopy(temp) )
    
f = open( 'processed.json' , 'w' )
f.write( json.dumps({'x':x, 'y':y }) )
f.close()
