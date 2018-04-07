'''
Possibly need to remove attribute with too few samples
'''

import json
import copy

attributes  = []
attributes_type = {}
x = []
y = []


missing = 74122.387

with open('training.csv') as f:
    r = f.readline()
    attributes = r.replace('\n','').split(',')
    for line in f:
        l = line.replace('\n', '').split(',')
        medic_sum = 0

        temp = []
        for ind in range( len(l) ):
            if( attributes[ind] == 'Product_Info_2' ):
                if( l[ind] == ''):
                    temp.append(missing)
                    temp.append(missing)
                else:
                    temp.append( ord( l[ind][0] )  - ord('A') )
                    temp.append( int( l[ind][1] ) )
            elif ( attributes[ind].find('Medical_Keyword')!=-1 ):
                if( l[ind]=='1' ): medic_sum+=1
            elif ( (ind!=0)  and ( attributes[ind] != 'Response' ) ):
                if( l[ind] == ''):
                    temp.append(missing)
                else:
                    temp.append( float(l[ind]) )
        
        temp.append( medic_sum )
        x.append( copy.deepcopy(temp) )
        y.append( int(l[-1]) )

testing = []        
with open('testing.csv') as f:
    r = f.readline()
    attributes = r.replace('\n','').split(',')
    for line in f:
        l = line.replace('\n', '').split(',')
        medic_sum = 0

        temp = []
        for ind in range( len(l) ):
            if( attributes[ind] == 'Product_Info_2' ):
                if( l[ind] == ''):
                    temp.append(missing)
                    temp.append(missing)
                else:
                    temp.append( ord( l[ind][0] )  - ord('A') )
                    temp.append( int( l[ind][1] ) )
            elif ( attributes[ind].find('Medical_Keyword')!=-1 ):
                if( l[ind]=='1' ): medic_sum+=1
            elif ( (ind!=0)  and ( attributes[ind] != 'Response' ) ):
                if( l[ind] == ''):
                    temp.append(missing)
                else:
                    temp.append( float(l[ind]) )
        
        temp.append( medic_sum )
        testing.append( copy.deepcopy(temp) )   
        
with open('processed_xgb.json', 'w') as f:
    f.write( json.dumps( {'x':x, 'y':y } ) )

with open('test_data_xgb.json', 'w') as f:
    f.write( json.dumps( testing ) )