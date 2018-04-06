'''
Possibly need to remove attribute with too few samples
'''

import json
import copy

attributes  = []
attributes_type = {}
processed_data = []
processed_data_ug = []
group_statistics = []
global_statistics = {}

for i in range(9): processed_data.append([])
for i in range(9): group_statistics.append({})


with open('training.csv') as f:
	r = f.readline()
	attributes = r.replace('\n','').split(',')
	for line in f:
		l = line.replace('\n', '').split(',')
		temp = {}
		
		
		for i in range(len(l)):
			if( i==0 ): continue			
			temp[ attributes[i] ] = l[i]
		label = int(temp['Response'])
		
		for i in range(len(l)):
			if( i==0 ): continue
			if( l[i] == '' ): continue
			if( (l[i].replace('-','').replace('.', '').replace('E','').isnumeric()) and (l[i][0]!='E') ):
				if( not attributes[i] in group_statistics[label] ):
					group_statistics[label] [ attributes[i] ] = {'count':0, 'sum': 0}
					
				if( not attributes[i] in global_statistics ):
					global_statistics[ attributes[i] ] = {'count':0, 'sum': 0}
					
				#print( group_statistics[label] [ attributes[i] ] , l[i] )
				group_statistics[label] [ attributes[i] ] ['count'] += 1
				group_statistics[label] [ attributes[i] ] ['sum'] += float( l[i] )
				
				global_statistics[ attributes[i] ] ['count'] += 1
				global_statistics[ attributes[i] ] ['sum'] += float( l[i] )
				
			else:
				attributes_type [ attributes[i] ] = 'STR'
				
				if( not attributes[i] in group_statistics[label] ):
					group_statistics[label] [ attributes[i] ] = {}
					
				if( not l[i] in group_statistics[label] [ attributes[i] ] ):
					group_statistics[label] [ attributes[i] ] [ l[i] ] = 0
					
				if( not attributes[i] in global_statistics ):
					global_statistics[ attributes[i] ] = {}
				
				if( not l[i] in global_statistics [ attributes[i] ]):
					global_statistics[ attributes[i] ] [ l[i] ] = 0
					
				global_statistics[ attributes[i] ] [ l[i] ] += 1
				group_statistics[label ][ attributes[i] ] [ l[i] ] += 1
		processed_data[ label ].append( copy.deepcopy(temp) )
		processed_data_ug.append( copy.deepcopy(temp) )
		
for i in attributes:	
	if( i == 'Id' ): continue
	if(not i in attributes_type):
		if(  global_statistics[i]['sum'] - int(global_statistics[i]['sum']) ==0 ): attributes_type[i] = 'INT'
		else: attributes_type[i] = 'REAL'

		
#print(attributes_type)

for ind in range( len( processed_data_ug ) ):
	label = int( processed_data_ug[ ind ] ['Response'] )
	for key in processed_data_ug[ ind ]:
		if( processed_data_ug [ind] [ key ] == '' ):
			# string, use most frequent one
			if( attributes_type [ key ] == 'STR' ):
				#print( attributes_type [ key ], key )
				count = 0
				best = ''
				for item in group_statistics[ label ][key]:
					if( group_statistics[ label ][key] [ item ] > count ):
						count = group_statistics[ label ][key] [ item ]
						best = item
				
				processed_data_ug [ ind ] [ key ] = best
				#if(ind ==0 ): print( best, count ,key, group_statistics[ label ][key])
			else: 
				if( attributes_type [ key ] == 'REAL' ):
					processed_data_ug[ind] [ key ] = group_statistics[ label ] [ key ] ['sum'] /group_statistics[ label ] [ key ] ['count']
				else:
					processed_data_ug[ind] [ key ] = int( group_statistics[ label ] [ key ] ['sum'] /group_statistics[ label ] [ key ] ['count'] )
				# num, use mean
					
with open('processed.json', 'w') as f:
	f.write( json.dumps( processed_data_ug ) )
	
with open('group_stat.json', 'w') as f:
	f.write( json.dumps( group_statistics ) )
	
with open('global_stat.json', 'w') as f:
	f.write( json.dumps( global_statistics ) )