from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', random_state=1)
import json

with open('data/processed.json') as f:
    t = json.loads( f.read() )
    clf.fit(t['x'], t['y'] )

test = []
with open('data/test_data.json') as f:
    test = json.loads( f.read() )
    res = clf.predict( test )
    
    with open('result2.csv','w') as R:
        R.write('Id,Response')
        count = 20000
        for i in res:
            R.write('\n' + str(count) + ',' + i)
            count += 1 
            
    