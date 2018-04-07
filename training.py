from sklearn import svm
clf = svm.SVR(C=1.0, epsilon=0.2)
import json

with open('data/processed.json') as f:
    t = json.loads( f.read() )
    clf.fit(t['x'], t['y'] )

test = []
with open('data/test_data.json') as f:
    test = json.loads( f.read() )
    res = clf.predict( test )
    
    with open('result.csv','w') as R:
        R.write('Id,Response')
        count = 20000
        for i in res:
            if(i<8):
                R.write('\n' + str(count) + ',' + str( int(i + 0.5 ) ) )
            else:
                R.write('\n' + str(count) + ',' + str( 8 ) )
            count += 1
    

