Required package: 
numpy, pandas

Credit to:
https://blog.csdn.net/jiede1/article/details/78245597 for the code framework
https://blog.csdn.net/haimengao/article/details/49615955 for the algorithm design
https://www.kaggle.com/thomascleberg/ordinal-regression for the data preprocessing idea and code

To run the code:
1. put training.csv and testing.csv in the data folder
	
2. in main folder, run
	random_forest.py (if you DO NOT want to regenerate model), result in res.csv
    OR random_forest_new.py (if you want to regenerate model, might take >1 hour), 
    THEN change line 103 to 'model_new.json', rerun random_forest.py