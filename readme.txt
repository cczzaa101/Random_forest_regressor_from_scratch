To compile and run the code:
1. install xgbboost and sklearn package
2. in data folder, run following files in order
	(1)data_preprocess.py 
	(2)dict_to_vector.py 
	(3)testing_preprocess.py
	(4)data_preprocess_xgb.py
	
3. in main folder, run
	(1)training.py to train and predict by SVM
	(2)training_nn.py to train and predict by NN
	(3)training_boost.py to train and predict by XGB (prefered, result is in result3.csv)