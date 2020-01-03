"""
	Documentation : This is just for clarifiaction purposes .
	Algo is the name commonly given to any algorithms so far i have used disable others to activate one.
	array_Y is the target array
	df is the dataframe
	nan_rows are any rows with nan value.
	
	This is the program which gave 99.902 % precision.
	
	
	
	
	
	
	PREPROCESSING NOT DONE PROPERLY, MUST BE DONE AGAIN.
	
	
	
	
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import tree,svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit, KFold

array_X = []
array_Y = []
sum1=0.0
sum2=0.0
sum3=0.0
"""
I pickled the dataset, no need of this preprocessing step, and loading dataset.

col_names = ["Time","A_ID","Heart_Rate"]
for i in ["Imu_Hand_","Imu_Chest_","Imu_ankle_"]:
	cn = []
	cn.extend([i+"Temp",i+"3d16x",i+"3d16y",i+"3d16z",i+"3d6x",i+"3d6y",i+"3d6z",i+"3dgx",i+"3dgy",i+"3dgz",i+"3dmx",i+"3dmy",i+"3dmz",i+"o1",i+"o2",i+"o3",i+"o4"])
	col_names+=cn
	=

df = pd.read_csv('merge1.dat', sep =' ', header = None,names=col_names)
df.to_pickle('merge1_dat.pkl')
"""
#Loading dataset from pickle..
df = pd.read_pickle("Dataset.pkl")
print("Read the file!!, %s is size"%(str(df.shape)))
#Preprocessing sanity check..
#print(np.unique(np.isnan(df.index.values),return_counts=True,return_index=True))
#print(df.iloc[376417,:]) try last digit 6,7,8


# Heart Rate values removed
df.drop(["Heart_Rate"],axis =1, inplace = True)

#Just Validation .
print("Read the file!!, %s is size"%(str(df.shape)))

nan_rows = (df.iloc[:,:].isnull().any(1))
print(nan_rows.value_counts())

"""
*************Please Ignore This part of code segment !!******************

df.drop(df.index[nan_rows],inplace = True)
array_null = df.groupby(df.iloc[:,1])
#print(array_null)
print(df)
for i ,group in array_null :
	#index_array = df.index[i]
	print("In group :",i)
	array_X = []
	nan_rows_heart_rate = (group.iloc[:,2].isnull())
	nan_heart_rows_index = group.index[nan_rows_heart_rate]
	nan_rows = (group.iloc[:,:].isnull().any(1))
	#nan_temp_rows = (group.iloc[:,3:].isnull().any(1))
	print("The index is",nan_heart_rows_index)
	#group1 = group.drop(group.index[nan_rows],inplace = False)

	group1 = group.drop(group.columns[[2]],axis=1,inplace = False)
	#group1 = temp.drop(group.index[nan_rows],inplace = False)
	#temp = df.drop(df.columns[[2]],axis = 1, inplace = False)
	#temp0 = temp.drop()
	group2 = group.drop(group.index[np.logical_or(nan_rows_heart_rate,nan_rows)],inplace = False)
	array_X= group2.drop(group2.columns[[2]],axis =1, inplace = False)#.drop(group.columns[[1,3,8,9,7,16,17,18,19,20,24,25,26,33,34,35,36,42,43,44,50,51,52,53,37]],axis = 1, inplace = False)
	array_Y = group2.iloc[:,2]

	algo = KNeighborsClassifier(n_neighbors = 3)
	algo.fit(array_X,array_Y)
	#print(array_Y)
	#nan_rows_heart_rate_temp = (group1.)
	for j in nan_heart_rows_index :
		ans = MLP.predict([group1.iloc[j,:]])

		print("The predicted values for heart rate at sec :",group1.iloc[j,1] ,"is :",ans)
		df.iloc[j,2] = ans
		print("Completed!!")
	cv = ShuffleSplit(n_splits = 2, train_size = 0.8, test_size =0.2)
	fin = cross_val_score(MLP,array_X,array_Y,cv =cv, scoring ='precision_macro')
	print(fin.mean())
	#algo.fit(array_Y,array_X)
	#group.drop(df.index[nan_temp],axis = 0, inplace = True)
print(df)
df.to_csv('merge2.dat',sep =' ', encoding = 'utf-8')
End here........
"""
#print("The number of NaN rows are :",len(df.index[nan_rows]))
df.drop(df.index[nan_rows], axis=0, inplace=True)
print("Shape after cleanup: %s"%(str(df.shape)))

#Target variable..
array_Y = df["A_ID"]
#Sanity Check for preprocessing...
#print(np.unique(np.isnan(df.index.values),return_counts=True,return_index=True))

#Dropping Irrelevant/As we know Columns
for i in ["Imu_Hand_","Imu_Chest_","Imu_ankle_"]:
	df.drop([i+"Temp",i+"3d6x",i+"3d6y",i+"3d6z",i+"o1",i+"o2",i+"o3",i+"o4"],axis=1,inplace=True)

#Dropping the target column
df.drop(["A_ID"],axis=1,inplace=True)

print("Final shape of the preprocessed dataset and target columns: ",df.shape,array_Y.shape)
array_X = df
#Sanity Check for preprocessing........
#print(df.index[array_Y==1])
#print(array_X.iloc[df.index[array_Y==1],:])
#Algorithms........
#algo = svm.SVC()
#algo = GaussianNB
#algo = tree.DecisionTreeClassifier()
algo =KNeighborsClassifier(n_neighbors = 3)
#algo=MLPClassifier(hidden_layer_sizes = (20,)*2)

cv = ShuffleSplit(n_splits = 5, train_size = 0.8,test_size = 0.2)

#for i in range(5):
	#print(array_X.shape, array_Y.shape)
fin = cross_val_score(algo,array_X,array_Y,cv = cv, scoring='f1_macro')
	#print (fin.mean() )
sum1+=fin.mean()
print (sum1,sum2,sum3)
fin1 = cross_val_score(algo,array_X,array_Y,cv = cv, scoring = 'recall_macro')
	#print(fin1.mean() )
sum2+=fin1.mean()
print (sum1,sum2,sum3)
fin2 = cross_val_score( algo,array_X,array_Y,cv = cv, scoring = 'precision_macro')
sum3+=fin2.mean()

print (sum1,sum2,sum3)
