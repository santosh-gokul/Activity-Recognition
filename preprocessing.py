import pandas as pd
import numpy as np
import numba
import time

@numba.jit
def preproc(A_col,Time):
	for i in range(1,len(A_col)):
		if(A_col[i]!=A_col[i-1]):
			Time[i] = 0
		else:
			Time[i] = Time[i-1]+0.01

	return(Time)
	"""
	It is used to check working of my logic
	ind_sum = 0	
	for i in range(0,len(A_col)-1):
		if(A_col[i]!=A_col[i+1]):
			if(np.rint(Time[i]/0.01)==ind_sum):
				pass
			else:
				print(np.rint(Time[i]/0.01),ind_sum)
				break
			ind_sum = 0
		else:
			ind_sum+=1
	"""
	print(Time)

time_tot = 0
for i in range(1,10):

	df = pd.read_pickle("subject10"+str(i)+"_dat.pkl")
	df = df.iloc[1:,:]

	A_col = df["A_ID"]
	Time = df["Time"]
	Time[0] = 0

	start = time.time()
	Time = preproc(A_col.to_numpy(),Time.to_numpy())
	end = time.time()
	df["Time"] = Time
	df.to_pickle("Msubject10"+str(i)+"_dat.pkl")
	time_tot+=(end-start)

print(time_tot)
