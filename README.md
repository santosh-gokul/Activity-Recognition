# Activity-Recognition
Using K-Nearest Neighbor

I've been able to classify the activities performed, using 28 (IMU, Gyroscope readings and more) different parameters and hyper-parameter value 'k'=3, with a F1-score of ~0.992. The preprocessing work for this large data-set(~1.2GB pickled version) was accelerated using numba, numpy and pandas.
  
I've validated the result using 5 iterations of shuffle-split with test-set size as 20\%  of the size of the data-set. 
