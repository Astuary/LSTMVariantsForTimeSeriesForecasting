# RNN Variants For TimeSeries Forecasting
The ExtraSensory data set has been used. This is a human activity recognition data set containing data from 60 individuals. The task that I will focus on, is probabilistic activity forecasting. Given a sub-dataframe consisting of between one and 30 consecutive observations for a single individual and a timestamp value t, the objective is to predict the (log) probability that each of the five labels (label:LYING_DOWN, label:SITTING, label:FIX_walking, label:TALKING, label:OR_standing) is active (e.g., takes the value 1) at the future specified time, t.

In this project, I will leveragemy knowledge of Recurrent Neural Networks. I will use the ExtraSensory data set for this. This is a human activity recognition data set containing data from 60 individuals. A full description of the base data set is available here: http://extrasensory.ucsd.edu/. 

As data for this project, there is a lightly processed version of the base ExtraSensory data set distributed as a Pandas dataframe. To get started with this project, one should first read over the ExtraSensory data set documentation, then load and explore the provided dataframe in Jupyter Notebook.

The primary modifications I have made to the base data set are 
(1) restrict the label set to a subset of the more frequently observed binary activity labels (label:LYING_DOWN, label:SITTING, label:FIX_walking, label:TALKING, label:OR_standing); 
(2) remove rows of the data set containing more than 20% missing data; 
(3) re-map the original user UUIDs to integers; 
(4) partition the users into training and held out sets.

All features in the base data set are present as described. Note that this is a complex, real-world data set. It is moderate dimensional and contains both missing feature and label values. The gaps between successive observations for an individual can also be non-uniform.

The task that we will focus on for this projrct is probabilistic activity forecasting. Given a sub-dataframe consisting of between one and 30 consecutive observations for a single individual and a timestamp value t, the objective is to predict the (log) probability that each of the five labels is active (e.g., takes the value 1) at the future specified time, t. 

The times t will be between one and 60 minutes after the last available observation. Performance will be assessed using average binary cross entropy (averaged over data cases and labels) using data from the held-out set of individuals.

I have mainly used a combination of methods from NumPy, SciPy, Pandas, Scikit-learn and PyTorch in my implementation. The goal was to get a  solution that must operate within the memory, compute, and time resource constraints of the a simple home computer environment.
