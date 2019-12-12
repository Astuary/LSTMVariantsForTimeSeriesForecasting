import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut
#import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(225, 5, 2, batch_first=True, dropout=0.2)
        self.l1 = nn.Linear(5, 5)
        self.nl1 = nn.Sigmoid()

    def forward(self, x):
        out, h = self.gru1(x)
        out = self.l1(self.nl1(out[:,-1]))
        return out

class activity_prediction_model:
    """Activity prediction

        You may add extra keyword arguments for any function, but they must have default values
        set so the code is callable with no arguments specified.

    """
    def __init__(self):
        self.model = GRU().float()

    def fit(self, df):
        """Train the model using the given Pandas dataframe df as input. The dataframe
        has a hierarchical index where the outer index (ID) is over individuals,
        and the inner index (Time) is over time points. Many features are available.
        There are five binary label columns:

        ['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:TALKING', 'label:OR_standing']

        The dataframe contains both missing feature values and missing label values
        indicated by nans. Your goal is to design and fit a probabilistic forecasting
        model that when given a dataframe containing a sequence of incomplete observations
        and a time stamp t, outputs the probability that each label is active (e.g.,
        equal to 1) at time t.

        Arguments:
            df: A Pandas data frame containing the feature and label data
        """

        """#print(df.to_string())
        print(df)
        print(df.shape)
        #df.to_excel("data.xlsx")
        print(df.columns[1:-5].values) #features -- without timestamp
        print(df.index.names)
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.loc[pd.IndexSlice[2, :],:'discrete:time_of_day:between21and3'])
        #print(df.loc(axis=0)[pd.IndexSlice[0, :]])"""

        df_timestamps, df_features, df_output = df.iloc[:,0], df.iloc[:,1:-5], df.iloc[:,-5:]
        loo = LeaveOneOut()
        batch_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        lr=0.005
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for train, test in loo.split(range(len(df.groupby(level=0)))):
            df_train, df_test = df.loc[pd.IndexSlice[train, :],:], df.loc[pd.IndexSlice[test, :],:]
            df_test_timestamps, df_test_features, df_test_output = df_test.iloc[:,0].values, df_test.iloc[:,1:-5].values, df_test.iloc[:,-5:].values
            random_int = torch.randint(low=1, high=61, size=(1,))
            #print(random_int)
            df_test_t = df_test_timestamps[-1] + random_int

            """print(random_int)
            print("%.0f" %(df_test_t.data.numpy()))
            print(df_test_timestamps[-1])
            print(df_test_t - df_test_timestamps[-1])
            input()"""

            #sub = df_test_timestamps[0]
            #df_test_timestamps = [i - sub + 1 for i in df_test_timestamps]
            #df_test_features = (df_test_features.T*df_test_timestamps).T
            #df_train_timestamps, df_train_features, df_train_output = df_train.iloc[:,0], df_train.iloc[:,1:-5], df_train.iloc[:,-5:]

            for train_index in train:
                df_train_timestamps, df_train_features, df_train_output = df_train.loc[pd.IndexSlice[train_index, :],:].values[:,0], df_train.loc[pd.IndexSlice[train_index, :],:].values[:,1:-5], df_train.loc[pd.IndexSlice[train_index, :],:].values[:,-5:]
                print(train_index)
                sub = df_train_timestamps[0]
                df_train_timestamps = [i - sub + 1 for i in df_train_timestamps]
                df_train_features = (df_train_features.T*df_train_timestamps).T
                #print(df_train_features.shape)
                #print(df_train_output.shape)

                for train_tuple in range(df_train_features.shape[0]):
                    tuple = np.reshape(df_train_features[train_tuple,:], (1, 1, df_train_features.shape[1]))
                    result = np.reshape(df_train_output[train_tuple,:], (1, 1, df_train_output.shape[1]))
                    #print(df_train_output.shape)
                    self.model.train()
                    optimizer.zero_grad()
                    out = self.model(torch.tensor(tuple).float())
                    loss = criterion(out, torch.tensor(result).float())
                    #print(loss)
                    loss.backward()
                    optimizer.step()

                    #input()

            #print(df_test_t)
            print(self.forecast(df_test, df_test_t[0].data.numpy()))

        self.save_model()

    def forecast(self, df, t):
        """Given the feature data and labels in the dataframe df, output the log probability
        that each labels is active (e.g., equal to 1) at time t. Note that df may contain
        missing label and/or feature values. Assume that the rows in df are in time order,
        and that all rows are for data before time t for a single individual. Any number of
        rows of data may be provided as input, including just one row. Further, the gaps
        between timestamps for successive rows may not be uniform. t can also be any time
        after the last observation in df. There are five labels to predict:

        ['label:LYING_DOWN', 'label:SITTING', 'label:FIX_walking', 'label:TALKING', 'label:OR_standing']

        Arguments:
            df: a Pandas data frame containing the feature and label data for multiple time
            points before time t for a single individual.
            t: a unix timestamp indicating the time to issue a forecast for

        Returns:
            pred: a python dictionary containing the predicted log probability that each label is
            active (e.g., equal to 1) at time t. The keys used in the dictionary are the label
            column names listed above. The values are the corresponding log probabilities.

        """
        df = df.fillna(0)
        df_timestamps, df_features, df_output = df.iloc[:,0].values, df.iloc[:,1:-5].values, df.iloc[:,-5:].values

        lr=0.005
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        sub = df_timestamps[0]
        last = df_timestamps[-1]
        df_timestamps = [i - sub + 1 for i in df_timestamps]
        df_features = (df_features.T*df_timestamps).T

        for test_tuple in range(df_features.shape[0]):
            tuple = np.reshape(df_features[test_tuple,:], (1, 1, df_features.shape[1]))
            result = np.reshape(df_output[test_tuple,:], (1, 1, df_output.shape[1]))
            #print(df_train_output.shape)
            self.model.train()
            optimizer.zero_grad()
            out = self.model(torch.tensor(tuple).float())
            loss = criterion(out, torch.tensor(result).float())
            #print(loss)
            loss.backward()
            optimizer.step()

        #print(out[0][0][0])
        #print(df_output[-1])

        #return {'label:LYING_DOWN':0.5, 'label:SITTING':0.5, 'label:FIX_walking':0.5, 'label:TALKING':0.5, 'label:OR_standing':0.5}
        return {'label:LYING_DOWN':out[0][0][0].data.numpy().item(), 'label:SITTING':out[0][0][1].data.numpy().item(), 'label:FIX_walking':out[0][0][2].data.numpy().item(), 'label:TALKING':out[0][0][3].data.numpy().item(), 'label:OR_standing':out[0][0][4].data.numpy().item()}

    def save_model(self):
        """A function that saves the parameters for your model. You may save your model parameters
           in any format. You must upload your code and your saved model parameters to Gradescope.
           Your model must be loadable using the load_model() function on Gradescope. Note:
           your code will be loaded as a module from the parent directory of the code directory using:
           from code.activity_prediction import activity_prediction_model. You need to take this into
           account when developing a method to save/load your model.

        Arguments:
            None
        """

        torch.save(self.model.state_dict(), "code/model_para.pth")


    def load_model(self):
        """A function that loads parameters for your model, as created using save_model().
           You may save your model parameters in any format. You must upload your code and your
           saved model parameters to Gradescope. Following a call to load_model(), forecast()
           must also be runnable with the loaded parameters. Note: your code will be loaded as
           a module from the parent directory of the code directory using:
           from code.activity_prediction import activity_prediction_model. You need to take this into
           account when developing a method to save/load your model

        Arguments:
            None
        """

        self.model = GRU().float()
        self.model.load_state_dict(torch.load("code/model_para.pth"))


    def pre_process(self, df):
        loo = LeaveOneOut()
        #index_count = len(df.groupby(level=0))
        #X = np.asarray([df.loc[pd.IndexSlice[i, :],:].to_numpy() for i in range(index_count)])
        #print(X.shape)
        #print(df.loc[pd.IndexSlice[1, :],:'discrete:time_of_day:between21and3'])
        """for i in range(1, df.shape[1]):
            #print(df.values[:,2])
            plt.scatter(df.values[:,0], df.values[:,i])
            plt.show()"""
        for i in list(range(1,184)) + list(range(210,218)):
            print(i)
            #print(df.values[:,i])
            print(np.nanmean(df.values[:,i]))
            print(np.nanstd(df.values[:,i]))
            #print((df.values[:,i] - np.nanmean(df.values[:,i]))/np.nanstd(df.values[:,i]))
            df.iloc[:,i] = ((df.values[:,i] - np.nanmean(df.values[:,i]))/np.nanstd(df.values[:,i]))
            #print(df.values[:,i])
            #input()

        df.to_pickle("../data/data_norm.pkl", compression='gzip')


def main():

    #Load the training data set
    df=pd.read_pickle("../data/train_data.pkl", compression='gzip')
    #df_norm = pd.read_pickle("data_norm.pkl")
    #print(df_norm)

    #Create the model
    apm = activity_prediction_model()

    ##Pre-process data
    #df_pp = apm.pre_process(df)

    ##cnn lstm
    #apm.evaluate_model_CNN_LSTM(df)

    #Fit the model
    df = df.fillna(0)
    apm.fit(df)

    #Save the mode
    apm.save_model()

    #Load the model
    apm.load_model()

    #Get a sample of data
    example = df.loc[[0]][:10]

    #Get a timestamp 5 minutes past the end of the example
    t = example["timestamp"][-1] + 5*60

    #Compute a forecast
    f = apm.forecast(example, t)
    print(f)

if __name__ == '__main__':
    main()
