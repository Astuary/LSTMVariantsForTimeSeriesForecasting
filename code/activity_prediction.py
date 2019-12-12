import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)

class CNN_LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.con1 = nn.Conv1d(1, 128, 3, stride=2, bias=True, padding=3)
        self.nl1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)
        self.mp1 =  nn.MaxPool1d(3)
        self.con2 = nn.Conv1d(128, 225, 3, stride=2, bias=True, padding=3)
        self.nl2 = nn.ReLU()
        self.dp2 = nn.Dropout(0.5)
        self.mp1 = nn.MaxPool1d(3)
        self.f1 = nn.Flatten(0,1)
        self.lstm_layer1 = nn.LSTM(225, 5, 2, batch_first=True)
        self.dp1 = nn.Dropout(0.5)
        self.l1 = nn.Linear(5, 5)
        self.nl1 = torch.nn.ReLU()
        self.lstm_layer2 = nn.LSTM(5, 5, 2, batch_first=True)
        self.dp2 = nn.Dropout(0.5)
        self.l2 = torch.nn.Linear(5, 5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lstm_layer1(x)
        x = self.dp1(x[0])
        x = self.l1(x)
        x = self.nl1(x)
        x = self.lstm_layer2(x)
        x = self.dp2(x[0])
        x = self.l2(x)
        x = self.sig(x)
        return x

class LSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm_layer1 = nn.LSTM(225, 5, 2, batch_first=True)
        self.dp1 = nn.Dropout(0.5)
        self.l1 = nn.Linear(5, 5)
        self.nl1 = torch.nn.ReLU()
        self.lstm_layer2 = nn.LSTM(5, 5, 2, batch_first=True)
        self.dp2 = nn.Dropout(0.5)
        self.l2 = torch.nn.Linear(5, 5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lstm_layer1(x)
        x = self.dp1(x[0])
        x = self.l1(x)
        x = self.nl1(x)
        x = self.lstm_layer2(x)
        x = self.dp2(x[0])
        x = self.l2(x)
        x = self.sig(x)
        return x

class activity_prediction_model:
    """Activity prediction

        You may add extra keyword arguments for any function, but they must have default values
        set so the code is callable with no arguments specified.

    """
    def __init__(self):
        self.model = LSTM().float()

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
            df_test_t = df_test_timestamps[-1] + torch.randint(low=1, high=61, size=(1,))
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

            self.forecast(df_test, df_test_t)


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

        df_timestamps, df_features, df_output = df.iloc[:,0].values, df.iloc[:,1:-5].values, df.iloc[:,-5:].values

        print(df_timestamps)
        print(df_features)
        print(df_output)
        print(t)

        lr=0.005
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        sub = df_timestamps[0]
        df_timestamps = [i - sub + 1 for i in df_timestamps]
        df_features = (df_features.T*df_timestamps).T

        t = t - df_timestamps[0]

        for test_tuple in range(df_features.shape[0]):
            tuple = np.reshape(df_features[test_tuple,:], (1, 1, df_features.shape[1]))
            result = np.reshape(df_output[test_tuple,:], (1, 1, df_output.shape[1]))
            #print(df_train_output.shape)
            self.model.train()
            optimizer.zero_grad()
            out = self.model(torch.tensor(tuple).float())
            loss = criterion(out, torch.tensor(result).float())
            print(loss)
            loss.backward()
            optimizer.step()

        out = out*t
        print(out)
        input()

        return {'label:LYING_DOWN':0.5, 'label:SITTING':0.5, 'label:FIX_walking':0.5, 'label:TALKING':0.5, 'label:OR_standing':0.5}

    def evaluate_model_LSTM(self, df_train, df_test):

        verbose, epochs, batch_size = 0, 15, 64

        for train, test in loo.split(range(len(df.groupby(level=0)))):
            df_train, df_test = df.loc[pd.IndexSlice[train, :],:], df.loc[pd.IndexSlice[test, :],:]
            df_test_timestamps, df_test_features, df_test_output = df_test.iloc[:,0], df_test.iloc[:,1:-5], df_test.iloc[:,-5:]
            df_train_timestamps, df_train_features, df_train_output = df_train.iloc[:,0], df_train.iloc[:,1:-5], df_train.iloc[:,-5:]
            #df_train_timestamps.reset_index(level=1, drop=True, inplace=True)
            #df_train_features.index = df_train_features.index.droplevel(1)
            #df_train_output.index = df_train_output.index.droplevel(1)
            #print(df_train.loc[pd.IndexSlice[2, :],:])
            for train_index in train:
                [w, h] = (df_train.loc[pd.IndexSlice[train_index, :],:'raw_acc:magnitude_stats:time_entropy']).shape
                temp = (df_train.loc[pd.IndexSlice[train_index, :],:'raw_acc:magnitude_stats:time_entropy']).values.reshape(1, w, h)
                print([w,h])
                model = torch.nn.Sequential(
                    torch.nn.LSTM(1, w, h, 100),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(100, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(100, 5),
                    torch.nn.Softmax()
                )
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters())

                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    y_ = model(df_train.loc[pd.IndexSlice[train_index, :],:'raw_acc:magnitude_stats:time_entropy'].values.reshape(1,w,h))
                    loss = criterion(y_, df_train.iloc[0,-5:].values)
                    loss.backward()
                    optimizer.step()
                    print(f"Epoch {epoch+1}/{n_epochs}, loss = {loss.item()}")

                [w, h] = (df_test.loc[pd.IndexSlice[test, :],:'raw_acc:magnitude_stats:time_entropy']).shape
                temp = (df_test.loc[pd.IndexSlice[test, :],:'raw_acc:magnitude_stats:time_entropy']).values.reshape(1, w, h)
                y_test_predict = model(temp)
                mae = np.sum(np.absolute((df_test.loc[pd.IndexSlice[test, :],'label:LYING_DOWN':]).values - y_test_predict))

    def evaluate_model_CNN_LSTM(self, df):

        verbose, epochs, batch_size = 0, 25, 64
        loo = LeaveOneOut()

        for train, test in loo.split(range(len(df.groupby(level=0)))):
            df_train, df_test = df.loc[pd.IndexSlice[train, :],:], df.loc[pd.IndexSlice[test, :],:]
            df_test_timestamps, df_test_features, df_test_output = df_test.iloc[:,0], df_test.iloc[:,1:-5], df_test.iloc[:,-5:]
            df_train_timestamps, df_train_features, df_train_output = df_train.iloc[:,0], df_train.iloc[:,1:-5], df_train.iloc[:,-5:]
            #df_train_timestamps.reset_index(level=1, drop=True, inplace=True)
            #df_train_features.index = df_train_features.index.droplevel(1)
            #df_train_output.index = df_train_output.index.droplevel(1)
            #print(df_train.loc[pd.IndexSlice[2, :],:])
            for train_index in train:
                [w, h] = (df_train.loc[pd.IndexSlice[train_index, :],:'raw_acc:magnitude_stats:time_entropy']).shape
                temp = (df_train.loc[pd.IndexSlice[train_index, :],:'raw_acc:magnitude_stats:time_entropy']).values.reshape(1, -1, w, h)
                print([w,h])
                #input()
                model = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 32, 3, stride=2, bias=True, padding=3),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.MaxPool2d(3),
                    torch.nn.Conv2d(32, 64, 3, stride=2, bias=True, padding=3),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.MaxPool2d(3),
                    torch.nn.Flatten(0,1),
                    torch.nn.LSTM(w, h, 100),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(100, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(100, 5),
                    torch.nn.Softmax()
                )
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters())

                for epoch in range(epochs):
                    model = model.float()
                    model.train()
                    optimizer.zero_grad()
                    y_ = model(torch.tensor(df_train.loc[pd.IndexSlice[train_index, :],:'raw_acc:magnitude_stats:time_entropy'].values.reshape(1,-1,w,h)).float())
                    print(y_)
                    input()
                    loss = criterion(y_, df_train.iloc[0,-5:].values)
                    loss.backward()
                    optimizer.step()
                    print(f"Epoch {epoch+1}/{n_epochs}, loss = {loss.item()}")

                [w, h] = (df_test.loc[pd.IndexSlice[test, :],:'raw_acc:magnitude_stats:time_entropy']).shape
                temp = (df_test.loc[pd.IndexSlice[test, :],:'raw_acc:magnitude_stats:time_entropy']).values.reshape(1, -1, w, h)
                y_test_predict = model(temp)
                mae = np.sum(np.absolute((df_test.loc[pd.IndexSlice[test, :],'label:LYING_DOWN':]).values - y_test_predict))

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

        torch.save(model.state_dict(), "../data/model_para.pth")


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

        self.model = LSTM().float()
        self.model.load_state_dict(torch.load("../data/model_para.pth"))


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
