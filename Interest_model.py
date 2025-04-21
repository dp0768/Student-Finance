"""Use this file to create Interest model and predict interest rate change over time
    - change be used for salary change as well or other interest rate changes"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_SF_interest_rate(SF_Interest_Rate: np.ndarray, option: str):
    """Plotting function for SF Interest Rate distribution.

    :param SF_Interest_Rate: Array of interest rates
    :type SF_Interest_Rate: np.ndarray
    "param option: Option to determine the type of distribution
    :type option: str
    """
    plt.hist(SF_Interest_Rate, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'{option} - Distribution of SF Interest Rate')
    plt.xlabel('Interest Rate')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def interest_RPI(mean: float, std: float):
    """Generates interest rate sample data based on the Retail Price Index (RPI) data.
    Historical data is read from an Excel file, and a premium interest rate of up to 3% is added.
    The premium interest values are not specifically defined, but it is assumed to be a normal distribution.

    :param mean: Mean value for the normal distribution
    :type mean: float
    :param std: Standard deviation for the normal distribution
    :type std: float
    :return: _description_
    :rtype: _type_
    """
    Excel_data = pd.read_excel(str(os.getcwd())+"\Data\Retail_Price_index_Data.xls", sheet_name="data")
    Retail_Price_index = Excel_data.to_numpy()[188:,1]
    
    probability_dist = np.empty(len(Retail_Price_index))
    for i in range(len(probability_dist)):
        value = np.random.normal(mean, std)
        while value > 3 or value < 0:
            value = np.random.normal(mean, std)
        probability_dist[i] = value
    
    #print((np.random.shuffle(Retail_Price_index)))  
    return Retail_Price_index + probability_dist 


def interest_rate_input_data(option: str, length: int, plot: bool=False):
    """Generates interest rate data from sampled data.
    The sample data is based on either historical student finance interest rates, or historical retail price index data

    :param option: Decides sampling method: "History", "Normal", or "Shifted"
    :type option: str
    :param length: Length of output interest rate data array
    :type length: int
    :param plot: Plot student finance interest data, defaults to False
    :type plot: bool, optional
    :raises ValueError: Invalid option error
    :return: Student finance interest rate data
    :rtype: np.ndarray
    """
    if option == "History":
        values = np.array([6.6, 6.3, 5.5, 3.9, 4.6, 6.1, 6.3, 5.4, 5.6, 5.3, 4.2, 4.1, 4.4, 4.5, 6.3, 6.5, 6.9, 7.1, 7.3, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0])
    elif option == "Normal":
        values = interest_RPI(mean=1.5, std=0.5)
    elif option == "Shifted":
        values = interest_RPI(mean=2.5, std=0.5)
    else:
        raise ValueError("Invalid option. Choose 'History', 'Normal', or 'Shifted'.")
    
    SF_Interest_Rate = np.empty(length)
    for i in range(length):
        value = np.random.normal(np.mean(values), np.std(values))
        while value > np.max(values) or value < np.min(values):
            value = np.random.normal(np.mean(values), np.std(values))
        SF_Interest_Rate[i] = value
    
    if plot:
        plot_SF_interest_rate(SF_Interest_Rate, option)
    
    return SF_Interest_Rate

IR_data = interest_rate_input_data(option="Normal", length=1000, plot=True)


#def interest_model(x_train):
    #model = keras.Sequential([
        #keras.layers.Input(shape=(x_train.shape[1],)),
        #keras.layers.LSTM(64, return_sequences=True),
        #keras
    #])
    
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MSE, metrics=["accuracy"])
    #return NotImplementedError("Interest model not implemented yet")
    
#out = interest_model(x_input)