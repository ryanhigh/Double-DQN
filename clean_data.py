import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument('--dfpt', type=str, default='/home/nlsde/RLmodel/Double-DQN/result2.csv')

args = parser.parse_args()


def getDataframe(data_dir):
    df = pd.read_csv(data_dir)
    df = df.ffill()
    return df


def Normalization(dataframe, mode):
    if mode == 'minmax':
        scaler = MinMaxScaler(feature_range=(1, 2))
    else:
        scaler = StandardScaler()

    dataframe.iloc[:, [2, 3]] = pd.DataFrame(scaler.fit_transform(dataframe.iloc[:, [2, 3]]))
    return dataframe


def Final_Dataframe():
    df = getDataframe(args.dfpt)
    df_scaled = Normalization(df, 'minmax')
    # print(df.iloc[:, [2, 3]].head(5))
    # print(df_scaled.iloc[:, [2, 3]].head(5))
    return df_scaled


def Final_Dataframe_woNorm():
    df = getDataframe(args.dfpt)
    return df
