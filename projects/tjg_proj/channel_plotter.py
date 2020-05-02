import pandas as pd
import matplotlib.pyplot as plt


class ChannelPlotter(object):
    def __init__(self, fnirs_df):
        self.fnirs_df = fnirs_df
        self.fnirs_df.drop(["time", "Index"], axis=1, inplace=True)
        print(self.fnirs_df)
        self.fnirs_df.plot()
        plt.show()


cp = ChannelPlotter(pd.read_csv("./exports/2020-04-27_017/2020-04-27_017.csv"))
