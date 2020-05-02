import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.stats import variation
from shutil import copy2
import os

import matplotlib.pyplot as plt

from pre_proc import convert_to_optical_density, beer_lambert_law, butter_bandpass_filter

class fNIRSChecker(object):
    def __init__(self, data_dir="./data_to_check", signal_threshold=.07):
        self.signal_threshold = signal_threshold
        self.data_dir = data_dir
        self.fnames, self.files = self.load_files()
        print(f"{len(self.files)} '.nirs' files loaded.")
        # Exporting is handled in this function as well.
        self.check_cv_of_channles()


    def load_files(self, extension_to_find=".nirs"):
        """
        Combs through the self.data_dir and finds all .nirs files.

        Args:
            self.data_dir
        Returns:
            fnames(list): Names of files w/o path or file extentsion.
            files (dict): A dictionary containing all of the file data.
        """

        # find all files with .nirs extension.
        nirs_fnames = []
        for root, dirs, files in os.walk(self.data_dir):
           for fname in files:
               if fname.endswith(extension_to_find):
                   nirs_fnames.append(os.path.join(root, fname))
        if extension_to_find == ".nirs":
            return [fname[fname.rfind("/"):-5] for fname in nirs_fnames], [loadmat(f"{fname}") for fname in nirs_fnames]
        elif extension_to_find == ".tri":
            return [fname[fname.rfind("/"):-8] for fname in nirs_fnames], [pd.read_csv(f"{fname}", sep=";", names=["Time", "Index_Point", "Trigger_Value"]) for fname in nirs_fnames]



    def print_and_export(self, string):
        """
        Prints a string to console and writes it to an output log file.

        Args:
            string: message to display.
        Returns:
            None
        """


        self.export_file.write(string)
        self.export_file.write("\n")
        print(string)


    def setup_export_file(self, fname):
        """
        Creates an export file for application to write log messages to.

        Args:
            fname(string): name of file to be written.
        Returns:
            None
        """

        if not os.path.exists(f"./exports/{fname}"):
            os.mkdir(f"./exports/{fname}")

        self.export_file = open(f"./exports/{fname}/{fname}_chlog.txt", "w")


    def close_export_file(self):
        """
        Closes export file object.

        Args:
            None
        Returns:
            None
        """

        self.export_file.close()


    def print_formatted_message(self, message):
        """Prints message with some visual pagebreaks

        Args:
            message(string): What to print to console / write to logfile.
        Returns:
            None.
        """

        self.print_and_export("----------------------")
        self.print_and_export(message)
        self.print_and_export("----------------------\n")


    def export_fnirs_csv(self, fnirs_data, time_series, fname, append_meta_data=True):
        """
        Exports fnirs data as .csv file.

        Args:
            fnirs_data(pd.DataFrame): DataFrame object of fnirs data.
            fname(string): The name of the the exported file.
        Args (Optional):
            append_meta_data(bool): True if meta data columns should be added.
        Returns:
            None.
        """

        # Adds timeseries as well as trigger col to dataframe.
        if append_meta_data:
            fnirs_data["time"] = time_series
            fnirs_data["triggers"] = self.create_trigger_col(fname, fnirs_data.shape[0])

        fnirs_data.to_csv(f"./exports/{fname}/{fname}.csv")

        return fnirs_data


    def create_fnirs_dataframe(self, file):
        """
        Creates a pandas dataframe of the fnirs data including the time each
        sample was taken as well as event markers. Then exports that dataframe as
        a .csv file.

        Args:
            file(dict): the dictionary created from reading the .nirs file.
        Returns:
            fnirs_data(pd.DataFrame): The pandas dataframe of the fnirs data.
        """

        fnirs_data = pd.DataFrame(file["d"])

        channel_num = fnirs_data.shape[1] / 2
        col_names = []
        for indx, col in enumerate(fnirs_data.columns):
            if indx < channel_num:
                tag = "_wl1"
                col_name = f"CH_{indx}{tag}"
            else:
                tag = "_wl2"
                col_name = f"CH_{str(int(indx - channel_num))}{tag}"

            col_names.append(col_name)

        fnirs_data.columns = col_names
        fnirs_data.index.rename("Index", inplace=True)

        return fnirs_data


    def create_trigger_col(self, fname, num_rows):
        """
        Creates a trigger column for the .csv data export. Data point is 0 if
        there is no trigger during that timestep, and is an int > 0 if a trigger
        was detected.

        Args:
            fname(string): The fnirs file name, used to matcher .tri file.
            now_rows(int): How many rows (samples) the fnirs file has.
        Returns:
            col(list): The list of triggers to be added to a pd.DataFrame.
        """

        # Finds corresponding trigger file.
        trigger_fnames, trigger_files = self.load_files(extension_to_find=".tri")

        for i, tri_fname in enumerate(trigger_fnames):
            if tri_fname == fname:
                trigger_file = trigger_files[i]

        current_trigger = 0
        num_triggers = len(trigger_file["Index_Point"].tolist())

        col = []
        for x in range(num_rows):
            if current_trigger + 1 <= num_triggers:
                if x == trigger_file["Index_Point"].tolist()[current_trigger]:
                    col.append(trigger_file["Trigger_Value"].tolist()[current_trigger])
                    current_trigger += 1
                else:
                    col.append(0)
            else:
                col.append(0)

        return col


    def check_cv_of_channles(self):

        for i, file in enumerate(self.files):
            self.setup_export_file(self.fnames[i])
            self.print_formatted_message(f"Channel signal checking for file: {self.fnames[i]}")

            fnirs_data = self.create_fnirs_dataframe(file)

            channel_cvs = variation(fnirs_data)

            # gets the coefficient of varience for each channel.
            # if cv > alloted threshold, report that the channel is no-good.
            invalid_channels = []
            valid_channels = []
            for indx, cv in enumerate(channel_cvs):
                if cv > self.signal_threshold:
                    msg = f"Channel value: ({indx}) is above the acceptable threshold."
                    invalid_channels.append(indx)
                else:
                    msg = f"Channel value: ({indx}) is OKAY."
                    valid_channels.append(indx)

                self.print_and_export(msg)


            # convert to optical density
            for col in fnirs_data.columns:
                fnirs_data[col] = convert_to_optical_density(fnirs_data[col])
                fnirs_data[col] = beer_lambert_law(fnirs_data[col])
                fnirs_data[col] = butter_bandpass_filter(fnirs_data[col], .0100, .5000, 10.2)
                # Zscore
                fnirs_data[col] = (fnirs_data[col] - fnirs_data[col].mean())/fnirs_data[col].std(ddof=0)
                fnirs_data[col].plot()
                plt.show()


            fnirs_data = self.export_fnirs_csv(fnirs_data, file["t"], self.fnames[i], append_meta_data=True)

            self.print_formatted_message(f"Summary of file: {self.fnames[i]}")
            self.print_and_export("Total Channels : {0}".format(len(channel_cvs)))
            self.print_and_export("Validity Threshold : {0}".format(self.signal_threshold))
            self.print_and_export(("Proportion acceptable : {0}".format(round(len(valid_channels)/len(channel_cvs), 3))))
            self.print_and_export(f"{len(valid_channels)} have acceptable signal quality.")
            self.print_and_export(f"{len(invalid_channels)} have inadequate signal quality.")

            self.close_export_file()



if __name__ == '__main__':
    fc = fNIRSChecker()
