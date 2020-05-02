import pandas as pd
import numpy as np
import os
import pickle

from trigger_dict import trigger_dict

class ConditionSplitter(object):
    def __init__(self, data_dir="./exports", label_by="left-right"):
        self.data_dir = data_dir
        self.label_by = label_by
        self.fnames, self.files = self.load_files()
        self.data_dict = self.get_data_dict()
        self.one_hot_labels(how=label_by)
        self.pickle_data()


    def pickle_data(self):

        pickle.dump(self.data_dict, open(f"./exports/as_pkl/data_dict_{self.label_by}.pkl", "wb"))


    def one_hot_labels(self, how="left-right"):

        if how == "left-right":
            left_vec = np.array([0, 0, 0])
            right_vec = np.array([0, 1, 0])
            both_vec = np.array([0, 0, 1])
        elif how == "rest-task":
            rest_vec = np.array([0, 0])
            task_vec = np.array([0, 1])


        for val in self.data_dict.values():
            for k in val.keys():
                if "_left_" in k:
                    val[k] = [val.pop(k), left_vec]
                elif "_right_" in k:
                    val[k] = [val.pop(k), right_vec]
                elif "_both_" in k:
                    val[k] = [val.pop(k), both_vec]
                elif "Rest" in k:
                    val[k] = [val.pop(k), np.array([0, 0, 0])]
                else:
                    print(k)


    def get_data_dict(self):

        data_chunks = {}
        for i, df in enumerate(self.files):
            triggers = df["triggers"].tolist()
            triggers = [(indx, trigger_dict[mark]) for indx, mark in enumerate(triggers) if mark > 0]

            data_chunk = {}
            for tri_indx, trigger in enumerate(triggers):
                if trigger[1] != "Rest_End" and trigger[1] != "Task_End":
                    try:
                        data = df.iloc[trigger[0]:triggers[tri_indx + 1][0]]
                        data.drop(["Index", "triggers", "time"], axis=1, inplace=True)
                        data = data.values
                        data_chunk[f"{trigger[1]}_{tri_indx}"] = data
                    except IndexError:
                        print(trigger)

            data_chunks[self.fnames[i]] = data_chunk

        for elm in data_chunks["/2020-04-27_013"]:
            for x in elm:
                print(len(elm))

        return data_chunks


    def load_files(self, extension_to_find=".csv"):
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
        elif extension_to_find == ".csv":
            return [fname[fname.rfind("/"):-4] for fname in nirs_fnames], [pd.read_csv(f"{fname}") for fname in nirs_fnames]











cs = ConditionSplitter()
