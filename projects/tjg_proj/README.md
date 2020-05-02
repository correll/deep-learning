### fNIRS_Quality_Checker.

### Installing Requirements.

Need >= python3.6

*All requirements should be included miniconda, anaconda.*

1. Clone, etc.
2. (*optional*): `python3.6 -m venv venv`
3. (*optional*): `source venv/bin/activate` (`venv/bin/activate.bat` on Windows, I believe)
4. `pip install -r requirements.txt` (`conda install requirements.txt`)
5. Place .nirs files into `data_to_check.`
6. run `python fnirs_checker.py`
7. exports located in `./exports`.
8. Run condition_splitter.py.
9. running modeling/exploration.py

### Project Description:

The goal of this project was to create a pre-processing pipelines for allowing deep learning modeling techniques on samples of functional near infrared spectroscopy (fNIRS) data. The sample data included in this repository was collected on the author of this repository, and is therefore, available for use for anyone with access to this repository.

### Dataset

The data set currently contains 16 individual data collection sections. The tasks the participant underwent during the data collection trials are refereed to as finger tapping tasks. Where the participant simply taps their finger on a given hand to a given tempo. Each file in the data set contains one of each of the following condition types:

"Finger_Tapping_right_120"
"Finger_Tapping_right_80"
"Finger_Tapping_left_120"
"Finger_Tapping_left_80"
"Finger_Tapping_both_120"
"Finger_Tapping_both_80"

The presentation of each of these conditions lasted for 45 seconds each and were separated by 10 seconds of rest between each condition where the participant stared at a fixation point in the center of the screen.

The 120 / 80s at the end of each condition label refers to the number of beats per minute the participant is tapping to. The stimulus presentation software that ran the stimulus materials also provided a "click track" for the participant to match their rate of finger tapping to.

### fnirs_checker.py

the fNIRSChecker() class combs through the data provided to the application and does a few things.

1. Ensures the signal quality is satisfactory.
2. Changes the data format from a MATLAB file format into a series of.csv files.
3. Adds stimulus data to those .csv files.

### condition_splitter.py

The ConditionSplitter() class takes the .csv files created from the fnirs_checker and spilts the tables based on the trigger information and stores the data in a pickled dictionary so that it can be easily loaded and fed into a model. The data samples are stored with the task labels.

### modeling/exploration.py

This class reads in the data as a data_dictionary (the pickle file created via the last step) and runs the model located within the self.run_model() section of code. More models will be add over time with an emphasis on comparing the model performance on fNIRS data using multiple model architectures.
