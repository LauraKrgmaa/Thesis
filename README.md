# Thesis
Code for my MSc thesis on "Implementation and application of machine learning methods for anomaly detection in periodic sensor signals"
gaitpreprocessing.py is for appling global z-score normalization on the gait data.
ecg200files_difpreprocess.py, ecgpreprocess200filesas8020.py, ecg600filesnopreprocess are scripts for picking ECG data and applying preprocessing if needed.
Under Thingy91 folder are codes used for on-device testing. Under folder models are the model codes from Edge Impulse that were used for trainig (LSTM.py and 1DCNN.py) and the generated zip folders that Edge Impulse generated (gait-model-1dcnn.zip and gait-model-lstm.zip).
Under LSTM_live_inference are files needed to run the inference. Since the files led.c and others repeat for all other live inference and testdata configurations they are not uploaded under others. Under 1DCNN-live_inference, 1DCNNtestdata, LSTM_live_inference and LSTMtestdata are ml.cpp files needed to run that specific type of testing.
The project is usually structured like this 
```
ml-test/
├── README.md
├── prj.conf
├── thingy91_nrf9160_ns.overlay
├── src/
│   ├── ml.cpp          
│   └── ml.h
│   └── led.h
│   └── led.c
│   └── main.cpp
├── model-parameters (from the zip file)
├── edge-impulse-sdk (from the zip file)
├── tflite-model (from the zip file)
├── build
```
if for example 1DCNNtestdata has to be run then this is the structure of the visual studio code project and files from under LSTM_live_inference can be used (only the ml.cpp changes). Visual studio code is used to build the project and nFR Connect for Desktop is used to flash on to the microcontroller and look at the terminal if needed. 
Under each folder are also zephyr.signed.hex files these are flashed onto the microcontroller.

