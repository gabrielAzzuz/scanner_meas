This repository contains a class with built-in functions which operates the soudfield scanner. 
The *ScannerMeasurement* class works with different API modules:

+ [**Telemetrix**](https://github.com/MrYsLab/telemetrix) is used to operate the stepper motors and the temperature/humidity sensor(s) which are connected to an Arduino board;

+ [**Pytta**](https://github.com/PyTTAmaster/PyTTa) is used to generate and record the audio signals when it is used an audio board like the Presonus USB audiobox or M-audio interfaces. Otherwise, you can use the National Instruments cDAQ with the signal generator and adquisition modules. In this case, the *ScannerMeasurement* class will use the [**nidaqmx**](https://github.com/ni/nidaqmx-python) API module.


# What is necessary to run the class propertly?

## 1 - Regarding _Telemetrix_
Before using Telemetrix, it's necessary to download and install [Arduino's IDE] software. After plug an Arduino board to the PC and open it's software,
- In _Windows -> Device Manager -> Port (COM and LPT)_,  after a right-click on Arduino board, click on _Preferences_. In _Port Settings_, set the 'baud-rate' to 115200 Hz and the 'flow control' to hardware. Then click on OK and close the Device Manager; 
- Open _tools_ tab and check the board's selection in _board_ and _port_;
- Click on _manage libraries_ (tools tab), search for the "Telemetrix4Arduino" and install it;
- On Arduino IDE's _file_ tab, click on _Examples -> Telemetrix4Arduino.ino_. Click to upload the code to the board and wait until finishes. 
### Installing _Telemetrix_ Python's API module
In Anaconda's prompt,
```
pip install telemetrix
```
Full instructions of this module can be found [here](https://mryslab.github.io/telemetrix/).
## 2 - Regarding _Pytta_
Since there's a lot a brands and versions of audio boards, search and install the right driver of your board. It's important to check it's sample rate settings in the driver, also in _Windows -> Control Panel -> Hardware and Sound -> Sound_ - it's value must correspond to the same value that will be used with _Pytta_ (in Python).
### Installing _Pytta_ Python's module
To install the latest version using the Anaconda Prompt, enter the command above,
```
pip install git+https://github.com/pyttamaster/pytta@development
```
Full instructions of this module can be found [here](https://pytta.readthedocs.io/).
## 3 - Regarding _nidaqmx_
First, it is necessary to install NI cDAQ's driver which can be found in it's original [site](https://www.ni.com/pt-br/support/downloads.html). Download and install it following the installation instructions.
### Installing _nidaqmx_ Python's API module
***_nidaqmx_ supports CPython 3.7+ and PyPy3!***. To install it by the Anaconda's prompt,
```
python -m pip install nidaqmx
```
Full instructions of this module can be found [here](https://nidaqmx-python.readthedocs.io/en/latest/).
