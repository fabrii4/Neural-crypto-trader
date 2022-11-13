# Neural crypto trader

An automatic trader of cryptocurrencies based on neural networks.

The principle of operation is based on 2 distinct neural networks:

- A prediction network feeded with past historical data, technical indicators and sentiment data extracted from social media 
(e.g. twitter and reddit submissions, lunarcrush, ...) which forecasts the future trend of the closing price.
- An action network feeded with the historical data and future trend prediction and trained by reinforcement learning to produce optimal 
buy and sell signals.

## Requirements

- The program only supports the Binance eschange platform, so a Binance account is required.

- A linux system is required (the code is written in python, but it includes some system specific operations e.g. ramdisk creation).
It has been tested on Ubuntu 18.04 and 20.04.

## Preparation

- To install the required dependencies, run the script [install-requirements.sh](./install-requirements.sh).

- The script will install, among others things, the library twint used to retrieve twitter submissions. 
In order for this to produce the correctly formatted output for our needs, its implementation needs to be modified.  
To do so, replace the content of the folder  
`/home/$USER/.local/lib/python<VERSION>/site-packages/twint/`  
with the content of [twint_library_modified_files/](./twint_library_modified_files/).

- Add the Binance api key and secret to the file [binance_keys.txt](./binance_keys.txt).
- Add the sudo password to the file [sudo.txt](./sudo.txt).

## Usage

- To launch the application, run the script [binance_interface.py](./binance_interface/binance_interface.py) 
in the folder [binance_interface](./binance_interface).

This will initially update the historical data (it could take some time to retrieve the social media submission) and 
then start processing the configured trading pairs in background.

- To interact with the application, a web interface is provided. To start it, launch the script [web-interface.py](./web_interface/web-interface.py) 
in the folder [web_interface](./web_interface). Then, connect to it from from a web browser at the address http://127.0.0.1:8080

![Example interface](./data/images/interface.png "Example interface")

- To train the prediction network use the script [cross_validation_training.sh](./prediction_network/training/cross_validation_training.sh) in 
[prediction_network/training/](./prediction_network/training).
- To train the action network use the script [train.sh](./action_network/training/train.sh) in [action_network/training](./action_network/training).
- To update/generate the training sets use the scripts in the folder [data_preparation](./data_preparation):
  - prediction network --> [prediction_network/prepare_data_for_training.sh](./data_preparation/prediction_network/prepare_data_for_training.sh)
  - action network --> [action_network/prepare_data_for_training.sh](./data_preparation/action_network/prepare_data_for_training.sh)

## TODO

- Upgrade the prediction network to the latest version and test its performance.
- Improve automatization in action network training procedure.
