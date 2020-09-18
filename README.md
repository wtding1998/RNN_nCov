# Predict the Distribution of Disease by RNN

This project is supported by the National Natural Science Foundation of China (Grant No: 11601327) and the Key Construction National “985” Program of China (Grant No: WF220426001).

## Data
### nCov2019
The folder `data/ncov/time_datas` contains the raw data, including confirmed, cured and dead data. The 233 rows correspond to the 233 timestep, and the 31 columns are the 31 space points.

### RNN
Here **LSTM** and **GRU** are used.

Commands for reproducing synthetic experiments:
#### LSTM
`python train_rnn.py --dataset ncov --rnn_model LSTM --manualSeed 1208 --xp LSTM_aids`

#### GRU
`python train_rnn.py --dataset aids --rnn_model GRU --manualSeed 1208 --xp GRU_aids`

