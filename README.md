# Predict the Distribution of Disease by RNN

This project is supported by the National Natural Science Foundation of China (Grant No: 11601327) and the Key Construction National “985” Program of China (Grant No: WF220426001).

## Data
### nCov2019
The file `ncov.csv` contains the raw temperature data. The 171 rows correspond to the 171 timestep, and the 31 columns are the 31 space points.
### Aids
The file `aids.csv` contains the raw temperature data. The 156 rows correspond to the 156 timestep, and the 29 columns are the 29 space points.
### Flu
The file `flu.csv` contains the raw temperature data. The 156 rows correspond to the 156 timestep, and the 29 columns are the 29 space points.
### Heat
The file `heat.csv` contains the raw temperature data. The 200 rows correspond to the 200 timestep, and the 41 columns are the 41 space points.


### train_rnn.py
```python
python train_rnn.py --lr 1e-2 --dataset ncov --rnn_model LSTM --test True
```