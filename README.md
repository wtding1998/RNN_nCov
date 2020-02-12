# Predict the Distribution of Disease by RNN and STNN

This project is supported by the National Natural Science Foundation of China (Grant No: 11601327) and the Key Construction National “985” Program of China (Grant No: WF220426001).

## train_rnn

### data 文件夹
下面存放数据。（之前的疾病没有整理好，这里只看ncov相关的文件夹）

每个文件夹下面的time_data文件夹存放时间数据，每种数据一个文件夹。例如ncov/time_data下面有3个文件，分别是确诊、死亡、治愈的数据。

每个文件的行数就是时间数，列数是地点数。ncov/time_data/confirmedday.csv有26行31列，就是26天31个省份。

- ncov_sum，所有省份数据加和，相当于只有一个地点。

- ncov_confirmed/dead/cured ：把每个数据的和单独出来

### train_rnn.py
运行的程序。用命令行运行的话就直接把参数输入进去(在参数前加2个-, 再空一格，将赋值输入进去)，比如 
```python
python train_rnn.py --lr 1e-2 --dataset ncov --rnn_model LSTM --test True
```
就是把lr这个参数指定为1e-2.

注意这里程序所在的文件夹的上一级需要加一个output文件夹，也就是不然可能会报错，输出的日志放在里面。

#### 需要调整的参数
- dataset : 训练的数据所在文件夹。可选ncov, ncov_sum, ncov_confirmed/dead/cured，按照data文件夹下的文件夹名称。
- start_time : 开始训练的时间点。因为考虑到前面的数据可能不准确，可以只用后面的数据。start_time取从0开始的整数，比如start_time=3的话就是只考虑第4天之后的数据，前3天数据不用。
- nt_train : 用于训练的时间数。用start_time后的前nt_train个数据作为训练集，注意这里start_time + nt_train <26，否则总时间就不够了。
- seq_length : rnn选择用连续的seq_length个数据预测下一个。一般这个越大，训练越慢。
- nhid : rnn层的单元数
- nlayers : rnn层的层数
- rnn_model : 选择模型。有LSTM, GRU, LSTM_one, GRU_one可以选择。 LSTM_one, GRU_one和LSTM, GRU分别只有最后的线形层有差别，一个是取rnn层输出序列作线性映射，另一个是取rnn层输出序列最后一层。
- lr ：学习率
- sch_bound / sch_factor /patience : 通过验证集的loss来调整学习率，当验证集的loss在patience个epoch中都小于sch_bound的话，学习率就会乘以sch_factor，在train_rnn.py 的171行-173行，可以注释掉
- beta1/beta2/eps : Adam的参数，可以不用调整
- wd : 学习的时候增加正则项，对参数的惩罚系数
- batch_size : 训练时选取的一个批次内的数据多少
- test ：是否在训练的时候取用验证集
- clip_value : 梯度裁剪的范围

## Data
### Aids
The file `aids.csv` contains the raw temperature data. The 156 rows correspond to the 156 timestep, and the 29 columns are the 29 space points.
The file `aids_relations.csv` contains the spatial relation between the 29 space points. It is a 29 by 29 adjacency matrix _A_, where _A(i, j)_ = 1 means that series _i_ is a direct neighbor of series _j_ in space, and is 0 otherwise.
### Flu
The file `flu.csv` contains the raw temperature data. The 156 rows correspond to the 156 timestep, and the 29 columns are the 29 space points.
The file `flu_relations.csv` contains the spatial relation between the 29 space points. It is a 29 by 29 adjacency matrix _A_, where _A(i, j)_ = 1 means that series _i_ is a direct neighbor of series _j_ in space, and is 0 otherwise.
### Heat
The file `heat.csv` contains the raw temperature data. The 200 rows correspond to the 200 timestep, and the 41 columns are the 41 space points.
The file `heat_relations.csv` contains the spatial relation between the 41 space points. It is a 41 by 41 adjacency matrix _A_, where _A(i, j)_ = 1 means that series _i_ is a direct neighbor of series _j_ in space, and is 0 otherwise.
## Model
### Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relation Discovery

ICDM 2018 - IEEE International Conference on Data Mining series (ICDM)

[Conference Paper](https://ieeexplore.ieee.org/document/8215543/)

[Journal Extension](https://link.springer.com/article/10.1007/s10115-018-1291-x)

Commands for reproducing synthetic experiments:

#### STNN
`python train_stnn.py --dataset aids --outputdir output_aids --manualSeed 1932 --xp stnn`

`python train_stnn.py --dataset flu --outputdir output_flu --manualSeed 7011 --xp stnn`

`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 2021 --xp stnn`

#### STNN-R(efine)
`python train_stnn.py --dataset aids --outputdir output_aids --manualSeed 3301 --xp stnn_r --mode refine --patience 800 --l1_rel 1e-8`

`python train_stnn.py --dataset flu --outputdir output_flu --manualSeed 3796 --xp stnn_r --mode refine --patience 800 --l1_rel 1e-8`

`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 5718 --xp stnn_r --mode refine --patience 800 --l1_rel 1e-8`
#### STNN-D(iscovery)
`python train_stnn.py --dataset aids --outputdir output_aids --manualSeed 1290 --xp stnn_d --mode discover --patience 1000 --l1_rel 3e-6`

`python train_stnn.py --dataset flu --outputdir output_flu --manualSeed 8837 --xp stnn_d --mode discover --patience 1000 --l1_rel 3e-6`

`python train_stnn.py --dataset heat --outputdir output_heat --manualSeed 9690 --xp stnn_d --mode discover --patience 1000 --l1_rel 3e-6`
<!-- ## Modulated Heat Diffusion
### STNN
`python train_stnn.py --dataset heat_m --outputdir output_heat_m --manualSeed 679 --xp stnn`

### STNN-R(efine)
`python train_stnn.py --dataset heat_m --outputdir output_heat_m --manualSeed 3488 --xp stnn_r --mode refine --l1_rel 1e-5`

### STNN-D(iscovery)
`python train_stnn_.py --dataset heat_m --outputdir output_m --xp test --manualSeed 7664 --mode discover --patience 500 --l1_rel 3e-6` -->

### RNN
Here **LSTM** and **GRU** are used.

Commands for reproducing synthetic experiments:
#### LSTM
`python train_rnn.py --dataset aids --model LSTM --manualSeed 1208 --xp LSTM_aids`

`python train_rnn.py --dataset flu --model LSTM --manualSeed 1471 --xp LSTM_flu`

`python train_rnn.py --dataset heat --model LSTM --manualSeed 6131 --xp LSTM_heat`
#### GRU
`python train_rnn.py --dataset aids --model GRU --manualSeed 1208 --xp GRU_aids`

`python train_rnn.py --dataset flu --model GRU --manualSeed 1471 --xp GRU_flu`

`python train_rnn.py --dataset heat --model GRU --manualSeed 6131 --xp GRU_heat`