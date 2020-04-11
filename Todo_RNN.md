# Todo

## ncov 20200404
0115 - 0328
- Jan : 0115(1)-0131(17) : 1-24 nt_train=17 
- Feb : 0201(18)-0229(46) : 18-53 nt_train=29
- Mar : 0301(47) - 0327(73) : 47-80 nt_train=27



## Code
- [x] fix the error in los's log in input and cancat STNN
- [ ] use other model to predict total amount and predict every space point with stnn
- [ ] change rmse loss with more F.mse_loss
- [x] add GRU model
    - [x] add GRU model in rnn_model.py
    - [x] add option in train_rnn.py
    - [x] modify model in train_rnn.py
- [x] add script
- [x] the name of  outputdir
- [x] auto name mode
- [x] make another dir for output
    - [x] os find the dir
- [x] update the git in server

- [x] complete batch_size
- [x] weight decay
- [x] add normalize

**result** : 
- [x] print the information for the given model | *test_time* 
- [x] summary the information of the total folder and get the best result | *nhid*
- [x] summary the information of the total mode | *aids_LSTM*
- [x] change the time for logs
- [x] add the mean to the beginning of the df
- [x] **write the information to be a class**
    - init with folder
    - info
- [x] save and load model
- [x] add pred for STNN
- [x] for the given model list, finish the jupyter notebook to get the information of them.
- [x] add start time and end time in config

something should left now : 
- [x] add test loss in config.json
- [x] add a file to get the best consequence


## To improve STNN:
STNN will weak in some epochs
1. maybe watch the parameters in every model may be 0
1. maybe add test set can improve the peformance

## 2020 - 04 - 08

09 : 39


**Task :**

---

**Diary :**
- 念起即断，念起不随，念起即觉，觉之既无

王维克问了别人，说你说的是对流扩散方程，但是哪个是项是对流，哪个是扩散。我觉得这种对pde的感觉很重要，因为这样子可以了解方程的解的大致状况。

如果脑袋里没有例子，这个数学是很危险的数学。
