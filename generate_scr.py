import configargparse

from utils import boolean_string, DotDict

def dataset_time():
    return {'jar': 15, 'feb': 36, 'mar': 34, 'jar_rnn': 15, 'feb_rnn': 44, 'mar_rnn': 71}

def output_one(di, f):
    print('python', end=' ', file=f)
    if di['model'] != 'keras':
        print('train_stnn_' + di['model'] + '.py', end=' ', file=f)
    else:
        print('train_keras.py')
    for k, v in di.items():
        if k != 'model' and k != 'wait':
            print('--' + k + ' ' + str(v), end=' ', file=f)
    
    if di.get('wait', True):
        print('&', end=' ', file=f)

def print_scr(opt, f):
    dataset = dataset_time()
    di = opt.copy()
    nt = dataset[opt.dataset]
    if opt.increase:
        nt_train = nt - opt.nt_pred - 1
    else:
        nt_train = nt - opt.nt_pred
    di.pop('nt_pred')
    di['nt_train'] = nt_train
    for activation in opt.activation:
        for nz in opt.nz:
            for lambd in opt.lambd:
                for nhid in opt.nhid:
                    for nlayers in opt.nlayers:
                        for mode in opt.mode:
                            di['activation'] = activation
                            di['nz'] = nz
                            di['lambd'] = lambd
                            di['nhid'] = nhid
                            di['nlayers'] = nlayers
                            di['mode'] = mode
                            output_one(di, f)
                            print(file=f)
        

if __name__ == "__main__":
    # --- arg ---
    p = configargparse.ArgParser()
    p.add('--datadir', type=str, help='path to dataset', default='data')
    p.add('--mode', type=str, help='path to dataset', default=['default'])
    p.add('--dataset', type=str, help='dataset', default='feb')
    p.add('--time_datas', type=str, help='dataset', default='confirmed')
    p.add('--outputdir', type=str, help='outputdir', default='mar051200')
    p.add('--nt_pred', type=int, help='time to pred', default=3)
    p.add('--increase', type=boolean_string, help='increase', default=False)
    p.add('--test', type=boolean_string, help='test', default=True)
    p.add('--log', type=boolean_string, help='log', default=True)
    p.add('--model', type=str, help='model', default='v0')
    p.add('--activation', type=str, nargs='+', help='activation', default=['tanh', 'sigmoid', 'relu'])
    p.add('--nlayers', type=int, nargs='+',help='time to pred', default=[2, 4])
    p.add('--nhid', type=int,nargs='+', help='time to pred', default=[100, 10])
    p.add('--nz', type=int,nargs='+', help='time to pred', default=[10, 20])
    p.add('--lambd', type=float,nargs='+', help='time to pred', default=[0.1, 10])
    p.add('--wait', type=boolean_string, default=True)
    p.add('--nepoch', type=int, default=3000)
    p.add('--batch_size', type=int, default=10)
    p.add('--validation_length', type=int, default=3)
    p.add('--data_normalize', type=str, default='d')


    f = open(r'scr_output.txt', 'w')
    opt = DotDict(vars(p.parse_args()))
    print_scr(opt, f)