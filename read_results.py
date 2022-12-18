import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_results(file):
    """
        requires tensorflow==1.12.0
    """
    train_acc = []
    test_acc = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'train_acc':
                train_acc.append(v.simple_value)
            elif v.tag == 'test_acc':
                test_acc.append(v.simple_value)
    return train_acc, test_acc

import glob

def all_baseline():
    all_test,all_train=[],[]
    percentage=0.6
    for lr in [0.05,0.1,0.2,0.3]:
        logdir = f'tensorboard_logs/baseline-200-lr{lr}-layers-dropout0.3/events*' #events*
        eventfile = glob.glob(logdir)[0]

        train,test = get_results(eventfile)
        all_test.append(max(test))
        all_train.append(max(train))
    print("test")
    for t in all_test:
        print(t)
    print("train")
    for t in all_train:
        print(t)

logdir = f'tensorboard_logs/baseline_200-lr{0.05}-layers-dropout0.3/events*' #events*
eventfile = glob.glob(logdir)[0]

train,test = get_results(eventfile)
print(len(train),len(test))
print(train[-1],test[-1])