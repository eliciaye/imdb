import numpy as np

def get_200_accs(fname,lrs):
    print("Epoch 200")
    test,train=[],[]
    for lr in lrs:
        output_file = fname.format(lr)
        found=False
        with open(output_file) as f:
            for line in f:
                if "Epoch: 200" in line:
                    found=True
                if found:
                    if "Test Acc" in line:
                        test.append(float(line.split(' ')[-1][:-2]))
                    if "Train Acc" in line:
                        train.append(float(line.split(' ')[-1][:-2]))
    print("\ntest=np.array({})".format(test))
    for t in test:
        print(t)
   # for i,t in enumerate(test):
   #     print(ids[i],t)
    print("\ntrain=np.array({})".format(train))
    for t in train:
        print(t)

def get_max_test_accs(fname,lrs):
    print("Epoch with highest test_acc")
    test,train=[],[]
    for lr in lrs:
        output_file = fname.format(lr)
        test_acc,train_acc=[],[]
        with open(output_file) as f:
            for line in f:
                if "Test Acc" in line:
                    test_acc.append(float(line.split(' ')[-1][:-2]))
                if "Train Acc" in line:
                    train_acc.append(float(line.split(' ')[-1][:-2]))
            test.append(max(test_acc))
            train.append(train_acc[np.argmax(np.array(test_acc))])
    print("\ntest=np.array({})".format(test))
    for t in test:
        print(t)
   # for i,t in enumerate(test):
   #     print(ids[i],t)
    print("\ntrain=np.array({})".format(train))
    for t in train:
        print(t)

def get_max_accs(fname,lrs):
    test,train=[],[]
    for lr in lrs:
        output_file = fname.format(lr)
        test_acc,train_acc=[],[]
        with open(output_file) as f:
            for line in f:
                if "Test Acc" in line:
                    test_acc.append(float(line.split(' ')[-1][:-2]))
                if "Train Acc" in line:
                    train_acc.append(float(line.split(' ')[-1][:-2]))
            test.append(max(test_acc))
            train.append(max(train_acc))
    print("\ntest=np.array({})".format(test))
    for t in test:
        print(t)
   # for i,t in enumerate(test):
   #     print(ids[i],t)
    print("\ntrain=np.array({})".format(train))
    for t in train:
        print(t)
   # for i,t in enumerate(train):
   #     print(ids[i],t)

# IDS=[1,6,11,16,21,2,7,12,17,22,26,31,36,41,46,27,32,37,42,47]

def get_tbr():
    percentages=[0.2,0.3,0.4,0.5,0.6] # not used, just seeds
    for percentage in percentages:
        FILENAME='logs/lr{}-percentage{}.log'.format('{}',percentage)
        IDS=[0.05,0.1,0.2,0.3]
        get_200_accs(FILENAME,IDS)
        get_max_test_accs(FILENAME,IDS)
#     get_accs(FILENAME, lrs)
FILENAME='imdb_evals_56149_{}.out'
IDS=list(range(1,10))
get_200_accs(FILENAME,IDS)
get_max_test_accs(FILENAME,IDS)

def get_baseline():
    FILENAME='logs/lr{}.log'
    IDS=[0.05,0.1,0.2,0.3]
    get_200_accs(FILENAME,IDS)
    get_max_test_accs(FILENAME,IDS)

