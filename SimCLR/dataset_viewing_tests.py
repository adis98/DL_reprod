import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

if __name__ == "__main__":
    file = r'C:\Users\adity\Documents\Course_stuff\Deep Learning\project\sthalles_simclr\SimCLR\datasets\cifar-10-batches-py\data_batch_1'
    data_batch_1 = unpickle(file)
    print(data_batch_1['batch_label'])
    print(len(data_batch_1['labels']))
    print(data_batch_1['data'].shape)
