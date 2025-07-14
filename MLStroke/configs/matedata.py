from .base import commen, data, model, train, test

data.scale = None

model.heads['ct_hm'] = 25

train.batch_size = 8
test.batch_size = 4
train.epoch = 300
train.dataset = 'metadata_train'

test.dataset = 'metadata_test'

class config(object):
    commen = commen
    data = data
    model = model
    train = train
    test = test