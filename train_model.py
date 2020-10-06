import numpy as np
from alexnet_trainer import alexnet


WIDTH = 300
HEIGHT = 200
LR = 1e-3
EPOCH = 8
MODEL_NAME = 'Nfs-Payback-car-{}-{}-{}-epochs.model'.format(LR, 'alexnet', EPOCH)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('Balanced_training_data.npy')
train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_Y = [i[1] for i in test]

model.fit({'input':X},{'targets':Y},n_epoch=EPOCH,validation_set=({'input':test_X},{'targets':test_Y}),snapshot_step=500, show_metric=True,run_id=MODEL_NAME)

model.save(MODEL_NAME)

# tensorboard --logdir=foo:D:DEV/python/PyGame/Payback_AI/log