# Learning rate parameters
BASE_LR = 0.001
EPOCH_DECAY = 30  # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1  # factor by which the learning rate is reduced.


# DATASET INFO
NUM_CLASSES = 2  # set the number of classes in your dataset
DATA_DIR = '../data/classification/'  # to run with the sample dataset, just set to 'hymenoptera_data'

# DATALOADER PROPERTIES
BATCH_SIZE = 32  # Set as high as possible. If you keep it too high, you'll get an out of memory error.

NUM_EPOCHS = 10
MODEL_PATH = '../trained_model/classification/fine_tuned_best_model.pt'
