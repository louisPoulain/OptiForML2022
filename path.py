import os


# Directories paths
DIRNAME = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(DIRNAME, 'output')
TRAIN_HISTORY_DIR = os.path.join(OUT_DIR, 'training_history')
TRAIN_MODEL_DIR = os.path.join(OUT_DIR, 'pretrained_model')

FIGURE_DIR = os.path.join(DIRNAME, 'fig')
