import torch
import json
import os

# Base directory for configs
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs')

def load_config(filename):
    with open(os.path.join(CONFIG_DIR, filename), 'r') as f:
        return json.load(f)

# Load JSON configs
features_config = load_config('features.json')
hyperparams_config = load_config('hyperparameters.json')

# OBD2 Parameters to use for training
OBD2_FEATURES = features_config['obd2_features']

# Model hyperparameters
SEQUENCE_LENGTH = hyperparams_config['sequence_length']
LATENT_DIM = hyperparams_config['latent_dim']
ENCODER_UNITS = hyperparams_config['encoder_units']
DECODER_UNITS = hyperparams_config['decoder_units']
BATCH_SIZE = hyperparams_config['batch_size']
EPOCHS = hyperparams_config['epochs']
LEARNING_RATE = hyperparams_config['learning_rate']
VALIDATION_SPLIT = hyperparams_config['validation_split']

# PINN parameters
PHYSICS_WEIGHT = hyperparams_config['physics_weight']
EGT_NORMAL_MAX = hyperparams_config['egt_normal_max']
EGT_ALARM_TEMP = hyperparams_config['egt_alarm_temp']

# RUL Forecaster parameters
RUL_SEQUENCE_LENGTH = hyperparams_config['rul_sequence_length']
RUL_HIDDEN_SIZE = hyperparams_config['rul_hidden_size']
RUL_NUM_LAYERS = hyperparams_config['rul_num_layers']

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

