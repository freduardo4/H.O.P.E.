import torch

# OBD2 Parameters to use for training
OBD2_FEATURES = [
    'engine_rpm',
    'vehicle_speed',
    'engine_load',
    'coolant_temp',
    'intake_air_temp',
    'maf_flow',
    'throttle_position',
    'fuel_pressure',
    'short_term_fuel_trim',
    'long_term_fuel_trim',
]

# Model hyperparameters
SEQUENCE_LENGTH = 60  # 60 seconds of data at 1 Hz
LATENT_DIM = 16
ENCODER_UNITS = 64
DECODER_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
