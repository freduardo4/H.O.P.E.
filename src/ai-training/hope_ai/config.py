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
    'ignition_timing',  # Added for EGT estimation
    'egt',             # Target for the virtual sensor
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

# PINN parameters
PHYSICS_WEIGHT = 0.1  # Lambda for physics loss
EGT_NORMAL_MAX = 950.0  # Celsius
EGT_ALARM_TEMP = 1000.0

# RUL Forecaster parameters
RUL_SEQUENCE_LENGTH = 30
RUL_HIDDEN_SIZE = 64
RUL_NUM_LAYERS = 2

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
