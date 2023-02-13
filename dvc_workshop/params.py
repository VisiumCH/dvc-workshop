"""Parameters used to run the different DVC steps."""


class GlobalParams:
    """Global parameters."""

    DEBUG = True


class PreprocessParams:
    """Parameters for the preprocessing step."""

    LOWER_BOUND_COLOR = [0, 0, 200]
    UPPER_BOUND_COLOR = [0, 0, 255]
    THRESHOLD = 0.005


class ModelParams:
    """Model parameters."""

    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    NUMBER_CHANNELS = 3
    POOLING = "max"
    ACTIVATION = "sigmoid"
    TOP = False


class TrainingParams:
    """Parameteres for the training step."""

    BACTH_SIZE = 32
    SEED = 42
    TRAINING_EPOCHS = 3
    TRAINING_LR = 1e-3
    TUNING_EPOCHS = 1
    TUNING_LR = 1e-5
    PATIENCE = 5
