class PreprocessParams:
    LOWER_BOUND_COLOR = [0, 0, 200]
    UPPER_BOUND_COLOR = [0, 0, 255]
    THRESHOLD = 0.005


class ModelParams:
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    NUMBER_CHANNELS = 3
    POOLING = "max"
    ACTIVATION = "sigmoid"
    TOP = False
