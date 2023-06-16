from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment(X, y, preprocessor=None, batch=16):
    '''
    Function to return augmented data
    '''
    # Create data generator object
    datagen = ImageDataGenerator(
        featurewise_center = False,
        featurewise_std_normalization = False,
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        zoom_range = (0.8, 1.2),
        preprocessing_function=preprocessor
        )

    # fits the data generator to the data
    datagen.fit(X)

    # produces the flow training set
    flow = datagen.flow(X, y, batch_size=batch)

    return flow
