import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, RandomUniform

# augmentation code
def augment_time_stretch(x, y):
    current_timesteps = 21
    current_features = 3
    
    # reshape to (Time, 1, Channels) 
    x_reshaped = tf.reshape(x, (current_timesteps, 1, current_features))
    
    # stretch factor
    factor = tf.random.uniform([], 0.8, 1.2)
    new_size = tf.cast(tf.cast(current_timesteps, tf.float32) * factor, tf.int32)
    
    # stretch/shrink
    #invents data points between existing ones and merges data points when shrinking
    x_resized = tf.image.resize(x_reshaped, [new_size, 1], method='bilinear')
    
    # crop or pad
    x_cropped = tf.image.resize_with_crop_or_pad(x_resized, current_timesteps, 1)
    
    # flatten
    x_final = tf.reshape(x_cropped, (current_timesteps * current_features,))
    
    return x_final, y
# end

tf.keras.utils.set_random_seed(42)

# Hyperparameters
timesteps = 21      
features = 3
cells = 12
classes = 3
learning_rate = 0.01
input_length = timesteps * features  # 39

BATCH_SIZE = args.batch_size or 32

train_dataset = train_dataset.map(augment_time_stretch, num_parallel_calls=tf.data.AUTOTUNE)

if not args.ensure_determinism:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE*4)

train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)


weight_init = RandomNormal(mean=0.0, stddev=1.0, seed=42)
bias_init = RandomUniform(minval=-0.1, maxval=0.1, seed=42)

model = Sequential()
model.add(Reshape((timesteps, features), input_shape=(input_length,)))

model.add(LSTM(
    units=cells,
    kernel_initializer=weight_init,
    recurrent_initializer=weight_init,
    bias_initializer=bias_init,
    unit_forget_bias=False,
    return_sequences=False
))

model.add(Dense(cells, activation='relu', kernel_initializer=weight_init, bias_initializer=bias_init))
model.add(Dense(classes, activation='softmax', kernel_initializer=weight_init, bias_initializer=bias_init))

opt = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
EPOCHS = args.epochs or 35
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, verbose=2, callbacks=callbacks)
