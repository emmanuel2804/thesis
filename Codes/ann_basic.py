from keras import models, layers

model = models.Sequential()

# Input - Layer
model.add(layers.Dense(50, input_dim=(41), 
    activation='relu', name='input_layer'))

# Hidden - Layers
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation='relu', 
    name='hidden_layer_1'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation='relu', 
    name='hidden_layer_2'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation='relu', 
    name='hidden_layer_3'))

# Output - Layer
model.add(layers.Dense(1, activation='sigmoid', 
    name='output_layer'))