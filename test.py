import tensorflow as tf

from plato.models.tensorflow import lenet5
from plato.datasources.tensorflow import mnist

model = lenet5.Model.get_model()
data = mnist.DataSource()

loss_criterion = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.01)

model.compile(optimizer=optimizer,
              loss=loss_criterion,
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(data.trainset, epochs=5)
model.summary()
score = model.evaluate(data.testset, verbose=0)
print(f'Accuracy = {score}')
