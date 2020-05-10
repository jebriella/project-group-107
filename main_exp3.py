from model import VGG_16
from metrics import precision, recall, class_one_accuracy, class_two_accuracy, class_three_accuracy, class_four_accuracy
from data_loaders import dataloader_augmented, testdataloader
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
img_size = [256, 256, 3]
f_maps = 32
l_rate = 0.00001
batch_s = 32
n_epochs = 100
p_dropout = 0.4

X, y, X_val, y_val = dataloader_augmented()

model = VGG_16(img_size, f_maps, dropout = p_dropout)

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=l_rate),
              metrics=['categorical_accuracy',
                       precision,
                       recall,
                       class_one_accuracy,
                       class_two_accuracy,
                       class_three_accuracy,
                       class_four_accuracy])

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, restore_best_weights = True, patience = 20)

History = model.fit(X, y,
                    batch_size = batch_s,
                    epochs = n_epochs,
                    validation_data = (X_val, y_val),
                    callbacks=[es])

plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(History.history["loss"], label="loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.plot(np.argmin(History.history["val_loss"]),
         np.min(History.history["val_loss"]),
         marker="x", color="r", label="best model")

plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend();
plt.show()

# Only run after model has been elected
X_test, y_test = testdataloader()
model.evaluate(X_test, y_test, batch_size = 32)
