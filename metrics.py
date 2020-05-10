import tensorflow.keras.backend as K

# Precision
def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_f, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Recall (sensitivitet)
def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def class_one_accuracy(y_true, y_pred):
    y_true = y_true[:,0]
    y_pred = y_pred[:,0]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def class_two_accuracy(y_true, y_pred):
    y_true = y_true[:,1]
    y_pred = y_pred[:,1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def class_three_accuracy(y_true, y_pred):
    y_true = y_true[:,2]
    y_pred = y_pred[:,2]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def class_four_accuracy(y_true, y_pred):
    y_true = y_true[:,3]
    y_pred = y_pred[:,3]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
