import tensorflow.keras.backend as K
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy

def focal_loss(alpha, gamma):
    def f_loss(y_true, y_pred):
        y_true1 = y_true[:,0]
        y_pred1 = y_pred[:,0]
        loss1 = (alpha[0] * y_true1 * K.pow((1-K.clip(y_pred1, 0, 1)), gamma))*(-y_true1*K.log(K.clip(y_pred1, 0, 1)))

        y_true2 = y_true[:,1]
        y_pred2 = y_pred[:,1]
        loss2 = (alpha[1] * y_true2 * K.pow((1-K.clip(y_pred2, 0, 1)), gamma))*(-y_true2*K.log(K.clip(y_pred2, 0, 1)))

        y_true3 = y_true[:,2]
        y_pred3 = y_pred[:,2]
        loss3 = (alpha[2] * y_true3 * K.pow((1-K.clip(y_pred3, 0, 1)), gamma))*(-y_true3*K.log(K.clip(y_pred3, 0, 1)))

        y_true4 = y_true[:,3]
        y_pred4 = y_pred[:,3]
        loss4 = (alpha[3] * y_true4 * K.pow((1-K.clip(y_pred4, 0, 1)), gamma))*(-y_true4*K.log(K.clip(y_pred4, 0, 1)))

        return K.sum(loss1) + K.sum(loss2) + K.sum(loss3) + K.sum(loss4)
    return f_loss
