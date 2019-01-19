import keras.backend as K

def den_loss_sqr(yTrue, yPred):
    dims = K.int_shape(yPred)
    n = dims[0] if dims[0] is not None else 1
    diff_sums = K.sum(K.square((yTrue-yPred)))
    return (1/(2*n)) * diff_sums

def den_loss_abs(yTrue, yPred):
    dims = K.int_shape(yPred)
    n = dims[0] if dims[0] is not None else 1
    diff_sums = K.sum(K.abs(yTrue-yPred))
    return (1/(2*n)) * diff_sums

den_loss = den_loss_sqr