from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_self_attention import SeqSelfAttention
 
 
def build_bert(nclass, selfloss, lr, is_train):

    
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
 
 
    for l in bert_model.layers:
        l.trainable = is_train
 
 
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
 
 
    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, :])(x)
 
 
    avg_pool_3 = GlobalAveragePooling1D()(x)
    max_pool_3 = GlobalMaxPooling1D()(x)
    attention_3 = SeqSelfAttention(attention_activation='softmax')(x)
    attention_3 = Lambda(lambda x: x[:, 0])(attention_3)
 
 
    x = keras.layers.concatenate([avg_pool_3, max_pool_3, attention_3])
    p = Dense(nclass, activation='sigmoid')(x)
 
 
    model = Model([x1_in, x2_in], p)
    model.compile(loss=selfloss,
                  optimizer=Adam(lr),
                  metrics=['acc'])
    print(model.summary())
    return model

def f1_loss(y_true, y_pred):
    # y_true:0 or 1
    loss = 2 * tf.reduce_sum(y_true * y_pred) / tf.reduce_sum(y_true + y_pred) + K.epsilon()
    return -loss


build_bert(1, 'binary_crossentropy', 1e-5, True)
build_bert(1, f1_loss, 1e-7, False)
