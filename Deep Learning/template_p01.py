import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    attention_scores = W_mult @ decoder_hidden_state  # (n_features_enc, 1)
    attention_weights = np.dot(encoder_hidden_states.T, attention_scores)  # (n_states, 1)
    attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights), axis=0)
    attention_vector = np.dot(encoder_hidden_states, attention_weights)  # (n_features_enc, 1)
    return attention_vector  # Возвращаем вектор в виде (n_features_enc, 1)

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    encoder_transformed = W_add_enc @ encoder_hidden_states  # (n_features_int, n_states)
    decoder_transformed = W_add_dec @ decoder_hidden_state  # (n_features_int, 1)
    scores = v_add @ (encoder_transformed + decoder_transformed.T)  # (1, n_states)
    scores = np.tanh(scores)  # (1, n_states)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # (1, n_states)
    attention_vector = encoder_hidden_states @ attention_weights.T  # (n_features_enc, 1)
    return attention_vector
