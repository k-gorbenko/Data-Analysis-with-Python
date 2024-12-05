import numpy as np

def softmax(vector):
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    attention_scores = W_mult @ decoder_hidden_state  # (n_features_enc, 1)
    attention_weights = np.dot(attention_scores.T, encoder_hidden_states)  # (1, n_states)
    attention_weights = softmax(attention_weights)  # (1, n_states)
    attention_vector = attention_weights.dot(encoder_hidden_states.T).T  # (n_features_enc, 1)
    return attention_vector

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    encoder_transformed = W_add_enc @ encoder_hidden_states  # (n_features_int, n_states)
    decoder_transformed = W_add_dec @ decoder_hidden_state  # (n_features_int, 1)
    scores = v_add.T @ (encoder_transformed + decoder_transformed.T)  # (1, n_states)
    scores = np.tanh(scores)  # (1, n_states)
    attention_weights = softmax(scores)  # (1, n_states)
    attention_vector = encoder_hidden_states @ attention_weights.T  # (n_features_enc, 1)
    return attention_vector
