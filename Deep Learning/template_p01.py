import numpy as np

def softmax(vector):
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    attention_scores = W_mult @ decoder_hidden_state
    attention_weights = np.dot(attention_scores.T, encoder_hidden_states)
    attention_weights = softmax(attention_weights)
    attention_vector = encoder_hidden_states @ attention_weights.T
    return attention_vector



def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    encoder_transformed = W_add_enc @ encoder_hidden_states
    decoder_transformed = W_add_dec @ decoder_hidden_state
    scores = v_add.T @ (encoder_transformed + decoder_transformed)
    scores = np.tanh(scores)
    attention_weights = softmax(scores)
    attention_vector = encoder_hidden_states @ attention_weights.T
    return attention_vector


