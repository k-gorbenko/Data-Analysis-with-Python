import numpy as np

def softmax(vector):
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    # Убедитесь, что decoder_hidden_state имеет размерность (n_features_dec, 1)
    assert decoder_hidden_state.shape[0] == W_mult.shape[0], "Size mismatch between decoder_hidden_state and W_mult"

    # Вычисляем оценки внимания
    attention_scores = W_mult @ decoder_hidden_state  # (n_features_enc, 1)
    
    # Умножаем оценки на скрытые состояния кодера
    attention_weights = np.dot(attention_scores.T, encoder_hidden_states)  # (1, n_states)
    
    # Применяем softmax
    attention_weights = softmax(attention_weights)  # (1, n_states)
    
    # Вычисляем итоговый вектор внимания
    attention_vector = encoder_hidden_states @ attention_weights.T  # (n_features_enc, 1)
    
    return attention_vector





def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    encoder_transformed = W_add_enc @ encoder_hidden_states
    decoder_transformed = W_add_dec @ decoder_hidden_state
    scores = v_add.T @ (encoder_transformed + decoder_transformed)
    scores = np.tanh(scores)
    attention_weights = softmax(scores)
    attention_vector = encoder_hidden_states @ attention_weights.T
    return attention_vector


