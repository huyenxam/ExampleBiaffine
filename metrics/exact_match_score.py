from metrics.normalize_answer import  normalize_answer

def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    prediction_tokens = normalize_answer(prediction)
    ground_truth_tokens = normalize_answer(ground_truth)

    return (prediction_tokens == ground_truth_tokens)
