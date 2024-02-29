def get_accuracy_dict(accuracy_dict_full):
    accuracy_dict = {}
    for k in accuracy_dict_full.keys():
        if k in [ "rouge1", "rouge2", "rougeL", "tokens_per_sample" ]:
            accuracy_dict[k] = accuracy_dict_full[k]
    return accuracy_dict

