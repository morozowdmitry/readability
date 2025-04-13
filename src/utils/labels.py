def label2class(label):
    if label == '1-2':
        return 0
    elif label == '3-4':
        return 1
    elif label == '5-7':
        return 2
    elif label == '8-9':
        return 3
    elif label == '10-11':
        return 4
    return label
