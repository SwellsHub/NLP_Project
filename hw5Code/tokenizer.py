import string

def tokenize(sent):
    """Tokenize and lowercase a string.

    Parameters:
    - sent (str): The string to tokenize

    Returns:
    - list of strs
    """
    
    words = []
    for w in sent.split():
        if w == "":
            continue
        elif w[0] in string.punctuation:
            words.append(w[0])
            if len(w) > 1:
                words.append(w[1:])
        elif w[-1] in string.punctuation:
            words.append(w[:-1])
            words.append(w[-1])
        else:
            words.append(w)
    return [w.lower() for w in words]
