alphabets = "abcdefghijklmnopqrstuvwxyz"
def spellMistakes(word) :
    """Function returns possible typos for a word
    Typos taken into account:
        Deletion of a single letter in a work
        Exchange of adjacent letters
    """
    split = [ (word[:i],word[i:]) for i in range(len(word)+1) ]
    delete = [ a + b[1:] for a,b in split if b ]
    transpose = [ a + b[1] + b[0] + b[2:] for a,b in split if len(b) > 1 ]
    return set(delete+transpose)
    

