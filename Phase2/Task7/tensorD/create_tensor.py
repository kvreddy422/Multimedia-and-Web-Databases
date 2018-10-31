import numpy as np


# returns all
def getTerms(inp):
    path = 'desctxt/devset_textTermsPer'+ inp+'.txt'
    a = open(path).readlines()
    entity = []
    entity_terms = []
    vocab = []
    for i in range(len(a)):
        splits = a[i].split(' ')
        entity.append(splits[0])
        terms = []
        for j in range(len(splits)):
            if splits[j].startswith('\"'):
                terms.append(splits[j])
                if inp == 'nothing' and splits[j] not in vocab:
                    vocab.append(splits[j])
        entity_terms.append(terms)
    return vocab, entity, entity_terms


vocab, loc, loc_terms = getTerms('POI')
_, images, image_terms = getTerms('Image')
_, users, user_terms = getTerms('User')

tensor = np.zeros((len(loc), len(images), len(users)))

for i in range(len(loc)):
    for j in range(len(images)):
        for k in range(len(users)):
            a = set(loc_terms[i]).intersection(set(image_terms[j])).intersection(set(user_terms[k]))
            count = len(a)
            if count != 0:
                tensor[i][j][k] = count
np.save('tensor.npy', tensor)