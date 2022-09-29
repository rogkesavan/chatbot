import operator
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import textdistance 
import json
with open('./data/faq.json', 'r') as fp:
    data = json.load(fp)

"""
WordNet is a lexical database of semantic relations between words in more than 200 languages. WordNet links words into semantic relations including synonyms, hyponyms, and meronyms. 
The synonyms are grouped into synsets with short definitions and usage 
"""
def tagged_to_synset(word):
    re=[]
    for i in word.split():
        for ss in wordnet.synsets(i):
            re.append(ss)
    return list(set(re))
""" this function helps to compare two sting and give a simiarity score based on the similarity score the main() function will trigered 
the threshold value is 60 so if sentence similarity is bellow 60 then system will say i wouldn't understand user query
"""
def sentence_similarity(sentence1, sentence2):
    try:
        # Get the synsets for the tagged words
        synsets1 = tagged_to_synset(sentence1)
        # print(synsets1)
        synsets2 = tagged_to_synset(sentence2)
        # print(synsets2)

        # Filter out the Nones
        synsets1 = [ss for ss in synsets1 if ss]
        synsets2 = [ss for ss in synsets2 if ss]

        score, count = 0.0, 0

        # For each word in the first sentence
        for synset in synsets1:
            # print(synset,"yyyyyyyyyy")
            # Get the similarity value of the most similar word in the other sentence
            # print([synset.path_similarity(ss) for ss in synsets2],"xxxxxxxxxxx")
            best_score = max([synset.path_similarity(ss) for ss in synsets2])
            # print(best_score)

            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1

        # Average the values
        score /= count
        return score
    except:
        # print(textdistance.jaro_winkler(sentence1,sentence2))
        return textdistance.jaro_winkler(sentence1,sentence2)

"""
https://www.aaai.org/Papers/AAAI/2006/AAAI06-123.pdf
this function is symmetric similarity two sentence can be measure in both side for reference please read that paper and use this link 
https://nlpforhackers.io/wordnet-sentence-similarity/
"""
def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2 

"""this wordnet function is process all sentence and forward to intent name to main()"""

def wordnet(sentence):
    result = []
    f = []
    for k,v in data.items():
        for j in v:
            score = symmetric_sentence_similarity(sentence,j)
            if score == None:
                pass
            else:
                if score>0.70:
                    f.append({"label":k,"score":score})
    if f ==[]:
        return None
    else:
        maxPricedItem = max(f, key=lambda x:x['score'])
        return maxPricedItem