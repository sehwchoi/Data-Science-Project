import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
from numba import jit
import logging

import math
import random
import time

import numpy as np
from scipy import spatial
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.


#... (4) Re-train the algorithm using different context windows. See what effect this has on your results.


#... (5) Test your model. Compare cosine similarities between learned word vectors.


#.................................................................................
#... global variables
#.................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      # list of all unique tokens
wordcodes = {}                          # dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  # how many times each token occurs
samplingTable = []                      # table to draw negative samples from

# load in the data and convert tokens to one-hot indices
def loadData(filename):
    global uniqueWords, wordcodes, wordcounts
    override = True
    # load existing data
    if override:
        fullrec = pickle.load(open("w2v_fullrec.p","rb"))
        wordcodes = pickle.load(open("w2v_wordcodes.p","rb"))
        uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
        wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
        logging.debug("len_unk: {} code: {}".format(wordcounts['UNK'], wordcodes['UNK']))
        return fullrec

    # load in the unlabeled data file. You can load in a subset for debugging purposes.
    handle = open(filename, "r", encoding="utf8")
    fullconts = handle.read().split("\n")
    fullconts = [entry.split("\t")[1].replace("<br />", "") for entry in fullconts[1:(len(fullconts)-1)]]

    # apply simple tokenization (whitespace and lowercase)
    fullconts = [" ".join(fullconts).lower()]

    print("Generating token stream...")
    # populate fullrec as one-dimension array of all tokens in the order they appear.
    # ignore stopwords in this process
    # keep track of the frequency counts of tokens in origcounts.
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(fullconts[0])
    fullrec = list(filter(lambda x: x not in stop_words, words))
    logging.debug("fullrec: {}".format(fullrec[:100]))
    min_count = 50
    origcounts = Counter(fullrec)

    print("Performing minimum thresholding..")
    # populate array fullrec_filtered to include terms as-is that appeared at least min_count times
    # replace other terms with <UNK> token.
    # update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
    fullrec_filtered = list(map(lambda x: x if origcounts[x] >= min_count else 'UNK', fullrec))
    logging.debug("fullrec_filtered: {}".format(fullrec_filtered[:100]))

    # after filling in fullrec_filtered, replace the original fullrec with this one.
    fullrec = fullrec_filtered
    wordcounts = Counter(fullrec)

    print("Producing one-hot indicies")
    # sort the unique tokens into array uniqueWords
    # produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
    # replace all word tokens in fullrec with their corresponding one-hot indices.
    uniqueWords = list(set(fullrec_filtered))
    wordcodes = {w: i for i, w in enumerate(uniqueWords)}
    #logging.debug("wordcodes: {}".format(wordcodes))
    logging.debug("len_unk: {} code: {}".format(wordcounts['UNK'], wordcodes['UNK']))
    fullrec = list(map(lambda x: wordcodes[x], fullrec))
    #logging.debug("fullrec to indices: {}".format(fullrec))

    #close input file handle
    handle.close()

    # store these objects for later.
    # for debugging, don't keep re-tokenizing same data in same way.
    # just reload the already-processed input data with pickles.
    pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
    pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
    pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
    pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))

    # output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
    return fullrec

def load_test(filename):
    handle = open(filename, "r", encoding="utf8")
    test_pair = handle.read().split("\n")
    test_pair = [entry.split("\t") for entry in test_pair[1:(len(test_pair) - 1)]]
    logging.debug("Filename: {} test_pair: {}".format(filename, test_pair))

    return test_pair

# compute sigmoid value
@jit(nopython=True)
def sigmoid(x):
    return float(1)/(1+np.exp(-x))

# generate a table of cumulative distribution of words
def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
    # global wordcounts
    # stores the normalizing denominator (count of all tokens, each count raised to exp_power)
    # max_exp_count = 0

    override = True
    if override:
        cumulative_dict = pickle.load(open("w2v_sampletable.p","rb"))
        logging.debug("sample table sz: {}".format(len(cumulative_dict)))
        logging.debug("sample table keys: {}".format(list(cumulative_dict.keys())[-5:]))
        return cumulative_dict

    print("Generating exponentiated count vectors")
    # for each uniqueWord, compute the frequency of that word to the power of exp_power
    # store results in exp_count_array.
    exp_count_array = list(map(lambda x: math.pow(wordcounts[x], exp_power), uniqueWords))
    max_exp_count = sum(exp_count_array)

    print("Generating distribution")
    # compute the normalized probabilities of each term.
    # using exp_count_array, normalize each value by the total value max_exp_count so that
    # they all add up to 1. Store this corresponding array in prob_dist
    prob_dist = list(map(lambda x: float(x/max_exp_count), exp_count_array))

    print("Filling up sampling table")
    # create a dict of size table_size where each key is a sequential number and its value is a one-hot index
    # the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
    # multiplied by table_size. This table should be stored in cumulative_dict.
    # we do this for much faster lookup later on when sampling from this table.
    table_size = 1e8
    cumulative_dict = {}
    #logging.debug("i: {} prob_dist: {}".format(0, prob_dist[:0]))
    for i, x in enumerate(prob_dist):
        for k in range(round(sum(prob_dist[:i])*table_size), round(sum(prob_dist[:i+1])*table_size)):
            cumulative_dict[k] = i

    logging.debug("sample table sz: {}".format(len(cumulative_dict)))
    logging.debug("sample table keys: {}".format(list(cumulative_dict.keys())[-5:]))
    pickle.dump(cumulative_dict, open("w2v_sampletable.p","wb+"))

    return cumulative_dict

# generate a specific number of negative samples
def generateSamples(context_idx, num_samples):
    global samplingTable, uniqueWords, randcounter
    results = []
    table_size = 1e8
    # randomly sample num_samples token indices from samplingTable.
    # don't allow the chosen token to be context_idx.
    # append the chosen indices to results
    while len(results) != num_samples:
        sample = samplingTable[random.randint(0, table_size-1)]
        randcounter += 1
        if (sample != context_idx) and (sample not in results):
            results.append(sample)

    #logging.debug("Negative samples: {}".format(results))
    return results

@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token, sequence_chars, W1, W2, negative_indices):
    # sequence chars was generated from the mapped sequence in the core code
    nll_new = 0
    pos_target = 1
    neg_target = 0
    h = np.copy(W1[center_token])
    #print(str(len(h)))
    #print(str(len(w1_error_t_np)))

    for k in range(0, len(sequence_chars)):
        # implement gradient descent. Find the current context token from sequence_chars
        # and the associated negative samples from negative_indices. Run gradient descent on both
        # weight matrices W1 and W2.
        # compute the total negative log-likelihood and store this in nll_new.
        w1_error_t_np = np.zeros(len(h))
        w2_k = np.copy(W2[:, sequence_chars[k]])
        pos_ll = np.dot(h, w2_k)
        neg_ll_t = 0

        for j in range(num_samples*k, num_samples*(k+1)):
            w2_j = np.copy(W2[:, negative_indices[j]])
            neg_prob = np.dot(h, w2_j)
            neg_ll_t += math.log(sigmoid(-1 * neg_prob))

            # W2 gradient for neg samples
            error = sigmoid(neg_prob) - neg_target

            w1_error_t_np = w1_error_t_np + (error * w2_j)
            W2[:, negative_indices[j]] = np.subtract(w2_j, learning_rate * error * h)

        # W2 gradient for seq char
        error = sigmoid(pos_ll) - pos_target
        W2[:, sequence_chars[k]] = np.subtract(w2_k, learning_rate * error * h)

        # W1 gradient
        w1_error_t_np = w1_error_t_np + (error * w2_k)
        W1[center_token] = np.subtract(h, learning_rate * w1_error_t_np)

        new_pos_ll = np.dot(h, np.copy(W2[:, sequence_chars[k]]))
        new_neg_ll_t = 0
        for j in range(num_samples*k, num_samples*(k+1)):
            w2_j = np.copy(W2[:, negative_indices[j]])
            neg_prob = np.dot(h, w2_j)
            new_neg_ll_t += math.log(sigmoid(-1 * neg_prob))
        nll = -1 * math.log(sigmoid(new_pos_ll)) - new_neg_ll_t
        nll_new += nll

    return [nll_new]

# learn the weights for the input-hidden and hidden-output matrices
def trainer(curW1 = None, curW2=None):
    global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size, np_randcounter, randcounter
    vocab_size = len(uniqueWords)           # unique characters
    hidden_size = 100                       # number of hidden neurons
    context_window = [1, 2, 3, 4]           # specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
    #context_window = [-1,1]
    nll_results = []                        # keep array of negative log-likelihood after every 1000 iterations

    # determine how much of the full sequence we can use while still accommodating the context window
    start_point = int(math.fabs(min(context_window)))
    end_point = len(fullsequence)-(max(max(context_window),0))
    mapped_sequence = fullsequence

    # initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
    if curW1 is None:
        np_randcounter += 1
        W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
        W2 = np.random.uniform(-.5, .5, size=(hidden_size, vocab_size))
    else:
        # initialized from pre-loaded file
        W1 = curW1
        W2 = curW2

    # set the training parameters
    epochs = 5
    num_samples = 2
    learning_rate = 0.05
    nll = 0
    iternum = 0


    # Begin actual training
    for j in range(0, epochs):
        print ("Epoch: ", j)
        prevmark = 0

        # For each epoch, redo the whole sequence...
        for i in range(start_point, end_point):

            if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
                print ("Progress: ", round(prevmark+0.1,1))
                prevmark += 0.1
            #if iternum%1000==0:
            if iternum % 10000 == 0:
                #print ("Negative likelihood: ", nll)
                logging.debug("Negative likelihood: {}".format(nll))
                nll_results.append(nll)
                nll = 0

            # determine which token is our current input. Remember that we're looping through mapped_sequence
            center_token = mapped_sequence[i]
            # don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
            iternum += 1

            if uniqueWords[center_token] == 'UNK':
                #logging.debug("Found UNK, pass")
                continue

            # now propagate to each of the context outputs
            mapped_context = [mapped_sequence[i+ctx] for ctx in context_window]
            negative_indices = []
            #start = time.time()
            for q in mapped_context:
                negative_indices += generateSamples(q, num_samples)
            #logging.debug("center_token: {} negative_samples: {} mapped_context: {}".format(center_token, negative_indices, mapped_context))
            #end = time.time()
            #logging.debug("gen sample time:{}".format(end-start))

            #start = time.time()
            [nll_new] = performDescent(num_samples, learning_rate, center_token, mapped_context, W1,W2, negative_indices)
            #end = time.time()
            #logging.debug("descent time:{}".format(end-start))
            if iternum % 10000 == 0:
                print ("descent nll: ", nll_new)

            nll += nll_new

    for nll_res in nll_results:
        print (nll_res)
    plot(nll_results)
    return [W1, W2]


def plot(nlls):
    #with open("win2_nll.txt", 'r') as file:
    #    nlls = file.readlines()
    #    nlls = [float(line.strip()) for line in nlls]
    #    logging.debug(nlls)

    x_coordinate = [i for i in range(len(nlls))]
    plt.plot(x_coordinate[::50], nlls[::50])
    plt.show()

# Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
def load_model():
    handle = open("saved_W1_sz2_pos.data","rb")
    W1 = np.load(handle)
    handle.close()
    handle = open("saved_W2_sz2_pos.data","rb")
    W2 = np.load(handle)
    handle.close()
    return [W1, W2]


# Save the current results to an output file. Useful when computation is taking a long time.
def save_model(W1,W2):
    handle = open("saved_W1_sz2_pos.data","wb+")
    np.save(handle, W1, allow_pickle=False)
    handle.close()

    handle = open("saved_W2_sz2_pos.data","wb+")
    np.save(handle, W2, allow_pickle=False)
    handle.close()

# so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
# of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
# vector predict similarity to a context word.


# code to start up the training function.
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
    global word_embeddings, proj_embeddings
    if preload:
        [curW1, curW2] = load_model()
    else:
        curW1 = None
        curW2 = None
    [word_embeddings, proj_embeddings] = trainer(curW1, curW2)
    save_model(word_embeddings, proj_embeddings)

def cosine_sim(vec1, vec2):
    #numerator = np.dot(vec1, vec2)
    #norm_vec1 = np.linalg.norm(vec1)
    #norm_vec2 = np.linalg.norm(vec2)
    #result = numerator / (norm_vec1 * norm_vec2)
    result = 1 - spatial.distance.cosine(vec1, vec2)

    return result

def learn_morphology(morph_pairs):
    global word_embeddings, wordcodes
    morph_diff = []
    for pair in morph_pairs:
        logging.debug("var: {} idx: {} root:{} idx: {}".format(pair[1], wordcodes[pair[1]], pair[0], wordcodes[pair[0]]))
        morph_diff.append(np.subtract(word_embeddings[wordcodes[pair[1]]], word_embeddings[wordcodes[pair[0]]]))

    morph_mean = np.mean(morph_diff, axis=0)
    print("morph_mean: {}".format(morph_mean))
    return morph_mean


def get_morph_root(morph_var):
    morph_type = None
    root = None
    sz = len(morph_var) - 1
    if morph_var[sz-3:sz+1] == 'less':
        morph_type = 'less'
        root = morph_var[:sz-3]
    elif morph_var[sz] == 's':
        morph_type = 's'
        root = morph_var[:sz]
    elif morph_var[sz-2:sz+1] == 'ing':
        morph_type ='ing'
        root = morph_var[:sz-2]
    elif morph_var[0:2] == 'un':
        morph_type ='un'
        root = morph_var[2:]
    elif morph_var[sz-2:sz+1] == 'est':
        morph_type ='est'
        root = morph_var[:sz-2]
    else:
        print("no morph type found")

    logging.debug("morph_type: {} root: {}".format(morph_type, root))
    return [morph_type, root]


def get_or_impute_vector(morph_type, word_pair, morph_code, do_impute=True):
    global word_embeddings, wordcodes, uniqueWords
    root = word_pair[0]
    morph_var = word_pair[1]
    #     [morph_type, root] = get_morph_root(morph_var)
    if do_impute is False:
        idx = wordcodes.get(morph_var)
        if idx is not None:
            return word_embeddings[idx]

    if (morph_type is not None) and (root in uniqueWords):
        diff = morph_code[morph_type]
        impute_vec = diff + word_embeddings[wordcodes[root]]
        return impute_vec
    else:
        return None


# for the averaged morphological vector combo, estimate the new form of the target word
def morphology(morph_type, word_pair, morph_code, k):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    target = wordcodes[word_pair[0]] # technique idx
    imputed_vec = get_or_impute_vector(morph_type, word_pair, morph_code) # suffix averaged

    # find whichever vector is closest to vector_math
    # Use the same approach you used in function prediction() to construct a list
    # of top 10 most similar words to vector_math. Return this list.
    sim = [cosine_sim(imputed_vec, vec2) for vec2 in embeddings]
    morph_pred = sorted(range(len(sim)), key=lambda k: sim[k], reverse=True)[:k]
    morph_pred_str = list(map(lambda x: uniqueWords[x], morph_pred))
    print(morph_pred_str)

    #result = 1 if target in morph_pred else 0
    #logging.debug("root: {} pred: {} result: {}".format(word_pair[0], morph_pred_str, result))
    return morph_pred_str


# for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
def analogy(word_seq):
    global word_embeddings, proj_embeddings, uniqueWords, wordcodes
    embeddings = word_embeddings
    vectors = [embeddings[wordcodes[word_seq[0]]],
    embeddings[wordcodes[word_seq[1]]],
    embeddings[wordcodes[word_seq[2]]]]
    vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
    # find whichever vector is closest to vector_math
    # Use the same approach you used in function prediction() to construct a list
    # of top 10 most similar words to vector_math. Return this list.


# find top 10 most similar words to a target word
def prediction(target_word):
    global word_embeddings, uniqueWords, wordcodes

    # search through all uniqueWords and for each token, compute its similarity to target_word.
    # compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
    # Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
    # return a list of top 10 most similar words in the form of dicts,
    # each dict having format: {"word":<token_name>, "score":<cosine_similarity>}

    k = 10
    target_vec = word_embeddings[wordcodes[target_word]]

    sim = [cosine_sim(target_vec, word_embeddings[i]) for i in range(len(uniqueWords))]
    pred = sorted(range(len(sim)), key=lambda k: sim[k], reverse=True)[:k+1]
    pred_str = list(map(lambda x: {"word": uniqueWords[x], "score": sim[x]}, pred))

    logging.debug("target_word: {} pred_str: {}".format(target_word, pred_str[1:]))
    return pred_str[1:]

def write_output(result, columns, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(result)

def eval_morph():
    # ... try morphological task. Input is averages of vector combinations that use some morphological change.
    # ... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
    # ... the morphology() function.

    # Build test data
    morph_data = {}
    morph_data_s = [["type", "types"], ["ship", "ships"], ["value", "values"], ["wall", "walls"],
                    ["spoiler", "spoilers"],
                    ["prisoner", "prisoners"], ["work", "works"], ["player", "players"], ["car", "cars"],
                    ["arm", "arms"],
                    ["door", "doors"], ["test", "tests"], ["book", "books"], ["paper", "papers"], ["school", "schools"],
                    ["wheel", "wheels"], ["teacher", "teachers"], ["student", "students"], ["setting", "settings"],
                    ["truck", "trucks"]]
    morph_data['s'] = morph_data_s
    morph_data_ing = [["spread", "spreading"], ["script", "scripting"], ["sing", "singing"], ["draw", "drawing"],
                      ["see", "seeing"],
                      ["find", "finding"], ["buy", "buying"], ["push", "pushing"], ["play", "playing"],
                      ["meet", "meeting"],
                      ["depict", "depicting"], ["stay", "staying"], ["park", "parking"], ["kill", "killing"],
                      ["pull", "pulling"],
                      ["watch", "watching"], ["approach", "approaching"], ["help", "helping"], ["build", "building"],
                      ["direct", "directing"]]
    morph_data['ing'] = morph_data_ing
    morph_data_less = [["meaning", "meaningless"], ["law", "lawless"], ["regard", "regardless"], ["use", "useless"],
                       ["flaw", "flawless"],
                       ["hope", "hopeless"], ["home", "homeless"], ["price", "priceless"], ["aim", "aimless"],
                       ["rest", "restless"],
                       ["harm", "harmless"], ["brain", "brainless"], ["count", "countless"], ["regard", "regardless"],
                       ["end", "endless"],
                       ["speech", "speechless"], ["seam", "seamless"], ["need", "needless"], ["thank", "thankless"],
                       ["wit", "witless"]]
    morph_data['less'] = morph_data_less
    morph_data_un = [["real", "unreal"], ["known", "unknown"], ["credited", "uncredited"], ["seen", "unseen"],
                     ["fair", "unfair"],
                     ["satisfying", "unsatisfying"], ["comfortable", "uncomfortable"], ["even", "uneven"],
                     ["expected", "unexpected"], ["realistic", "unrealistic"],
                     ["conventional", "unconventional"], ["clear", "unclear"], ["rated", "unrated"],
                     ["explained", "unexplained"], ["developed", "undeveloped"],
                     ["happy", "unhappy"], ["flinching", "unflinching"], ["familiar", "unfamiliar"],
                     ["believable", "unbelievable"], ["pleasant", "unpleasant"]]
    morph_data['un'] = morph_data_un
    morph_data_est = [["dark", "darkest"], ["high", "highest"], ["low", "lowest"], ["fine", "finest"],
                      ["dumb", "dumbest"],
                      ["old", "oldest"], ["young", "youngest"], ["deep", "deepest"], ["stupid", "stupidest"],
                      ["strong", "strongest"],
                      ["weak", "weakest"], ["slight", "slightest"], ["hard", "hardest"], ["long", "longest"],
                      ["strange", "strangest"],
                      ["fine", "finest"], ["easy", "easiest"], ["funny", "funniest"], ["close", "closest"],
                      ["late", "latest"]]
    morph_data['est'] = morph_data_est
    data_len = 20
    morph_train_data = {key: data[:int(data_len * 0.8)] for key, data in morph_data.items()}
    morph_test_data = {key: data[int(data_len * 0.9):] for key, data in morph_data.items()}

    morph_code = {key: learn_morphology(data) for key, data in morph_train_data.items()}

    # evaluate at k precision
    k = 10
    prec_k = {i: [] for i in range(1, k + 1)}
    morph_result = []
    for morph_type, data in morph_test_data.items():
        targets = [pair[0] for pair in data]
        morph_preds = [morphology(morph_type, pair, morph_code, k) for pair in data]
        for idx, pred in enumerate(morph_preds):
            logging.debug("root: {} pred: {}".format(targets[idx], pred))
            for i in range(1, k + 1):
                prec_k[i].append(1 if targets[idx] in pred[:i] else 0)

    prec_k_avg = [sum(morph_result) / len(morph_result) for morph_result in prec_k.values()]
    logging.debug("prec_k_avg: {}".format(prec_k_avg))
    # x_coordinate = [i + 1 for i in range(len(prec_k_avg))]
    # plt.plot(x_coordinate, prec_k_avg)
    # plt.title("Morphology Precision with Window of 2")
    # plt.show()


def eval_pred():
    # ... we've got the trained weight matrices. Now we can do some predictions
    window = 4
    targets = ["good", "bad", "scary", "funny"]
    pred_flattened = []
    for targ in targets:
        bestpreds = prediction(targ)
        for pred in bestpreds:
            pred_flattened.append([targ, pred["word"], pred["score"]])
        print ("\n")
    filename = "p8_output_sz2_pos.txt" if window is 2 else "p9_output_sz2_pos.txt"
    write_output(pred_flattened, ['target_word', 'similar_word', 'similar_score'], filename)


def eval_intrinsic(test_data):
    global word_embeddings, wordcodes
    eval = [[data[0], cosine_sim(word_embeddings[wordcodes[data[1]]], word_embeddings[wordcodes[data[2]]])]
            for data in test_data]
    logging.debug("eval: {}".format(eval))
    write_output(eval, ['id', 'similarity'], "intrinsic-test_sz2_pos.csv")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    nltk.download('stopwords')
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        # load in the file, tokenize it and assign each token an index.
        # the full sequence of characters is encoded in terms of their one-hot positions

        fullsequence = loadData(filename)
        print ("Full sequence loaded...")
        #print(uniqueWords)
        #print (len(uniqueWords))

        # now generate the negative sampling table
        print ("Total unique words: ", len(uniqueWords))
        print("Preparing negative sampling table")
        samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)

        # we've got the word indices and the sampling table. Begin the training.
        #train_vectors(preload=False)

        [word_embeddings, proj_embeddings] = load_model()

        eval_morph()

        eval_pred()

        test_data = load_test("intrinsic-test_v2.tsv")
        eval_intrinsic(test_data)

    else:
        print("Please provide a valid input filename")
        sys.exit()


