from collections import Counter, defaultdict
from itertools import count
import random
import dynet as dy
import numpy as np

DEBUG = False
HASLM = False
LM_EPOCH = 5
TAG_EPOCH = 10

# CRF parameters
MAX_LIK_ITERS = 3
SMALL_NUMBER = -1e10
MARGIN = 0

lm_train_file = "LM_TRAIN"
lm_test_file = "LM_DEV"
train_file = "TAG_TRAIN"
dev_file = "TAG_DEV"

if DEBUG:
    lm_train_file += "_SMALL"
    lm_test_file += "_SMALL"
    train_file += "_SMALL"
    dev_file += "_SMALL"

# Language Model
print "--------Language Model Training--------"

def read_lm(fname):
    with file(fname) as fh:
        for line in fh:
            sent = line.strip().split()
            sent.append("<s>")
            yield sent

lm_train = list(read_lm(lm_train_file))
lm_test = list(read_lm(lm_test_file))
lm_words = []

for sent in lm_train:
    for w in sent:
        lm_words.append(w)

lm_words.append("_UNK_")

lm_w2i = defaultdict(count(0).next)
for word in lm_words:
    lm_w2i[word]
lm_i2w = {i:w for w, i in lm_w2i.iteritems()}

lm_nwords = len(lm_w2i)
lm_model = dy.Model()
lm_trainer = dy.AdamTrainer(lm_model)
lm_WORDS_LOOKUP = lm_model.add_lookup_parameters((lm_nwords, 64))
lm_RNN = dy.LSTMBuilder(1, 64, 128, lm_model)
lm_pW = lm_model.add_parameters((lm_nwords, 128))
lm_pb = lm_model.add_parameters(lm_nwords)

def calc_lm_loss(sent):
    dy.renew_cg()
    W = dy.parameter(lm_pW)
    b = dy.parameter(lm_pb)
    f_init = lm_RNN.initial_state()

    wids = []
    for w in sent:
        if w in lm_words:
            wids.append(lm_w2i[w])
        else:
            wids.append(lm_w2i["_UNK_"])

    s = f_init.add_input(lm_WORDS_LOOKUP[wids[-1]])

    losses = []
    for wid in wids:
        score = W * s.output() + b
        loss = dy.pickneglogsoftmax(score, wid)
        losses.append(loss)
        s = s.add_input(lm_WORDS_LOOKUP[wid])
    
    return dy.esum(losses)

def calc_lm_embdding(words):
    dy.renew_cg()
    f_init = lm_RNN.initial_state()
    wids = []
    for w in words:
        if w in lm_words:
            wids.append(lm_w2i[w])
        else:
            wids.append(lm_w2i["_UNK_"])
    wids.append(lm_w2i["<s>"])
    s = f_init.add_input(lm_WORDS_LOOKUP[wids[-1]])
    outputs = []
    for wid in wids:
        outputs.append(s.output().value())
        s = s.add_input(lm_WORDS_LOOKUP[wid]) 
    return outputs

if HASLM:
    for ITER in xrange(LM_EPOCH):
        lm_num_tagged = lm_cum_loss = 0
        random.shuffle(lm_train)
        i = 0
        for s in lm_train:
            loss_exp = calc_lm_loss(s)
            lm_cum_loss += loss_exp.scalar_value()
            lm_num_tagged += len(s)
            loss_exp.backward()
            lm_trainer.update()
            if DEBUG == False:
                i += 1
                if i % 100 == 0:
                    print "train loss ", i, ": ", lm_cum_loss / lm_num_tagged

        dev_loss = dev_words = 0
        i = 0
        for sent in lm_test:
            loss_exp = calc_lm_loss(sent)
            dev_loss += loss_exp.scalar_value()
            dev_words += len(sent)
            # if DEBUG == False:
            #     i += 1
            #     if i % 100 == 0:
            #         print "dev loss ", i, ": ", dev_loss / dev_words
        print "epoch %r finished" % ITER
        print "total train loss: ", lm_cum_loss / lm_num_tagged
        print "total dev loss: ", dev_loss / dev_words

# Tagger
print "--------Sequence Tagger Training--------"

def read(fname):
    with file(fname) as fh:
        for line in fh:
            line = line.strip().split()
            sent = [tuple(x.rsplit("/", 1)) for x in line]
            yield sent

train = list(read(train_file))
dev = list(read(dev_file))
words = []
tags = []
chars = set()
wc = Counter()
for sent in train:
    for w, p in sent:
        words.append(w)
        tags.append(p)
        chars.update(w)
        wc[w] += 1

words.append("_UNK_")
words.append("_S_")
tags.append("_S_")
chars.add("<*>")

w2i = defaultdict(count(0).next)
for word in words:
    w2i[word]
i2w = {i:w for w, i in w2i.iteritems()}

t2i = defaultdict(count(0).next)
for tag in tags:
    t2i[tag]
i2t = {i:w for w, i in t2i.iteritems()}

c2i = defaultdict(count(0).next)
for char in chars:
    c2i[char]
i2c = {i:w for w, i in c2i.iteritems()}

UNK = w2i["_UNK_"]
S_W = w2i["_S_"]
S_T = t2i["_S_"]

nwords = len(w2i)
ntags  = len(t2i)
nchars  = len(c2i)

model = dy.Model()
trainer = dy.AdamTrainer(model)
WORDS_LOOKUP = model.add_lookup_parameters((nwords, 128))
CHARS_LOOKUP = model.add_lookup_parameters((nchars, 20))
TRANS_LOOKUP = model.add_lookup_parameters((ntags, ntags))
pH = model.add_parameters((32, 50*2))
pO = model.add_parameters((ntags, 32))

fwdRNN = dy.LSTMBuilder(2, 128, 50, model)
bwdRNN = dy.LSTMBuilder(2, 128, 50, model)
if HASLM:
    fwdRNN = dy.LSTMBuilder(2, 256, 50, model)
    bwdRNN = dy.LSTMBuilder(2, 256, 50, model)

cFwdRNN = dy.LSTMBuilder(1, 20, 64, model)
cBwdRNN = dy.LSTMBuilder(1, 20, 64, model)

def word_rep(w, cf_init, cb_init):
    if wc[w] > 5:
        w_index = w2i[w]
        return WORDS_LOOKUP[w_index]
    else:
        pad_char = c2i["<*>"]
        char_ids = [pad_char] + [c2i[c] for c in w] + [pad_char]
        char_embs = [CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

def build_tagging_graph(words):
    lm_wembs = []
    if HASLM:
        lm_wembs = calc_lm_embdding(words)

    dy.renew_cg()
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()
    cf_init = cFwdRNN.initial_state()
    cb_init = cBwdRNN.initial_state()

    wembs = [word_rep(w, cf_init, cb_init) for w in words]
    if HASLM:
        wembs1 = []
        for lmw, w in zip(lm_wembs, wembs):
            wv = w.value()
            wv.extend(lmw)
            wembs1.append(wv)
        wembs = [dy.inputTensor(w) for w in wembs1]
    wembs = [dy.noise(we, 0.1) for we in wembs]

    fw_exps = f_init.transduce(wembs)
    bw_exps = b_init.transduce(reversed(wembs))
    bi_exps = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

    exps = []
    for x in bi_exps:
        r_t = O * (dy.tanh(H * x))
        exps.append(r_t)

    return exps

def viterbi_decoding(vecs, gold_tags = []):
    # Initialize
    init_prob = [SMALL_NUMBER] * ntags
    init_prob[S_T] = 0
    for_expr = dy.inputVector(init_prob)
    best_ids = []
    trans_exprs = [TRANS_LOOKUP[tid] for tid in range(ntags)]
    # Perform the forward pass through the sentence
    for i, vec in enumerate(vecs):
        my_best_ids = []
        my_best_exprs = []
        for next_tag in range(ntags):
            # Calculate vector for single next tag
            next_single_expr = for_expr + trans_exprs[next_tag]
            next_single = next_single_expr.npvalue()
            # Find and save the best score
            my_best_id = np.argmax(next_single)
            my_best_ids.append(my_best_id)
            my_best_exprs.append(dy.pick(next_single_expr, my_best_id))
        # Concatenate the scores for all vectors together
        for_expr = dy.concatenate(my_best_exprs) + vec
        # Give a bonus to all but the correct tag if using margin
        if MARGIN != 0 and len(gold_tags) != 0:
            adjust = [MARGIN] * ntags
            adjust[t2i[gold_tags[i]]] = 0
            for_expr = for_expr + dy.inputVector(adjust)
        # Save the best ids
        best_ids.append(my_best_ids)
    # Perform the final step to the sentence terminal symbol
    next_single_expr = for_expr + trans_exprs[S_T]
    next_single = next_single_expr.npvalue()
    my_best_id = np.argmax(next_single)
    best_expr = dy.pick(next_single_expr, my_best_id)
    # Perform the reverse pass
    best_path = [i2t[my_best_id]]
    for my_best_ids in reversed(best_ids):
        my_best_id = my_best_ids[my_best_id]
        best_path.append(i2t[my_best_id])
    best_path.pop() # Remove final <s>
    best_path.reverse()
    # Return the best path and best score as an expression
    return best_path, best_expr

def forced_decoding(vecs, tags):
    # Initialize
    for_expr = dy.scalarInput(0)
    for_tag = S_T
    # Perform the forward pass through the sentence
    for i, vec in enumerate(vecs): 
        my_tag = t2i[tags[i]]
        for_expr = for_expr + dy.pick(TRANS_LOOKUP[my_tag], for_tag) + vec[my_tag]
        for_tag = my_tag
    for_expr = for_expr + dy.pick(TRANS_LOOKUP[S_T], for_tag)
    return for_expr

def viterbi_sent_loss(words, tags):
    vecs = build_tagging_graph(words)
    viterbi_tags, viterbi_score = viterbi_decoding(vecs, tags)
    if viterbi_tags != tags:
        reference_score = forced_decoding(vecs, tags)
        return viterbi_score - reference_score
    else:
        return dy.scalarInput(0)

def sent_loss(words, tags):
    vecs = build_tagging_graph(words)
    errs = []
    for v,t in zip(vecs,tags):
        tid = t2i[t]
        err = dy.pickneglogsoftmax(v, tid)
        errs.append(err)
    return dy.esum(errs)

def tag_sent(words):
    vecs = build_tagging_graph(words)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in probs:
        tag = np.argmax(prb)
        tags.append(i2t[tag])
    return tags

for ITER in xrange(TAG_EPOCH):
    num_tagged = cum_loss = 0
    random.shuffle(train)
    i = 0
    for s in train:
        words = [w for w, t in s]
        golds = [t for w, t in s]

        if ITER < MAX_LIK_ITERS:
            loss_exp =  sent_loss(words, golds)
        else:
            loss_exp =  viterbi_sent_loss(words, golds)
        cum_loss += loss_exp.scalar_value()
        num_tagged += len(golds)
        loss_exp.backward()
        trainer.update()
        if DEBUG == False:
            i += 1
            if i % 1000 == 0:
                print "train loss ", i, ": ", cum_loss / num_tagged

    good_sent = bad_sent = good = bad = 0.0
    i = 0
    for sent in dev:
        words = [w for w, t in sent]
        golds = [t for w, t in sent]
        if ITER < MAX_LIK_ITERS:
            tags = tag_sent(words)
        else:
            vecs = build_tagging_graph(words)
            tags, loss_exp = viterbi_decoding(vecs)
        if tags == golds: good_sent += 1
        else: bad_sent += 1
        for go, gu in zip(golds, tags):
            if go == gu: good += 1
            else: bad += 1
        # if DEBUG == False:
        #     i += 1
        #     if i % 1000 == 0:
        #         print "F1 ", i, ": ", good / (good + bad)
    print "epoch %r finished" % ITER
    print "total loss: ", cum_loss / num_tagged
    print "total F1: ", good / (good + bad), good_sent / (good_sent + bad_sent)

def tagging(sentence):
    words = sentence.strip().split()
    if TAG_EPOCH <= MAX_LIK_ITERS:
        tags = tag_sent(words)
    else:
        vecs = build_tagging_graph(words)
        tags, loss_exp = viterbi_decoding(vecs)
	for w, t in zip(words, tags):
		print w, t

if __name__ == '__main__':
	sentence = "if you could see that i am the one who understands you ."
	tagging(sentence)
