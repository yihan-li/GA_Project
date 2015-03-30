from __future__ import division

import sqlite3
import pandas
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

TRAINING_SET = 'data/train.csv'
TEST_SET = 'data/test.csv'
RESULTS = 'data/submission.csv'

def get_card(row, c):
    return int(row['C'+str(c)])

def set_card(row, c, card):
    row['C'+str(c)] = card

def get_suit(row, c):
    return int(row['S'+str(c)])

def set_suit(row, c, suit):
    row['S'+str(c)] = suit

def sort_row(row):
    cards = []
    for c in range(1, 6):
        card = get_card(row, c)
        suit = get_suit(row, c)
        # converts card (1-13) and suit (1-4) to a universal card id (0-51)
        card_id = (card-1)*4 + (suit-1)
        # card_id = ((card-2)%13)*4 + (suit-1)
        cards.append(card_id)
    cards.sort()
    for c in range(1, 6):
        card_id = cards[c-1]
        # converts a universal card id (0-51) to card (1-13) and suit (1-4)
        card = int(card_id / 4) + 1
        # card = (int(card_id/4)+1)%13 + 1
        suit = card_id % 4 + 1
        set_card(row, c, card)
        set_suit(row, c, suit)

# load training data and sort rows
df = pandas.read_csv(TRAINING_SET, index_col=False, header=0)
for index, row in df.iterrows():
    sort_row(row)

def suit_count_attr(s):
    return '#S'+str(s)

def card_count_attr(c):
    return '#C'+str(c)

def diff_cards_attr(c):
    return 'diff'+str((c-1)%5+1)+str(c%5+1)

# create additional attributes
for c in range(1, 6):
    df[diff_cards_attr(c)] = 0
for s in range(1, 5):
    df[suit_count_attr(s)] = 0
for c in range(1, 14):
    df[card_count_attr(c)] = 0

for index, row in df.iterrows():
    for c in range(1, 6):
        suit = get_suit(row, c)
        row[suit_count_attr(suit)] = row[suit_count_attr(suit)] + 1
        card = get_card(row, c)
        row[diff_cards_attr(c)] = card - get_card(row, c%5+1)
        row[card_count_attr(card)] = row[card_count_attr(card)] + 1

# define response and explanatory series
response_df = df.hand
explanatory_features = [col for col in df.columns if col not in ['hand']]
explanatory_df = df[explanatory_features]

clf_extra = RandomForestClassifier(
    n_estimators = 100,
    criterion = 'entropy',
    max_features = None,
    max_depth = None,
    min_samples_split = 2,
    min_samples_leaf = 2,
    max_leaf_nodes = None,
    random_state = 400)

# train model
clf_extra = clf_extra.fit(explanatory_df, response_df)
score = clf_extra.score(explanatory_df, response_df)
print "Training score: %.2f%%" % (score * 100)

# validate model
# scores = cross_val_score(
#     clf_extra,
#     explanatory_df,
#     response_df,
#     cv = 10,
#     scoring = 'accuracy',
#     n_jobs = -1)
# mean_accuracy = numpy.mean(scores) 
# print "Validation score: %.2f%%" % (mean_accuracy * 100)

#Test data
test_df = pandas.read_csv(TEST_SET, index_col=False, header=0)
for index, row in test_df.iterrows():
    sort_row(row)

# create additional attributes
for c in range(1, 6):
    test_df[diff_cards_attr(c)] = 0
for s in range(1, 5):
    test_df[suit_count_attr(s)] = 0
for c in range(1, 14):
    test_df[card_count_attr(c)] = 0

for index, row in test_df.iterrows():
    for c in range(1, 6):
        suit = get_suit(row, c)
        row[suit_count_attr(suit)] = row[suit_count_attr(suit)] + 1
        card = get_card(row, c)
        row[diff_cards_attr(c)] = card - get_card(row, c%5+1)
        row[card_count_attr(card)] = row[card_count_attr(card)] + 1

features = [col for col in test_df.columns if col not in ['id']]
test_df = test_df[features]
f = open(RESULTS, 'w')
f.write('id,hand\n')
for index, row in test_df.iterrows():
    f.write('%d,%d\n' % (index+1, clf_extra.predict(row)[0]))
