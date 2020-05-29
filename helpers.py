from numpy.random import random
def counts(rows):
    '''Returns counts of each label in rows'''
    counts = {}
    for row in rows:
        if row[-1] not in counts:
            counts[row[-1]] = 0
        counts[row[-1]] += 1
    return counts

def gini(rows):
    '''Returns gini impurity of given rows'''
    label_counts = counts(rows)
    total = len(rows)
    gini = 1
    for i in label_counts:
        gini -= (label_counts[i]/total)**2
    return gini

def gain(t, f, uncertainty):
    '''
    Takes the true/false branches and uncertainty of a decision node
    to return the information gain of a question *** Not actually information gain,
    just gini split.
    '''
    total = len(t)+len(f)
    pt = len(t)/total
    gain = uncertainty - (pt*gini(t) + (1-pt)*gini(f))
    return gain

def find_best_question(rows):
    ''' 
    Creates a question for each unique value in each column,
    returns the best information gain along with the corresponding question
    '''
    best_gain = 0
    best_question = None
    q_gain = 0
    for col in range(len(rows[0])-2):
        vals = set([row[col] for row in rows])
        for val in vals:
            q = Question(col, val)
            t_branch, f_branch = q.ask(rows)
            q_gain = gain(t_branch, f_branch, gini(rows))

            if q_gain > best_gain:
                best_gain = q_gain
                best_question = q
    return best_gain, best_question

class Question:
    def __init__(self, col, value):
        self.col = col
        self.value = value
    
    def __repr__(self):
        return f'col: {self.col}, val: {self.value}'

    def ask(self, rows):
        '''Partitions a list of rows into true and false branches'''
        t_branch = []
        f_branch = []
        for row in rows:
            value = row[self.col]
            if isinstance(value, int) or isinstance(value, float):
                if value >= self.value:
                    t_branch.append(row)
                else:
                    f_branch.append(row)
            else:
                # handles non-numeric/catergorical data
                if value == self.value:
                    t_branch.append(row)
                else:
                    f_branch.append(row) 
        return t_branch, f_branch
        
class DecisionNode:
    def __init__(self, question, t, f):
        self.question = question
        self.t_branch = t
        self.f_branch = f

class LeafNode:
    def __init__(self, rows):
        self.preds = counts(rows)