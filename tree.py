from helpers import find_best_question, LeafNode, DecisionNode
class DecisionTree:
    def __init__(self):
        self.root = None

    def train(self, train):
        '''Recursive function to establish full tree, returns root node'''
        info_gain, question = find_best_question(train)

        if info_gain == 0:
            return LeafNode(train)

        t_branch, f_branch = question.ask(train)
        
        return DecisionNode(question, self.train(t_branch), self.train(f_branch))

    def fit(self, train):
        '''Calls the train function to set the tree's root node'''
        self.root = self.train(train)
    
    def get_pred(self, test, node):
        if isinstance(node, LeafNode):
            most_counts = 0
            pred = None
            for p in node.preds:
                if node.preds[p] > most_counts:
                    most_counts = node.preds[p]
                    pred = p
            return pred
        
        t, _ = node.question.ask([test])
        if len(t) > 0:
            return self.get_pred(test, node.t_branch)
        else:
            return self.get_pred(test, node.f_branch)

    def predict(self, test):
        if self.root:
            preds = []
            for t in test:
                preds.append(self.get_pred(t, self.root))
            return preds
        else:
            return "You must fit the model before getting predictions"

    def print_tree(self, node):
        if isinstance(node, LeafNode):
            return node.preds
        
        print(f't: {node.t_branch}, f: {node.f_branch}, {node.question}')
        return self.print_tree(node.t_branch), self.print_tree(node.f_branch)
    
    def tree_print(self):
        return self.print_tree(self.root)