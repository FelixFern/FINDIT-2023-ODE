from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, X, y, X_test, depth=2):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.depth = depth
        self.clf = RandomForestClassifier(max_depth=self.depth, random_state=0)

    def fit(self):
        self.model = self.clf.fit(self.X, self.y)

    def predict(self):
        result = self.clf.predict(self.X_test)
        return result
