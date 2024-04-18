# ============================================================================ #
#                               DECISION TREE                                  #
# ============================================================================ #
# Author: Miguel Marines


# ============================================================================ #
#                                  LIBRARIES                                   #
# ============================================================================ #
from sklearn.tree import DecisionTreeClassifier                                # Works with trees.
from sklearn import tree                                                       # Works with trees.
import graphviz                                                                # Creates decision tree graph.
import pandas as pd                                                            # Helps import dataset locally.


# ============================================================================ #
#                               DATASET MANUALLY                               #
# ============================================================================ #
columns = ["PT","PED","SC","NI", "CS", "AT"]                                   # Columns names.
market = pd.read_csv('market.data', names = columns)                           # Imports dataset.

X = market[["PT","PED","SC","NI", "CS"]]                                       # Divides the features from the dataset.
y = market[["AT"]]                                                             # Divides the results from the dataset.

print("\nMARKET DATASET:")                                                     # Prints dataset.
print(market)                                                                  # Prints dataset.


# ============================================================================ #
#                                TREE CREATION                                 #
# ============================================================================ #
tree_clf = DecisionTreeClassifier(max_depth = 6)                               # Determines the maximum depth of the tree.
tree_clf.fit(X,y)                                                              # Creates tree.


# ============================================================================ #
#                                CLASSIFICATIONS                               #
# ============================================================================ #
print("\nTree Classiefier Configuration")
print("Class 1: Exportt")
print("Class 2: Import")
print("Class 3: Reserve")
print("Class 4: Same-Market")
print("Class 5: Subsidy")


# ============================================================================ #
#      PROBABILITY AND PREDICTIONS OF PRODUCTS OF BELONGING TO A CLASS         #
# ============================================================================ #
# Egg (PT = 0.29, PED = 1.50, SC = 1, NI = 1, CS = -0.45)
print("\nEgg:")
probability = tree_clf.predict_proba([[0.29, 1.50, 1, 1, -0.45]])
print("Class Probability: ",[[0.29, 1.50, 1, 1, -0.45]], probability)

prediction = tree_clf.predict([[0.29, 1.50, 1, 1, -0.45]])
print("Class Prediction: ",[[0.29, 1.50, 1, 1, -0.45]], prediction)


# Gas (PT = 0.51, PED = 0.15, SC = 1, NI = 1, CS = -0.25)
print("\nGas:")
probability = tree_clf.predict_proba([[0.51, 0.15, 1, 1, -0.25]])
print("Class Probability: ",[[0.51, 0.15, 1, 1, -0.25]], probability)

prediction = tree_clf.predict([[0.51, 0.15, 1, 1, -0.25]])
print("Class Prediction: ",[[0.51, 0.15, 1, 1, -0.25]], prediction)


# Water (PT = 0.54, PED = 0.00, SC = 1, NI = 1, CS = 0.10)
print("\nWater:")
probability = tree_clf.predict_proba([[0.54, 0.00, 1, 1, 0.10]])
print("Class Probability: ",[[0.54, 0.00, 1, 1, 0.10]], probability)

prediction = tree_clf.predict([[0.54, 0.00, 1, 1, 0.10]])
print("Class Prediction: ",[[0.54, 0.00, 1, 1, 0.10]], prediction)


# Milk Powder (PT = 0.09, PED = 1.65, SC = 1, NI = 0, CS = 0.0)
print("\nMilk Powder:")
probability = tree_clf.predict_proba([[0.09, 1.65, 1, 0, 0.0]])
print("Class Probability: ",[[0.09, 1.65, 1, 0, 0.0]], probability)

prediction = tree_clf.predict([[0.09, 1.65, 1, 0, 0.0]])
print("Class Prediction: ",[[0.09, 1.65, 1, 0, 0.0]], prediction)


# Random
print("\nRandom:")
probability = tree_clf.predict_proba([[0.62, 0.25, 1, 0, -0.30]])
print("Class Probability: ",[[0.62, 0.25, 1, 0, -0.30]], probability)

prediction = tree_clf.predict([[0.62, 0.25, 1, 0, -0.30]])
print("Class Prediction: ",[[0.62, 0.25, 1, 0, -0.30]], prediction)


# ============================================================================ #
#                             TREE GRAPH CREATION                              #
# ============================================================================ #

# Graph description language dot
dot = tree.export_graphviz(tree_clf, out_file = None, feature_names = ["PT","PED","SC","NI", "CS"], class_names = ["Import", "Export", "Same-Market", "Subsidy", "Reserve"], filled = True, rounded = True, special_characters = True)

# Create Tree Graph with graphviz
graph = graphviz.Source(dot)
graph