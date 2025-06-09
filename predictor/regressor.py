# # Importing libraries 
# import numpy as np 
# import pandas as pd 
# from sklearn.model_selection import train_test_split 
# import warnings 
# warnings.filterwarnings( "ignore" ) 
  
# # to compare our model's accuracy with sklearn model 
# from sklearn.linear_model import LogisticRegression 
# # Logistic Regression 
# class LogitRegression(): # Logistic Regression class
#     def __init__( self, learning_rate, iterations ) :       #Initialization of model   
#         self.learning_rate = learning_rate         
#         self.iterations = iterations 
          
#     # Function for model training     
#     def fit( self, X, Y ) :         
#         # no_of_training_examples, no_of_features         
#         self.m, self.n = X.shape         # Calculates shape of X matrix and save it in m and n
#         # weight initialization         
#         self.W = np.zeros( self.n )         
#         self.b = 0        
#         self.X = X         
#         self.Y = Y 
          
#         # gradient descent learning 
                  
#         for i in range( self.iterations ) :  # Looping on the basis of provided iterations            
#             self.update_weights()    # Calculating optimal value of W(weight) and b(bias).         
#         return self
      
    
      
#     def update_weights( self ) :     # Helper function to update weights in gradient descent 
#         A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) ) 
          
#         # calculate gradients         
#         tmp = ( A - self.Y.T )         
#         tmp = np.reshape( tmp, self.m )         
#         dW = np.dot( self.X.T, tmp ) / self.m    #Calculate dW (derivative)
#         db = np.sum( tmp ) / self.m  #Calculate db(derivative)
          
#         # update weights     
#         self.W = self.W - self.learning_rate * dW     
#         self.b = self.b - self.learning_rate * db 
          
#         return self
      
#     # Sigmoid function  h( x )  
      
#     def predict( self, X ) :     
#         Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )         
#         Y = np.where( Z > 0.5, 1, 0 )         
#         return Y , Z
  
  
  

# def find():  #To split the dataset into X and Y 
#     df = pd.read_csv("C:\\Users\\hp\Desktop\HDPS\predictor\heart.csv") 
#     X = df.iloc[:,:-1].values 
#     Y = df.iloc[:,-1:].values
#     return X,Y

    
  
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score , confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=5000, regularization=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for _ in range(self.iterations):
            self.update_weights()
        return self

    def update_weights(self):
        A = 1 / (1 + np.exp(-(self.X.dot(self.W) + self.b)))
        tmp = (A - self.Y.T).reshape(self.m)
        dW = (np.dot(self.X.T, tmp) / self.m) + (self.regularization * self.W)
        db = np.sum(tmp) / self.m

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict(self, X):
        Z = 1 / (1 + np.exp(-(X.dot(self.W) + self.b)))
        Y_pred = np.where(Z > 0.5, 1, 0)
        return Y_pred, Z

def load_data():
    # Use os.path to handle paths correctly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "heart.csv")
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, Y

def evaluate_model(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC: {roc_auc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def train_and_evaluate():
    # Load and preprocess data
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    model = LogisticRegression(learning_rate=0.1, iterations=5000, regularization=0.01)
    model.fit(X_train, Y_train)

    Y_pred, Y_prob = model.predict(X_test)
    evaluate_model(Y_test, Y_pred, Y_prob)
    return model

# Remove the automatic training code
if __name__ == '__main__':
    train_and_evaluate()


