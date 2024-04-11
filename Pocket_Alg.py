import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def initialize_weights(X, y, method):
    if (method == "method_1"):
        weights = X[0].copy() # using first data point
    elif (method == "method_2"):
        weights = np.linalg.pinv(X).dot(y) # using linear regression
    
    return weights

def Pocket_Alg_training(X, y, nr_iterations, weight):
    N, attributes = X.shape
    weights = weight.copy() #intitiliazing the weights based on the method
    best_weights = np.zeros(attributes) #
    N = X.shape[0] # total data points
    smallest_Ein = 1.0 # smallest Ein the worst Ein or the biggest 
    Ein_list = []

    for i in range(nr_iterations):
        prediction = np.sign(np.dot(X,weights.T))
        mistake_index = np.where(prediction != y)[0] #storing indexes of mistakes
        
        if (len(mistake_index) == 0):          #no mistakes break the loop 
            break

        random_index = np.random.choice(mistake_index, 1)[0]
        weights += X[random_index] * y[random_index] # update/correct weights

        Ein = len(mistake_index) / N  
        Ein_list.append(Ein)  # storing Ein of each iteration for ploting purposes
        
        if (Ein < smallest_Ein):         #if smaller mistake, improve my hipothesis
            smallest_Ein = Ein
            best_weights = weights.copy()
    avg_Ein_per_nr_Iterations = np.mean(Ein_list)
    return best_weights, Ein_list, avg_Ein_per_nr_Iterations

def Pocket_Alg_train_test(X_train,X_test, y_train,y_test, nr_iterations, method):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=0)

   
    weights = initialize_weights(X_train, y_train, method) #initialize weights based on method

    best_weights, Ein_list, avg_Ein_per_nr_Iterrations = Pocket_Alg_training(X_train, y_train, nr_iterations, weights) # training

    prediction = np.sign(np.dot(X_test,best_weights.T)) # testing
    Eout = 1 - (np.mean(prediction == y_test))  # 1 -  mean accuracy
   
    return Eout, Ein_list, nr_iterations, best_weights, avg_Ein_per_nr_Iterrations


# ######################################################################################

if __name__ == "__main__":

    data = load_breast_cancer()
    # print(data.DESCR)
    X = data.data
    y = 2 * data.target - 1  # Setting the labels to -1 and 1

    nr_iterations = 300
    size = 0.3
   
    methods = ["method_1", "method_2"]
    avg_Ein_method_1 = [] # will store 5 avarages( 1 per split)
    avg_Ein_method_2 = []
    Eout_method_1 = []  # will store 5 Eouts (1 per split)
    Eout_method_2 = []
    total_Ein_avarage_method_1 = np.zeros(nr_iterations)  #total Ein avg of all splits
    total_Ein_avarage_method_2 = np.zeros(nr_iterations)
    total_Eout_avarage_method_1 = np.zeros(nr_iterations)  #total Eout avg of all splits
    total_Eout_avarage_method_2 = np.zeros(nr_iterations)

    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)  # 5 random splits

        for index, method in enumerate(methods):
            Eout, Ein_list,  nr_iteration,best_weights, avg_Ein_per_nr_Iterrations  = Pocket_Alg_train_test(X_train, X_test, y_train, y_test, nr_iterations, method)
            
            # Accumulate Ein values for averaging
            if (index == 0):
                avg_Ein_method_1.append(avg_Ein_per_nr_Iterrations) #  will be 5 avgs
                Eout_method_1.append(Eout)  # 5 Eout
                total_Ein_avarage_method_1 += np.array(Ein_list) # as many avarages as iterations over nr of splits
               
            else:
                avg_Ein_method_2.append(avg_Ein_per_nr_Iterrations)
                Eout_method_2.append(Eout)
                total_Ein_avarage_method_2 += np.array(Ein_list)
     
           
            
   
   # Values of avg lines for each method fig 2/3 and 3/3
    total_Ein_avarage_method_1 /= 5
    avarage_line_Ein_1 =np.mean(total_Ein_avarage_method_1) #for plotting
    avarage_line_Eout_1 = np.mean(Eout_method_1) #for plotting
    total_Ein_avarage_method_2 /= 5
    avarage_line_Ein_2 =np.mean(total_Ein_avarage_method_2) #for plotting
    avarage_line_Eout_2 = np.mean(Eout_method_2)#for plotting
    
  
    # 1) Plot General Preformance for Both Methods
    plt.figure(figsize=(12, 6))
    plt.plot( range(1,6), avg_Ein_method_1, label=f'5 Ein Avarages on {nr_iterations} Iterations (One Each Split)(method_1)', marker='o',markerfacecolor = 'black', color='blue')
    plt.plot(range(1,6), avg_Ein_method_2, label=f'5 Ein Avarages on {nr_iterations} Iterations (One Each Split) (method_2)', marker='.',markerfacecolor = 'black', color='red')
    plt.plot(range(1,6), Eout_method_1, label='5 Eout (One Each Split) (method_1)', marker='o',markerfacecolor = '#b511a7', color='#b511a7')
    plt.plot(range(1,6),  Eout_method_2, label='5 Eout (One Each Split) (method_2)', marker='.',markerfacecolor = 'black', color='orange')
    plt.xlabel('Random Splits')
    plt.ylabel('Error')
    plt.title(f'Fig#1/3 General Performance On 5 Splits For Both Methods||(Split-Size: {size})')
    plt.legend()
    plt.grid(False)
    plt.savefig('General_Performance.png')
    #plt.show()
     
    #Plot For Method 1
    plt.figure(figsize=(12, 6))
    plt.plot(range(nr_iterations),total_Ein_avarage_method_1, label='AVG Ein Over 5 Splits', marker='.', color='#36a0fb')
    plt.axhline(y=avarage_line_Ein_1, color='red', linestyle='-', label='Ein Total AVG')
    plt.axhline(y=avarage_line_Eout_1, color='orange', linestyle='-', label='Eout Total AVG')
    plt.xlabel('NR Iterations')
    plt.ylabel('Error')
    plt.title(f'Fig#2/3 Perfomance Method 1 (First Data Point)||(Split-Size: {size}) ')
    plt.legend()
    plt.grid(True)
    plt.savefig('Method_1.png')
    # plt.show()

    # Plot for Method 2
    plt.figure(figsize=(12, 6))
    plt.plot(range(nr_iterations),total_Ein_avarage_method_2, label='AVG Ein Over 5 Splits', marker='.', color='#fc84f2')
    plt.axhline(y=avarage_line_Ein_2, color='#dc5806', linestyle='-', label='Ein Total AVG')
    plt.axhline(y=avarage_line_Eout_2, color='#cabb06', linestyle='-', label='Eout Total AVG')
    plt.xlabel('NR Iterations')
    plt.ylabel('Error')
    plt.title(f'Fig#3/3 Perfomance Method 2 (Linear Regression)||(Split-Size: {size}) ')
    plt.legend()
    plt.grid(True)
    plt.savefig('Method_2.png')
    # plt.show()

    