import numpy as np
class matrix (object) :
    array_2d = np.array
    def __init__(self,file_name) :
        self.load_from_csv(file_name)
        self.n = self.array_2d.shape[0] #rows
        self.m = self.array_2d.shape[1] #columns
    def load_from_csv (self,file_name) :
        #Transfer csv to 2D array
        self.array_2d = np.loadtxt(file_name, delimiter=",")
    def standardise (self) :
        #standardise a matrix
        self.array_2d = (self.array_2d - np.mean(self.array_2d, axis=0))/np.std(self.array_2d, axis=0)
    def temp_matrix_order (self,order) :
        #To call specified array_2d rows or columns
        self.order = order
    def get_distance(self,other_matrix, weights,beta) :
        #Calculate weighted Euclidean distance
        distance_matrix = np.zeros([other_matrix.shape[0],1])
        array_row = self.array_2d[self.order]
        for i in range(other_matrix.shape[0]):
            for j in range(self.m) :
                distance_matrix[i,0] += (weights[0,j]**beta)*((array_row[j] - other_matrix[i,j])**2)
        return distance_matrix     
    def get_count_frequency (self) :
        #Counter
        counter = dict()
        if self.m == 1 :
            for row in self.array_2d :
                for value in row :
                    if value not in counter :
                        counter[value] = 1
                    else :
                        counter[value] = counter[value]+1
        return counter
def get_initial_weights (m) :
    #To get the initial weights
    weights = np.random.random([1,m])
    weights = weights / np.sum(weights)
    return weights
def get_centroids (m, S, K) :
    temp_sum = 0
    count = 0
    for val in range(S.shape[0]) :
        if K == S[val,0]:
            temp_sum += m.array_2d[K,val]
            count += 1
    return(temp_sum / count)
def get_groups(m,K,beta):
    brk = 0 # For deciding the stopping point
    #STEP 2
    weights = get_initial_weights (m.m)
    #STEP 3
    #for matrix centroid
    centroids = np.empty([K,m.m])
    #STEP 4
    S = np.zeros([m.n,1])
    #STEP 5 & 6
    random_rows = np.random.choice(m.n, size=K, replace=False) #selects K different values from n rows of data matrix
    centroids = m.array_2d[random_rows, : ]
    #STEP 7
    m.standardise()
    while True :
        for i in range(m.n):
            m.temp_matrix_order(i)
            get_distance_list =m.get_distance(centroids, weights, beta)
            temp = S[i]
            S[i] = np.argmin(get_distance_list)
            #STEP 8
            if S[i] == temp :
                brk = 1
                break
        #For leaving the loop and to return the S matrix
        if brk == 1:
            break
        #STEP 9
        for j in range(centroids.shape[1]) :
            centroids[K,j]= get_centroids(m,S,K)
        #STEP 10
        weights = get_new_weights(m,centroids, S)
    return S
def get_new_weights(m, centroids, S):
    #To update the weights
    delta_j = []
    new_weights = np.array
    for j in range(m.m):
        delta_j[j] = 0
        for k in range(centroids.shape[0]) :
            for i in range(m.m) :
                if S[i] == k :
                    u = 1
                else :
                    u = 0
                delta_j[j] += (u * (m.array_2d[i,j] - centroids[k.j])**2)
    for j in range(self.m) :
        if delta_j[j] == 0:
            new_weights[j] = 0
        else :
            summation = 0
            for t in range(m.m) :
                summation += ((delta_j[j] / delta_j[t])**(1/ beta - 1))
            new_weights[j] = (1 / summation ) 
    return new_weights
def run_test():
    m = matrix('Data.csv')
    for k in range(2,5):
        for beta in range(11,25):
            S = get_groups(m, k, beta/10)
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))
