import numpy as np 
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
import time

# Define converter
def label_converter(label):
    if label == b'b':
        return 0  # Mapping "b" to 0
    elif label == b'g':
        return 1  # Mapping "g" to 1
    else:
        return label

data = np.loadtxt("ionosphere.data.txt", delimiter=",", converters={34: label_converter})

features = data[: ,  :-1]
labels = data[ : , -1]

n_features = features.shape[1]



X_tv, X_test, y_tv, y_test = train_test_split(features, labels, random_state=42, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, random_state=42, test_size=0.3)
n_train = X_train.shape[0]
print(n_train)
print(y_test)


def train_weak_classifier(X, y, kernel='rbf', C=1.0, gamma='scale'):
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(X, y)
    return svm  # Return trained SVM model


def split_into_subsets(X, y, m):
    r = n_features + 1 # subset size d+1
    length = X.shape[0]
    
    # Create a list to hold the subsets
    subsets_X, subsets_y = [], []
    
    # Split the data into 'm' subsets
    for i in range(m):
        indexes = np.random.randint(length, size=r)
        subset_X = X[indexes]
        subset_y = y[indexes]
        subsets_X.append(subset_X)
        subsets_y.append(subset_y)
    
    return subsets_X, subsets_y


num_m = 20

m_values = [int((i/num_m)*n_train) for i in range(1, num_m + 1)]
m_scores = []
m_classifers = []
m_times = []

for m in m_values:

    startime = time.time() # monitor train times

    # Train multiple weak classifiers and validate them
    weak_classifiers = []
    scores = []
    X_subsets, y_subsets = split_into_subsets(X_train, y_train, m)
    
    for i in range(m):
        # Train weak classifier
        X_subset, y_subset= X_subsets[i], y_subsets[i]
        #print(y_subset)
        if 0 in y_subset and 1 in y_subset:
            weak_classifier = train_weak_classifier(X_subset, y_subset)
            preds = weak_classifier.predict(X_val)
            score = zero_one_loss(y_val, preds)
            weak_classifiers.append(weak_classifier)
            scores.append(score)

    m_scores.append(scores)
    m_classifers.append(weak_classifiers)

    deltatime = time.time() - startime
    m_times.append(deltatime)

#print(scores)
def calculate_rho(lambda_val, n_minus_r, L_val):
    n = len(L_val)
    pi_h = np.empty([n])
    pi_h.fill(1/n) # uniform distribution.
    min_val_loss = np.min(L_val)
    upperfracs = np.empty(n, dtype=np.float64)
    for i in range(n):
        # Compute the unnormalized rho for each hypothesis
        upperfrac = pi_h[i]*np.exp(-lambda_val * n_minus_r * (L_val[i] - min_val_loss))
        upperfracs[i] =  upperfrac

    lowerfrac = np.sum(upperfracs)
    rho = upperfracs / lowerfrac
    return rho


n_min_r = n_train - (n_features + 1)   
rhos = []
lamba = 0.5
for i in range(num_m):
    scores = m_scores[i]
    rho = calculate_rho(lamba, n_min_r, scores)
    rhos.append(rho)

#print(rhos)
#print(np.array(rhos))


    
def Majority_vote(predictions: np.array, rho: np.array):
    if (np.sum(predictions * rho)) >= 0.5:
        #print((np.sum(predictions * rho)))
        #print(rho)
        return 1
    else:
        return 0
    

n_test = X_test.shape[0]


# Predict with majority vote
majority_preds = []
for i in range(num_m):

    startime = time.time() # monitor inference times

    m_preds = np.empty(n_test)
    m_hyp = m_classifers[i]
    rho = rhos[i]
    #print(rho)
    hyp_len = len(m_hyp)
    for j in range(n_test):
        preds = np.empty(hyp_len)
        for k in range(hyp_len):
            hyp = m_hyp[k]
            pred = hyp.predict(X_test[j].reshape(1, -1))
            preds[k] = pred
        #print(preds)
        majority_pred = Majority_vote(preds, rho)
        m_preds[j] = majority_pred
    majority_preds.append(m_preds)

    deltatime = time.time() - startime
    m_times[i] += deltatime



def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def Expectation(L, p):
    return sum(L * p)

def pac_bound(lambda_val, n_minus_r, L, rho):
    n = len(L)
    pi_h = np.empty([n])
    pi_h.fill(1/n) # uniform distribution.
    first = Expectation(L, rho)/(1-(lambda_val/2))
    secnd = (kl_divergence(rho, pi_h) + np.log((n_minus_r+1)/lambda_val)) / (lambda_val*(1-lamba/2)*n_minus_r)
    bound = first + secnd
    return bound


# Score majority votes
losses = []
bound_losses = []
for i in range(num_m):
    m = m_values[i]
    y_pred = majority_preds[i]
    
    # Convert predictions to integer type if needed (depending on the Majority_vote() function)
    y_pred = y_pred.astype(int)
    
    #print("Predictions for iteration", m)
    #print(y_pred)
    
    # Ensure that y_test and y_pred have the same shape for zero_one_loss calculation
    if y_test.shape != y_pred.shape:
        # Handle shape mismatch if needed
        pass
    #print(y_test, y_pred)
    loss = zero_one_loss(y_test, y_pred)
    losses.append(loss)
    bound_losses.append(pac_bound(lamba, n_min_r, m_scores[i], rhos[i]))
    print(f"m: {m} loss: {loss}")


fig, ax1 = plt.subplots()
plt.plot(m_values,losses)

color = 'tab:blue'
ax1.set_xlabel('m')
ax1.set_ylabel('loss', color=color)
ax1.plot(m_values, losses, color=color, label="loss")
ax1.tick_params(axis='y', labelcolor=color)
color = 'tab:grey'
ax1.plot(m_values, bound_losses, color=color, label ="bound")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('time (s)', color=color)  # we already handled the x-label with ax1
ax2.plot(m_values, m_times, color=color, linestyle='dashed', label = "runtime")
ax2.tick_params(axis='y', labelcolor=color)
fig.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



        
        

            
        
        