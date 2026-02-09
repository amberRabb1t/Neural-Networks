#import time, to count the time it takes for the whole program but also some of its individual parts to run
import time

start = time. perf_counter()

#to avoid certain ROCm-related warnings, probably caused by the use of a GPU which is not officially supported by AMD for GPU compute via ROCm
import warnings
warnings.filterwarnings("ignore")

#import pytorch, which constitutes the main tool used in this code
import torch

#define the method that loads the data, as described in the CIFAR-10 dataset documentation
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#use GPU if available
devvar = 'cuda' if torch.cuda.is_available() else 'cpu'

#load test data, transform it to tensor, give it datatype "float" and move it to GPU
cifar10_test = unpickle('/home/redpanda/Documents/Auth/Neural_Networks/assignment/cifar-10-batches-py/test_batch')
test_data = torch.tensor(cifar10_test[b'data'], dtype=torch.float, device=devvar)
test_labels = torch.tensor(cifar10_test[b'labels'], dtype=torch.float, device=devvar)

#load the first batch of training data, transform it to tensor, give it datatype "float" and move it to GPU
cifar10_train = unpickle(f'/home/redpanda/Documents/Auth/Neural_Networks/assignment/cifar-10-batches-py/data_batch_1')
train_data = torch.tensor(cifar10_train[b'data'], dtype=torch.float, device=devvar)
train_labels = torch.tensor(cifar10_train[b'labels'], dtype=torch.float, device=devvar)

#load the rest of the training data batches and concatenate them onto a single batch
for n in range(2,6):
	cifar10_train = unpickle(f'/home/redpanda/Documents/Auth/Neural_Networks/assignment/cifar-10-batches-py/data_batch_{n}')
	train_data = torch.cat((train_data, torch.tensor(cifar10_train[b'data'], dtype=torch.float, device=devvar)), 0)
	train_labels = torch.cat((train_labels, torch.tensor(cifar10_train[b'labels'], dtype=torch.float, device=devvar)), 0)

initialend = time. perf_counter()

#print time taken to import libraries and load data
print(f'Time taken for initialization and data loading: {initialend-start:.2f} seconds')

#time knn for k=1
knn1start = time. perf_counter()

#implement k-nn algorithm for k=1
#calculate the euclidean distance
dists = torch.cdist(test_data, train_data, p=2)

#find the indices of the minimum distances
min_dist_indices = dists.argmin(dim=1)

#define the predicted labels by applying the minimum distance indices to the training label tensor
predicted_labels = train_labels[min_dist_indices]

#calculate and print the accuracy
accuracy = torch.mean(torch.tensor(predicted_labels == test_labels, dtype=torch.float))
print(f'Accuracy of 1-NN classifier: {accuracy * 100:.2f}%')

knn1end = time. perf_counter()

#print time taken for knn with k=1
print(f'Time taken for 1-NN: {knn1end-knn1start:.2f} seconds')

knn3start = time. perf_counter()
#implement k-nn algorithm for k=3
k=3

#the distances have already been calculated above, so we simply need to find the indices of the 3 minimum distances for each test sample, aka each row
knn_indices = torch.topk(dists, k, largest=False).indices

#define the knn labels by applying the minimum distance indices to the training label tensor
knn_labels = train_labels[knn_indices]

#iterate over the knn labels and find the most frequently appearing label among the 3 nearest neighbors. "torch.unique" returns the unique elements of the input tensor containing the labels and the "return_counts" parameter
#returns the count of each unique element, which we need to decide which label to pick
for i in range(knn_labels.size(0)):
	#count how many times each label appears among the 3 nearest neighbors
    unique_labels, counter = torch.unique(knn_labels[i], return_counts=True)
	#find the label that appears most frequently using argmax, similar to argmin
    most_frequent = unique_labels[torch.argmax(counter)]
    predicted_labels[i] = most_frequent

#calculate and print the accuracy
accuracy = torch.mean(torch.tensor(predicted_labels == test_labels, dtype=torch.float))
print(f'Accuracy of 3-NN classifier: {accuracy * 100:.2f}%')


knn3end = time. perf_counter()
print(f'Time taken for 3-NN: {knn3end-knn3start:.2f} seconds')


nccstart = time. perf_counter()

#implement nearest centroid algorithm
#set the number of classes as defined by CIFAR-10
num_classes=10

#declare an empty list to store the centroids
centroids = []

#for each class,
for c in range(num_classes):
    #select all data points belonging to class c
    class_data = train_data[train_labels == c]
        
    #calculate the centroid (mean) for class c
    centroid = class_data.mean(dim=0)
    centroids.append(centroid)
    
#stack centroids into a tensor for easier distance calculations
centroid_tensor = torch.stack(centroids)

#compute L2 distance from each test sample to each centroid
dists = torch.cdist(test_data, centroid_tensor, p=2)
    
#find the index of the closest centroid for each test sample. this tensor serves as our prediction tensor
nearest_centroid_indices = dists.argmin(dim=1)

# calculate and print the accuracy
accuracy = (nearest_centroid_indices == test_labels).float().mean().item()
print(f'Accuracy of Nearest Centroid classifier: {accuracy * 100:.2f}%')

nccend = time. perf_counter()
print(f'Time taken for Nearest Centroid: {nccend-nccstart:.2f} seconds')

end = time. perf_counter()

print(f'Time taken for program execution: {end-start:.2f} seconds')

