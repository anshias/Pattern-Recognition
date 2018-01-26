clc;
%  trainimages = loadMNISTImages('train-images.idx3-ubyte');
%  trainlabels = loadMNISTLabels('train-labels.idx1-ubyte');
%  testimages = loadMNISTImages('t10k-images.idx3-ubyte');
%  testlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
load('traindata.mat')
load('testdata.mat')
k = 1:10; 
vec = zeros(size(k)); 
for i = k 
Mdl = fitcknn(trainimages',trainlabels,'NumNeighbors',i, 'Standardize',1);

result = predict(Mdl,testimages'); 
error_rate = sum(result~=testlabels)/length(testlabels);
vec(i) = error_rate;
end                      
plot(k, vec);

%w

