function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Setting the values to be tested
c=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
s=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%load("ex6data3.mat");
% Setting up the dummy parameters
x1 = [1; 2; 1], x2 = [0; 4; -1],
error=zeros(length(c),length(s));
% Loop to store all the error values
for i=1:length(c)
  for j=1:length(s)
    d=c(i);
    e=s(j);
    model=svmTrain(X,y,d,@(x1,x2)gaussianKernel(x1,x2,e));
    predictions=svmPredict(model,Xval);
    error(i,j)=mean(double(predictions ~=yval));
  end
end
b=min(min(error));
[i,j]=find(error==b);
C=c(i);
sigma=s(j);







% =========================================================================

end
