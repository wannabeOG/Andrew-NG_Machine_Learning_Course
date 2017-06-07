function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
k=0;
j=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
[k,j]=size(z);
for i=1:k
  for l=1:j
  g(i,l)=1/(1+exp(-z(i,l)));
 end 





% =============================================================

end
