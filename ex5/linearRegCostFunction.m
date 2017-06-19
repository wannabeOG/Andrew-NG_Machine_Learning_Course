function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Cost Function Calculation
a=((X*theta-y)'*(X*theta-y));
b=theta(2:end);
b=b.^2;
c=sum(b);
J= ((a+lambda*c)*0.5)/m; 

% Gradient Calculation
d=(X')*(X*theta-y);
e=theta(2:end);
e=[0;e];
grad=(d+lambda*e)/m;











% =========================================================================

grad = grad(:);

end
