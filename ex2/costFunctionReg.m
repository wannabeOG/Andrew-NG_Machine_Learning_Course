function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% cost function
s=0;
h=0;
t=0;
for i=1:m
   s=0;
  for j=1:size(theta)
    s=s+theta(j)*X(i,j);
  end
  h=h+(-1)*(y(i)*log(sigmoid(s))+(1-y(i))*log(1-sigmoid(s)));
end
for i=2:size(theta)
  t=t+theta(i)^2;
end
J=(h+(0.5*lambda*t))/m;

% gradient 
for k=1:size(theta)
   f=0;
  for i=1:m
     e=0;
    for j=1:size(theta)
     
      e=e+theta(j)*X(i,j);
    end
    f=(sigmoid(e)-y(i))*X(i,k)+f;
  end
  if (k==1)
    grad(k)=f/m;
  else
    grad(k)=f/m+lambda*theta(k)/m;
  end
  
  
 
end
  

 
 





% =============================================================

end
