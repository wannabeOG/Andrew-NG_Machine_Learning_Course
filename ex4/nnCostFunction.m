function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% cost function without regularization

% Calculation of h_theta(x)
a1=X';
a1=[ones(1,m);a1];
z2=Theta1*(a1);
a2=sigmoid(z2);
a2=[ones(1,m);a2];
z3=Theta2*a2;
a3=sigmoid(z3);

% Converting y into a matrix of binary values 
Y=zeros(num_labels,m);
for i=1:length(y)
  Y(y(i),i)=1;
end

% cost function calculation
s=0;
% could not achieve 100% vectorization 
for i=1:num_labels
  s=s+(-1)*(log(a3(i,:))*(Y'(:,i))+log(1-a3(i,:))*(1-Y'(:,i)));
end
J=s/m;

% Cost function considering regularization
w1=Theta1(:,(2:end));
w2=Theta2(:,(2:end));
q=[w1(:);w2(:)];
q=q.^2;
a=sum(q);
J=J+(lambda*a*0.5)/m;

% Gradient Calculation 
Delta_1=zeros(size(Theta1));
Delta_2=zeros(size(Theta2));
for i=1:m
  % Step 1
  b1=X(i,:);
  a1=(b1)';
  a1=[1;a1];
  z2=Theta1*a1;
  a2=sigmoid(z2);
  a2=[1;a2];
  z3=Theta2*a2;
  a3=sigmoid(z3);
  
  % Step 2
  delta_3=a3-Y(:,i);
  delta_2=(Theta2)'*delta_3.*sigmoidGradient([1;z2]);
  delta_2 = delta_2(2:end);
  
  Delta_1=Delta_1+delta_2*(a1)';
  Delta_2=Delta_2+delta_3*(a2)';
end
Theta1_grad=Delta_1/m;
Theta2_grad=Delta_2/m;
w1=[zeros(size(Theta1,1),1) w1];
w2=[zeros(size(Theta2,1),1) w2];
% Gradient calculation with regularization:
Theta1_grad=Theta1_grad+(lambda*w1)/m;
Theta2_grad=Theta2_grad+(lambda*w2)/m; 
  
  
  
  
  
  

  


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
