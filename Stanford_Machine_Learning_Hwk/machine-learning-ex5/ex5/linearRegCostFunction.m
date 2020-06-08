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

h_theta = theta' * X';
J = (h_theta' - y).^2;
J = sum(J)/(2*m);
theta_reg = theta(2:end);
J_reg = theta_reg.^2;
J_reg = sum(J_reg) * (lambda/(2*m));
J = J + J_reg;

grad(1) = (1/m) * sum(h_theta' - y) * X(1);
grad_reg = 0;
for j = 2:size(theta)
  grad(j) = ((1/m) * sum((h_theta' - y) .* X(:,j))) + ((lambda/m) * theta(j));
end
% =========================================================================

grad = grad(:);

end
