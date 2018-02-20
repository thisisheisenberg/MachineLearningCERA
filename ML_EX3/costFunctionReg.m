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

ans = 0;

for i = 1:m,

	sig = sigmoid(theta' * X(i,:)');
	temp = (y(i) .* log(sig));
	temp1 = (1 - y(i)) .* log(1-sig);
	ans = ans + temp + temp1;

end;

temp2 = -((1/m)*ans);
theta_temp = theta;
theta_temp(1) = 0;
temp3 = (lambda/(2 * m)) * (sum(theta_temp .^ 2));
J = temp2 + temp3;



% Calculating Gradian Descent

for j = 1:size(theta)(1),
	
	temp_grad = 0;

	for i = 1:m,

		temp_grad = temp_grad + ((sigmoid(theta' * X(i,:)') - y(i)) * X(i,j));
	
	end;

	temp_grad = (1/m)*temp_grad;

	if j == 1

		grad(j) = temp_grad;

	else

		grad(j) = temp_grad + (lambda * theta(j)/m);

	endif;

end;
% =============================================================

end
