% Code for Gauss-Newton, with back-tracking line search
clear, clc, clf;

% Fontsize, for plotting
fs = 14;

% Set initial condition
x0 = zeros(2,1);

% Set stopping criterion for main loop
stop_crit = 1E-8;

% Set initial interate to zero
k = 0;

% Set initial inverse Hessian approx
I = eye(2);
H0 = I;

% Initialize loop variables
xold = x0;
Hold = H0;

% Create potential energy function
t = [-2, -1, 0, 1];
y = [0.5, 1, 2, 4];
%y = [0.5, 1, 5, -4];
R =@(x1,x2) [exp(x1 + t(1)*x2) - y(1);
             exp(x1 + t(2)*x2) - y(2);
             exp(x1 + t(3)*x2) - y(3);
             exp(x1 + t(4)*x2) - y(4)];
f =@(x1,x2) (1/2)*(R(x1,x2)'*R(x1,x2));
J =@(x1,x2) [exp(x1 - 2*x2), -2*exp(x1 - 2*x2);
             exp(x1 - x2), -exp(x1 - x2);
             exp(x1), 0;
             exp(x1 + x2), exp(x1 + x2)];
gradf =@(x1,x2) J(x1,x2)'*R(x1,x2);



% Create the 'A' in the Ax=b
A =@(x1,x2) J(x1,x2)'*J(x1,x2);
Ainv =@(x1,x2) inv(A(x1,x2));

% MAKE THIS INTO A NON-SYMBOLIC?

% Set initial step length for the backtracking line search
alpha0 = 1;
c = 1E-4;

% Create history arrays to store stats of the optimization algorithm
jHist = 0;
fkHist = f(x0(1), x0(2));
alphakHist = alpha0;
gradHist = norm(gradf(x0(1), x0(2)));

p0 = -Ainv(x0(1), x0(2)) * J(x0(1), x0(2))' * R(x0(1), x0(2));

% Print status of the script to the screen
fprintf('Optimization algorithm started... yehaw! \n');

% Outer loop of BFGS
while (norm(J(xold(1), xold(2))'*R(xold(1), xold(2))) > stop_crit)

    % Set the search direction
    p = -Ainv(xold(1), xold(2)) * J(xold(1), xold(2))' * R(xold(1), xold(2));

    % Use a line-searching method to find a good step length
    % Initialize internal iterate counter
    j = 0;

    % Initialize step length and step
    alpha = alpha0;

    % Update the minimizer estimate
    xnew = xold + alpha*p;

    % Main backtracking loop
    %while (rosenbrock_2Nd(x + alpha*p,0) > Armijo(alpha,x,p,c))
    while (f(xnew(1), xnew(2)) > (f(xold(1), xold(2)) + c*alpha*p'*gradf(xold(1), xold(2))))

        % Contract the step length
        alpha = alpha * (1/2);
        xnew = xold + alpha*p;

        % Update exception variable
        j = j + 1;
    end

    iterates = j;

    % Update the BFGS 's', 'y' and 'rho' variables
    s = xnew - xold; % s = alpha*p
    y = gradf(xnew(1), xnew(2)) - gradf(xold(1), xold(2));
    rho = 1/((y')*s);

    % Update Hessian approximation
    temp1 = I - rho*(s*y');
    temp2 = I - rho*(y*s');
    Hnew = temp1*Hold*temp2 + rho*(s*s');
    %Henew = inv(rosenbrock_2Nd(xnew,2));

    % Record the updated variables (push variables into history vector)
    jHist(k  + 1) = iterates;
    fkHist(k + 1) = f(xold(1), xold(2));
    alphakHist(k + 1) = alpha;
    gradHist(k + 1) = norm(gradf(xold(1), xold(2)));

    % Update loop variables
    xold = xnew;
    Hold = Hnew;
    k = k + 1;
end

% Create a history array to store stats of the optimization algorithm
klist = 1:1:k;
Full_History = [klist', jHist', fkHist', alphakHist', gradHist'];
FirstSix = Full_History(1:4,:);
LastSix = Full_History(end-3:end,:);

ColumnNames = {'k', 'Backtracking iterations', 'f(x)', 'alpha', '|grad f(x)|'};
FirstSixTable = table(FirstSix(:,1), FirstSix(:,2), FirstSix(:,3), FirstSix(:,4), ...
    FirstSix(:,5), 'VariableNames', ColumnNames);
LastSixTable = table(LastSix(:,1), LastSix(:,2), LastSix(:,3), LastSix(:,4), ...
    LastSix(:,5), 'VariableNames', ColumnNames);


% Print a summary of the results of the algorithm to the screen
format long
fprintf('\nThe algorithm terminated after: %1.f iterations\n',k)
fprintf('The first 6 iterations of the algorithm looked like:\n')
disp(FirstSixTable)
fprintf('The last 6 iterations of the algorithm looked like:\n')
disp(LastSixTable)


% Make some charts of more algorithm stats
figure (1)
semilogy(klist, fkHist)
xlabel('Iteration number, $k$','Interpreter','latex','FontSize',fs)
ylabel('Objective function, $f(\textbf{x})$','Interpreter','latex','FontSize',fs)
grid on
