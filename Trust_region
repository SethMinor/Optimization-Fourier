% Code for trust-region search with the Dogleg and Cauchy point methods
clear, clc, clf;
fs = 14; % font size for plots



%% Rosenbrock optimization function
MyRosenbrock =@(x1, x2) 100*(x2 - x1.^2).^2 + (1 - x1).^2;

% Define the gradient of f
Grad_f =@(x1,x2) [-400*x1.*(x2-x1.^2) - 2*(1-x1), 200*(x2-x1.^2)]';

% Define the Hessian matrix
H =@(x1,x2) [(2 - 400*x2 + 1200*x1.^2), -400*x1; -400*x1, 200];

% H inverse
Hinv =@(x1,x2) [1/(400*x1.^2 - 400*x2 + 2), x1./(200*x1.^2 - 200*x2 + 1);
        x1./(200*x1.^2 - 200*x2 + 1), (600*x1^2 - 200*x2 + 1)./(200*(200*x1^2 - 200*x2 + 1))];

% Quadratic model of objective fcn centered at (x1k, x2k)
Mk =@(x1,x2,x1k,x2k) MyRosenbrock(x1k, x2k) + [x1 - x1k, x2 - x2k]*Grad_f(x1k, x2k) + ...
    (1/2).*[x1 - x1k, x2 - x2k]*H(x1k, x2k)*[x1 - x1k, x2 - x2k]';



%% Perform a trust-region optimization algorithm

% Choose a trust-region technique (!!!)
Trust_region_technique = "dogleg";
%Trust_region_technique = "cauchy_point";


% Steepest descent: unit vector in direction of negative gradient
Steepest_Pk =@(x1,x2) -Grad_f(x1,x2)./norm(Grad_f(x1,x2));
% Newton-Raphson method: -H^(-1)gradient(f)
Newton_Pk =@(x1,x2) -Hinv(x1,x2)*Grad_f(x1, x2);


% Set the stopping criterion parameters
Stop_Value = 10^(-8);
Stop_Crit =@(x1, x2) norm(Grad_f(x1,x2)); % norm of the gradient
Max_Iterations = 2500;

% Initial conditions to give a whirl
x0one = [-1.2, 1];
%x0two = [2.8, 4];

% Pick an initial condition (!!!)
Initial_condition = x0one;

% trust region radius parameters
Dk_0 = 1; % initial trust region radius
Dk_max = 300; % max allowable trust region radius


% Initialize these sneaky counters and loop variables
k = 0;
x_k = Initial_condition;
D_k = Dk_0;
f0 = MyRosenbrock(x_k(1), x_k(2));

% move-on condition
eta = 0.24; % (if rho_k <= eta, then shrink the trust region and try again)

% Create history arrays to store stats of the optimization algorithm
xkHist1 = x_k(1);
xkHist2 = x_k(2);
fkHist = f0;

% Print status of the script to the screen
fprintf('Optimization algorithm started, using the initial values: \n');
fprintf('  (x1, x2) = (%0.1f, %0.1f) ', Initial_condition(1), Initial_condition(2));
fprintf('\n f(x1, x2) = %0.1f \n\n', f0);


% Initialize condition for while loop to continue
stop_bool = false;

% Loop over the iterations of the trust-region optimization algorithm
while (stop_bool == false)

    % Print the iteration number every nprint iterations
    nprint = 500;
    if (mod(k,nprint) == 0)
        fprintf('\n... Iteration: k = %1.f\n',k)
    end

    % Changing the trust region radius ----------------------
    % Compute Newton and steepest descent directions
    Steepest_step = Steepest_Pk(x_k(1), x_k(2));
    %Newton_step = Newton_Pk(x_k(1), x_k(2));
    
    % Cauchy point computation
    if (Trust_region_technique == "cauchy_point")
        % tau condition
        tau_condition = Grad_f(x_k(1), x_k(2))'*H(x_k(1), x_k(2))*Grad_f(x_k(1), x_k(2));

        % determine what tau is
        if (tau_condition <= 0)
            tau_k = 1;
        else
            tau_k = min(1, (norm(Grad_f(x_k(1), x_k(2)))^3)/(D_k * tau_condition));
        end
        P_k = -tau_k*D_k*Steepest_step;
    end
    
    % Dogleg point computation
    if (Trust_region_technique == "dogleg")
        % Full Newton Step
        Pnewton_k = Newton_Pk(x_k(1), x_k(2));

        % Compute the steepest descent direction
        temp1 = Grad_f(x_k(1), x_k(2))'*Grad_f(x_k(1), x_k(2));
        temp2 = Grad_f(x_k(1), x_k(2))'*H(x_k(1), x_k(2))*Grad_f(x_k(1), x_k(2));
        Psteepest_k = - (temp1/temp2)*Grad_f(x_k(1), x_k(2));

        % Combine em for the dogleg! Arf arf!
        if (norm(Pnewton_k) <= D_k) % if the Newton step is in T_k
            P_k = Pnewton_k;
        elseif (norm(Psteepest_k) >= D_k) % if both the Newton step and SD step are outside of T_k
            P_k = D_k*Psteepest_k./norm(Psteepest_k);
        else % if the Newtwon step is outside of T_k but the SD Mk minimizer is inside T_k
            % Initialize the tau parameter
            tau = 1;

            % Choose the tau parameter tau_star that gives the dogleg exit point for T_k
            while ((norm(P_k) < D_k) && (tau <= 2))
                % update the dogleg vector
                P_k = Psteepest_k + (tau-1)*(Pnewton_k - Psteepest_k);

                % update tau
                tau = tau + 0.01;
            end
            tau_star = tau;
        end

    end

    % Success condition for a step
    rho_k = ( MyRosenbrock(x_k(1), x_k(2)) - MyRosenbrock(x_k(1) + P_k(1), x_k(2) + P_k(2)) )/...
        ( Mk(x_k(1),x_k(2), x_k(1),x_k(2)) - Mk(x_k(1) + P_k(1), x_k(2) + P_k(2), x_k(1),x_k(2)) );

    % Shrink/expand the trust region size based on rho_k
    if (rho_k < 0.25)
        D_kPlusOne = (1/4)*D_k; % shrink
    elseif ((rho_k > 0.75) && ( abs(norm(P_k) - D_k) < 0.00000001))
        D_kPlusOne = min(2*D_k, Dk_max); % expand
    else
        D_kPlusOne = D_k + 0; % don't shrink or expand
    end
    % -------------------------------------------------------

    % Use the eta condition to decide whether to move on or not
    if (rho_k > eta)
        x_kPlusOne = x_k + P_k';
    else
        x_kPlusOne = x_k;
    end

    % Use P_k to find the next iteration of the other loop variables
    f_kPlusOne = MyRosenbrock(x_kPlusOne(1), x_kPlusOne(2));

    % Record the updated variables (push variables into history vector)
    xkHist1(end + 1) = x_kPlusOne(1);
    xkHist2(end + 1) = x_kPlusOne(2);
    fkHist(end + 1) = f_kPlusOne;
    
    % Update the iteration number
    k = k + 1;
    
    % Check the overall stopping condition to see if we should continue
    if (Stop_Crit(x_kPlusOne(1), x_kPlusOne(2)) < Stop_Value) || (k >= Max_Iterations)
        stop_bool = true;
        fprintf('\nAlgorithm has terminated! Whoah!\n');
        break;
    end
    
    % Update the loop variables
    x_k = x_kPlusOne;
    D_k = D_kPlusOne;
    f_k = f_kPlusOne;

end



%% Plot some stats of the algorithm

% Create a history array to store stats of the optimization algorithm
klist = 1:1:k+1;
Full_History = [klist', xkHist1', xkHist2', fkHist'];
FirstFour = Full_History(1:4,:);
LastFour = Full_History(end-3:end,:);

ColumnNames = {'k', 'x1', 'x2', 'f(x1,x2)'};

FirstFourTable = table(FirstFour(:,1), FirstFour(:,2), FirstFour(:,3), FirstFour(:,4), ...
    'VariableNames', ColumnNames);
LastFourTable = table(LastFour(:,1), LastFour(:,2), LastFour(:,3), LastFour(:,4), ...
    'VariableNames', ColumnNames);


% Print a summary of the results of the algorithm to the screen
format long
fprintf('\nThe algorithm terminated after: %1.f iterations\n',k+1)
fprintf('The first 4 iterations of the algorithm looked like:\n')
disp(FirstFourTable)
fprintf('The last 4 iterations of the algorithm looked like:\n')
disp(LastFourTable)
fprintf('The final stopping criterion value was: %0.15f\n\n', ...
    Stop_Crit(x_kPlusOne(1), x_kPlusOne(2)))

% Make some charts of more algorithm stats
figure (1)
semilogy(klist, fkHist)
xlabel('Iteration number, $k$','Interpreter','latex','FontSize',fs)
ylabel('Objective function, $f(x_1, x_2)$','Interpreter','latex','FontSize',fs)
title("Initial condition, $(x_1, x_2)=$ (" + Initial_condition(1) + ", " ...
    + Initial_condition(2) + ")",'Interpreter','latex','FontSize',fs)
grid on


% Plot the path that the algorithm took to x* = (x1*, x2*)
figure (2)

plot3(xkHist1(1), xkHist2(1), MyRosenbrock(xkHist1(1), xkHist2(1)),...
    'ro','MarkerSize',6,'MarkerFaceColor','w')
hold on
plot3(xkHist1, xkHist2, MyRosenbrock(xkHist1, xkHist2), '-r')
hold on
plot3(xkHist1(end), xkHist2(end), MyRosenbrock(xkHist1(end), xkHist2(end)),...
    'ro','MarkerSize',6,'MarkerFaceColor','r')
hold on
dx = 0.01;
[L, R] = deal(-1.5, 1.5);
dy = 0.01;
[D, U] = deal(-2, 3);
[X1, X2] = meshgrid(L:dx:R, D:dy:U);
Z = MyRosenbrock(X1, X2);
surf(X1,X2,Z)
shading interp
hold off

legend('Starting point, $x_0$','Path taken, $\{x_k\}$','Numerical $x^{\star}$',...
    'Interpreter','Latex','FontSize',fs-2)
xlim([-1.5 1.5])
ylim([-2 3])
xlabel('$x_1$','Interpreter','latex','FontSize',fs)
ylabel('$x_2$','Interpreter','latex','FontSize',fs)
zlabel('$f(x_1, x_2)$','Interpreter','latex','FontSize',fs)
title("Initial condition, $(x_1, x_2)=$ (" + Initial_condition(1) + ", " ...
    + Initial_condition(2) + ")",'Interpreter','latex','FontSize',fs)
grid on
