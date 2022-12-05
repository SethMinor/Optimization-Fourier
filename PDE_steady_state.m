%% Numerical steady states to the NLS
% On the surface of the elongated torus
clear, clc;

% Fontsize, for plotting
fs = 14;

%% Set up for Gauss-Newton

% Define the grid spacing (treating 0 = 2*pi)
Nphi = 25;    % Number of phi mesh points
Ntheta = Nphi;  % Number of theta mesh points
dphi = (2*pi)/(Nphi - 1);     % Phi mesh density
dtheta = (2*pi)/(Ntheta - 1); % Theta mesh density

% Define the discretized (phi,theta) space
Phi = linspace(0, 2*pi - dphi, Nphi);
Theta = linspace(0, 2*pi - dtheta, Ntheta);

% Define parameters related to the torus/Laplace-Beltrami operator
a = 3; % Semi-major axis of ellipse
b = 3; % Semi-minor axis of ellipse
r = 1; % Tube radius of elongated torus
gamma =@(phi,theta) sqrt(((a + r*cos(theta)).^2).*(sin(phi)).^2 ...
    + ((b + r*cos(theta)).^2).*(cos(phi)).^2);
k1 =@(phi,theta) -(sin(theta).*(cos(phi).^2.*(b + r*cos(theta)) ...
    + (sin(phi).^2).*(a + r*cos(theta))))./(gamma(phi,theta));
k2 =@(phi,theta) (2*r*cos(phi).*sin(phi).*((a + r*cos(theta)).^2 ...
    - (b + r*cos(theta)).^2))./(2*gamma(phi,theta).^2);

% Define the weights related to the five-point Laplacian stencil
A = 1/((r*dtheta)^2);
B =@(phi,theta) k1(phi,theta)./(2*r*dtheta*gamma(phi,theta));
C =@(phi,theta) k2(phi,theta)./(2*r*dphi*gamma(phi,theta));
D =@(phi,theta) 1./((dphi*gamma(phi,theta)).^2);

wc =@(phi,theta) A + D(phi,theta);            % Center weight
wl =@(phi,theta) D(phi,theta) - C(phi,theta); % Left weight
wr =@(phi,theta) C(phi,theta) + D(phi,theta); % Right weight
wu =@(phi,theta) A + B(phi,theta);            % Up weight
wd =@(phi,theta) A - B(phi,theta);            % Down weight

% Define the temporal frequency of the solution we're looking for
omega = 1;

% Create the Delta2D and D matrices
Delta2D = Delta2d(wc,wl,wr,wu,wd,Phi,Theta);
D = [Delta2D, zeros(Nphi*Ntheta); zeros(Nphi*Ntheta), Delta2D];

% Define the residual and objective functions
r =@(v) (1/2)*D*v - Lambda(v)*v - omega*v;
f =@(v) (1/2)*(r(v)'*r(v));

% Define the gradient of the objective
gradf =@(v) Jacobian(Delta2D,v,omega)'*r(v);

% Set stopping criterion for main loop
stop_crit = 1E-6;

% Set initial seed
v0 = 10*sech(ones(2*Nphi^2,1));

% Initialize loop variables
vold = v0;
k = 0;

% Initialize history arrays to store stats of the optimization algorithm
fHistory = f(v0);
GradientHistory = norm(gradf(v0));

%% Running the optimization algorithm

% Print status of the script to the screen
fprintf('Optimization algorithm started... yehaw! \n');

% Main outer loop of Gauss-Newton
while (abs(f(vold)) > stop_crit)

    % Set the search direction
    p = - Jacobian(Delta2D,vold,omega) \ r(vold);

    % Update the minimizer estimate
    vnew = vold + p;

    % Record the updated variables (push variables into history vector)
    fHistory(k + 1) = f(vold);
    GradientHistory(k + 1) = norm(gradf(vold));

    % Update loop variables
    vold = vnew;
    k = k + 1;
end

% Define the field that we've converged to
v = vold;


%% Generate a report of the performance of the algorithm

% Define a history array to store algorithm stats
klist = 1:1:k;
Full_History = [klist', fHistory', GradientHistory'];
FirstFive = Full_History(1:5,:);
LastFive = Full_History(end-4:end,:);

ColumnNames = {'k', 'f(v)', '|grad f(v)|'};
FirstFiveTable = table(FirstFive(:,1), FirstFive(:,2), FirstFive(:,3), ...
    'VariableNames', ColumnNames);
LastFiveTable = table(LastFive(:,1), LastFive(:,2), LastFive(:,3), ...
    'VariableNames', ColumnNames);

% Print a summary of the algorithm stats to the screen
format long
fprintf('\nThe algorithm terminated after: %1.f iterations\n',k)
fprintf('The first 5 iterations of the algorithm looked like:\n')
disp(FirstFiveTable)
fprintf('The last 5 iterations of the algorithm looked like:\n')
disp(LastFiveTable)

% Make some charts of more algorithm stats
figure (1)
semilogy(klist, fHistory)
xlabel('Iteration number, $k$','Interpreter','latex','FontSize',fs)
ylabel('Objective function, $|f(\textbf{v}_k)|$','Interpreter','latex','FontSize',fs)
title('Gauss-Newton Convergence','Interpreter','latex','FontSize',fs)
grid on


%% Helper functions

% Define a function that creates Delta1D matrices
function DelTaco = Delta1d(wc,wl,wr,m,Phi)
    DelTaco = diag(-2*wc(Phi,m)) + diag(wr(Phi(1:end-1),m),1) + diag(wl(Phi(2:end),m),-1);
    DelTaco(end,1) = wr(Phi(end),m);
    DelTaco(1,end) = wl(Phi(1),m);
end

% Define a function that returns the Iu and Id matrices
function [Iu,Id] = IuAndId(wu,wd,Phi,Theta)
    Iu = diag(wu(Phi,Theta));
    Id = diag(wd(Phi,Theta));
end

% Define a function that places the Iu and Id matrices in Delta2d
function Bingus = PlaceI(wu,wd,Phi,Theta)
    [Iu,Id] = IuAndId(wu,wd,Phi,Theta);
    
    % Create the Iu portion
    IuBlock = {0};
    for m = 1:length(Theta)
        IuBlock{m} = Iu;
    end
    IuBlock = blkdiag(IuBlock{1:end});
    IuBlock = circshift(IuBlock,length(Phi),2);

    % Create the Id portion
    IdBlock = {0};
    for m = 1:length(Theta)
        IdBlock{m} = Id;
    end
    IdBlock = blkdiag(IdBlock{1:end});
    IdBlock = circshift(IdBlock,length(Phi),2)';

    % Return the combined blocks
    Bingus = IuBlock + IdBlock;
end

% Define a function that returns the Delta2D matrix
function DelTaco = Delta2d(wc,wl,wr,wu,wd,Phi,Theta)
    
    % Add the Delta1D blocks that run down the diagonal
    Block = {0};
    for m = 1:length(Theta)
        Block{m} = Delta1d(wc,wl,wr,m,Phi);
    end
    DelTaco = blkdiag(Block{1:end});

    % Add the Iu and Id blocks on the off-diagonals
    Bingus = PlaceI(wu,wd,Phi,Theta);

    % Deliver the final spicy meatball
    DelTaco = DelTaco + Bingus;
end

% Define a function to return the nonlinear term/Lambda matrix
function L = Lambda(v)
    L = diag(v(1:end/2).^2 + v(end/2+1:end).^2);
    L = blkdiag(L,L);
end

% Define a function which returns the Jacobian matrix
function J = Jacobian(Delta2D,v,omega)

    % Define the blocks of the Jacobian
    J11 = (1/2)*Delta2D - diag(3*v(1:end/2).^2 + v(end/2+1:end).^2 + omega);
    J12 = diag(-2*v(1:end/2).*v(end/2+1:end));
    J22 = (1/2)*Delta2D - diag(v(1:end/2).^2 + 3*v(end/2+1:end).^2 + omega);

    % Put it all together
    J = [J11, J12; J12, J22];
end
