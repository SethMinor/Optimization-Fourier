% CONJUGATE GRADIENT METHOD 
clear, clc, clf;

% Set a dimension counter
d = 1;

% Specify the dimensions to loop over
dims = [5,8,12,20];

% Loop over these dimensions
for n = dims

    % Specify the stopping condition for the alg.
    Stop_cond = 1e-6;
    
    % The Hilbert matrix, H
    H = hilb(n);
    K = cond(H); 
    
    % The RHS vector, b
    b = ones(n,1);
    
    % Initial conditions
    x0 = zeros(n,1);
    p0 = -(H*x0 - b);
    
    % Initial residual
    r0 = H*x0 - b;
    
    % Initialize counter
    k = 0;
    
    % Set iterates as initial values
    xk = x0;
    rk = r0;
    pk = p0;
    
    % Create history arrays to store stats of the algorithm
    RkNormList = norm(rk);
    
    % Main for-loop
    while (norm(rk) > Stop_cond)
    
        % Update the step length
        ak = (rk'*rk)/(pk'*H*pk);
    
        % Take a step
        xk = xk + ak*pk;
    
        % Update the residuals
        rnew = rk + ak*H*pk;
    
        % Update beta
        Bk = (rnew'*rnew)/(rk'*rk);
    
        % Update the step direction
        pnew = Bk*pk - rnew;
    
        % Record the updated variables (push into history array)
        RkNormList(end + 1) = norm(rnew);
    
        % Update iterated variables
        rk = rnew;
        pk = pnew;
    
        % Update the counter
        k = k + 1;
    
    end
    
    
    % Create plots summarizing the performance of the algorithm
    % Plot the log of the norm of the residual
    figure (1)
    subplot(2,2,d)
    plot(log10(RkNormList))
    xlabel('Iteration number, $k$','Interpreter','latex')
    ylabel('$\log_{10}(|r_k|)$','Interpreter','latex')
    title("Matrix Dimension $=$ "+dims(d),'Interpreter','latex')
    grid on
    hold on
    
    % Plot the iteration number vs n
    figure (2)
    plot(n,k,'.','MarkerSize',25)
    xlabel('Matrix Dimension, $n$','Interpreter','latex')
    ylabel('Number of Iterations, $k$','Interpreter','latex')
    grid on
    hold on
    
    % Plot the log of the condition number vs n
    figure (3)
    plot(n,log10(K),'.','MarkerSize',25)
    xlabel('Matrix Dimension, $n$','Interpreter','latex')
    ylabel('$\log_{10}($cond$(H))$','Interpreter','latex')
    grid on
    hold on

    % Plot the log of the eigenvalue vs n
    figure (4)
    subplot(2,2,d)
    plot(log10(abs(eig(H))))
    xlabel('$n$ for Eigenvalue $\lambda_n$','Interpreter','latex')
    ylabel('$\log_{10}(|\lambda_n|)$','Interpreter','latex')
    title("Matrix Dimension $=$ "+dims(d),'Interpreter','latex')
    grid on
    hold on
    
    % Update dimension counter
    d = d + 1;
end
