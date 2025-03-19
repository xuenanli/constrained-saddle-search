% initialization of bar framework
clc;
clear;
cla;

% choose the bar framework
data = load('heptagon-initial-4.mat');
framework = data.framework;

E = framework.E;
n = framework.n;
dim = framework.dim;
pfix = framework.pfix;
x = framework.x;

% for hexagon
free_edge_list = 6;
stress_list = 1;

limx = [-3, 5];
limy = [-1, 6];

% Video name
video_name = 'heptagon-saddle-test.mp4';

figure(1)
plot_framework_2D(framework,free_edge_list)
axis equal
xlim(limx)
ylim(limy)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% function definition

points = sym('p', [n*dim, 1]);
[m,~] = size(E);

% objective function
Fo = sym(length(free_edge_list));
for i=1:length(free_edge_list)
    e = free_edge_list(i);
    ind1 = E(e,1); ind2 = E(e,2); 
    p1 = points(ind1*dim-dim+1:ind1*dim);
    p2 = points(ind2*dim-dim+1:ind2*dim);
    Fo(i) = stress_list(i)*((p1(1)-p2(1))^2+(p1(2)-p2(2))^2);
end
Fo = sum(Fo);
Fo_grad = gradient(Fo,points);

Fo_hess = hessian(Fo,points);

% constraint functions Fc
Fc = sym(m-length(free_edge_list)+length(pfix)*dim);

% the edges with fixed lengths
Efix = E;
Efix(free_edge_list,:) = [];
% edge constraints
for i=1:m-length(free_edge_list)
    ind1 = Efix(i,1); ind2 = Efix(i,2); 
    p1 = points(ind1*dim-dim+1:ind1*dim);
    p2 = points(ind2*dim-dim+1:ind2*dim);
    Fc(i) = (p1(1)-p2(1))^2 + (p1(2)-p2(2))^2 - Efix(i,3)^2;
end
Fc = Fc';

% pinning constraints
for k=1:length(pfix)
    ind = pfix(k);
    Fc(m-length(free_edge_list)+(k-1)*dim+1:m-length(free_edge_list)+k*dim) ...
        = points(ind*dim-dim+1:ind*dim) - x((ind*dim-dim+1:ind*dim));
end

% take the Jacobian matrix
J_mat = jacobian(Fc, points);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% start the optimization

% initialization
xc = x;
framework_cur = framework;
framework_cur.x = xc;

% open video
video = VideoWriter(video_name, 'MPEG-4');
video.FrameRate = 5; % Frames per second
open(video);
figure(1)
plot_framework_2D(framework,free_edge_list)
axis equal
xlim(limx)
ylim(limy)

frame = getframe(gcf); % Capture the frame
writeVideo(video, frame); % Write the frame to video

% optimization setup
eta = 1e-1; % step size
Max_iter = 1e3; % Max iteration number
tol = 1e-8; % optimization termination error
Fo_list = zeros(Max_iter,1); % storing Fo

err = 1;
count = 0;

I = eye(n*dim);

vc = zeros(n*dim,1);
vc(1) = 1;
while count < Max_iter & err > tol
    cla;
    count = count + 1;
    fprintf(['Current iteration count:', int2str(count),':\n'])

    xp = xc;

    grad_eval = double(subs(Fo_grad,points,xc));
    J_mat_eval = double(subs(J_mat,points,xc));
    proj_grad = I - J_mat_eval'*inv(J_mat_eval*J_mat_eval')*J_mat_eval;
    
    % saddle search direction
    dxc = -eta*(I-2*vc*vc')*proj_grad*grad_eval;

    % Newton's update
    xc = newton_a(xc,dxc,Fc,J_mat,points);

    % update direction for v
    J_mat_eval = double(subs(J_mat,points,xc));
    proj_grad = I - J_mat_eval'*inv(J_mat_eval*J_mat_eval')*J_mat_eval;

    hess_eval = double(subs(Fo_hess,points,xc));
    vc = vc - (eye(n*dim) - vc*vc')*hess_eval*vc;
    vc = proj_grad*vc;
    vc = vc / norm(vc);

    % store the objective
    Fo_eval = double(subs(Fo,points,xc));
    Fo_list(count) = Fo_eval;

    err = norm(xc-xp);
    fprintf(sprintf('Optimization error is: %.4e.\n\n',err))

    % plot the current framework
    framework_cur = framework;
    framework_cur.x = xc;
    plot_framework_2D(framework_cur,free_edge_list)
    axis equal
    xlim(limx)
    ylim(limy)

    frame = getframe(gcf); % Capture the frame
    writeVideo(video, frame); % Write the frame to video
end

close(video) % Close the video
Fo_list = Fo_list(1:count);
% plot the objective function
figure(3)
plot(1:count, Fo_list, 'b-', 'LineWidth',2)
title('Value of the objective','FontSize',14)

%% 2D Rigidity tests
clc;

a = framework.a; 
% this adjacency matrix did not count edges between fixed vertices
% insert the edges between fixed vertices
for k=1:length(pfix)
    for j=k+1:length(pfix)
        a(pfix(k),pfix(j)) = 1; a(pfix(j),pfix(k)) = 1;
    end
end
% construct the original rigidity matrix
[R,ER] = formconstraints_bonds(xc,a,pfix,dim);
R = full(R); % ER stores the corresponding edge information (different than E)

% find the infinitesimal 3 rigid body motions
t1 = zeros(size(xc)); t2 = zeros(size(xc));
t1(1:2:end) = 1; t2(2:2:end) = 1;
t3 = inf_rotation_2D(xc);

% There is only one self-stress
[U,S,V] = svd(R); S = diag(S);
fprintf(sprintf('The smallest singular value of the rigidity matrix is: %.4e.\n\n',min(S)))

% find the left and right null space of S
num_rigid = 3;
Nright = V(:,end-num_rigid:end); % first-order flexes
Nleft = U(:,end); % the only self-stress

[Q,~] = qr([t1,t2,t3,Nright]);
% Orthonormal basis for the infinitesimal rigid body motions
q1 = Q(:,1);
q2 = Q(:,2);
q3 = Q(:,3);

% The rest non-trivial first-order flexes
q4 = Q(:,4);
q5 = Q(:,5);

% the original second-order rigidity test
stress_test4 = 0;
stress_test5 = 0;
[m,~] = size(ER);
for k=1:m
    ind1 = ER(k,1); ind2 = ER(k,2);
    v1 = q4(ind1*dim-dim+1:ind1*dim);
    v2 = q4(ind2*dim-dim+1:ind2*dim);
    stress_test4 = stress_test4+Nleft(k)*norm(v1-v2)^2;

    v1 = q5(ind1*dim-dim+1:ind1*dim);
    v2 = q5(ind2*dim-dim+1:ind2*dim);
    stress_test5 = stress_test5+Nleft(k)*norm(v1-v2)^2;
end

fprintf(sprintf('The stress test for q4 is: %.4e.\n\n',stress_test4))
fprintf(sprintf('The stress test for q5 is: %.4e.\n\n',stress_test5))


%% find the two directions that vanish in the second order stress test
clc;

Nangle = 1e5;
angle_list = linspace(0,2*pi,Nangle+1);

stress_test_list = zeros(1,Nangle+1);
for i=1:Nangle+1
    qtest = q4*cos(angle_list(i))+q5*sin(angle_list(i));
    for k=1:m
        ind1 = ER(k,1); ind2 = ER(k,2);
        v1 = qtest(ind1*dim-dim+1:ind1*dim);
        v2 = qtest(ind2*dim-dim+1:ind2*dim);
        stress_test_list(i) = stress_test_list(i)+Nleft(k)*norm(v1-v2)^2;
    end
end

min(abs(stress_test_list))
ind_list = find(abs(stress_test_list)<1e-5);

%% useful function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting bar frameworks
function plot_framework_2D(framework,varargin)

x = framework.x; % vertices of the framework
pfix = framework.pfix; % fixed vertices
A = framework.a; % adjacency matrix
ind = find(triu(A)==1); % indices of bonds
E = framework.E;

if(nargin > 1)
    list_free_edge = varargin{1};
else
    list_free_edge = [];
end

% plot the bar between two fixed vertices
X1 = x(2*pfix(1)-1:2*pfix(1)); X2 = x(2*pfix(2)-1:2*pfix(2));
plot([X1(1), X2(1)],[X1(2), X2(2)], 'k-', 'Linewidth', 2)
hold on
% plot the other bars
for ji=1:length(ind)
    p1 = E(ji,1); p2 = E(ji,2); % two vertices connected together
    X1 = x(2*p1-1:2*p1); X2 = x(2*p2-1:2*p2); % positions of the two vertices
    if find(list_free_edge == ji, 1) % this is a free edge
        plot([X1(1), X2(1)],[X1(2), X2(2)], 'r-', 'Linewidth', 2)
    else
        plot([X1(1), X2(1)],[X1(2), X2(2)], 'k-', 'Linewidth', 2)
    end
    hold on
    if find(pfix == p1) % if p1 is fixed
        scatter(X1(1),X1(2),100,'r^','filled')
        scatter(X2(1),X2(2),100,'bo','filled')
    elseif find(pfix == p2) % if p2 is fixed
        scatter(X1(1),X1(2),100,'bo','filled')
        scatter(X2(1),X2(2),100,'r^','filled')
    else % either p1 or p2 is fixed
        scatter(X1(1),X1(2),100,'bo','filled')
        scatter(X2(1),X2(2),100,'bo','filled')
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Newton iteration
function xc = newton_a(xc,dxc,Fc,J_mat,points)
    % Newton's initialization
    tol = 1e-10; % termination tolerance
    Max_iter = 10; % Max iteration for Newton
    err = 1; 
    count = 0;

    J_eval = double(subs(J_mat,points,xc));
    [m,~] = size(J_eval);
    a_vec = zeros(m,1);
    xc_update = xc+dxc+J_eval'*a_vec; 
    Fc_eval = double(subs(Fc,points,xc_update));

    while count < Max_iter & err > tol
        Jc_eval = double(subs(J_mat,points,xc_update));
        Jc_eval = Jc_eval*(J_eval');
        a_vec = a_vec - Jc_eval \ Fc_eval;
        xc_update = xc+dxc+J_eval'*a_vec;
        Fc_eval = double(subs(Fc,points,xc_update));
        
        % update the error
        count = count + 1;
        err = norm(Fc_eval);
    end
    
    xc = xc_update;
    fprintf(sprintf('Newton error is: %.4e.\n',err))
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function inf_rot = inf_rotation_2D(x)
Rot = [0,1;-1,0];
inf_rot = zeros(size(x));
for i=1:floor(length(x)/2)
    inf_rot(i*2-1:i*2) = Rot*x(i*2-1:i*2);
end
end
