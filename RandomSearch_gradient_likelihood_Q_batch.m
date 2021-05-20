% J. C. Spall, June 2013
% RandomSearch_gradient_likelihood_Q_batch
% This code is a basic random search (Alg. B) recursion for state-space estimation of Q matrix. This does batch processing over all data. 
% The alg. offers both a random search to minimize neg. log-likelihood and
% a random search to minimize the norm of the gradient of the
% log-likelihood. The methods are ordered sequentially below, so that one, the other, or both can be used.
%
clear all
global F C R dimx dQ Plag0 xlag0 ndata; %ndata is only used in function call when using cumulative gradient calculation; 
format long
n_loss=1000; %no. of iterations in random search algorithm B as applied to loss fn
n_grad=10; %no. of iterations in random search algorithm B as applied to minimizing norm of gradient of loss fn
ndata=2000; %no. of outputs generated via state-space modeli
cases=1;
p=3; %fixed dimension for theta
%Steps below generate vector of output data
F=eye(3);
C(:,:,:,1)=[1 0 0]; %defines first (of 3) C vectors; cycle through blocks of three C vectors in state-space model; ML treats as column vector! 
C(:,:,:,2)=[0 1 0]; %defines second (of 3) C vectors
C(:,:,:,3)=[0 0 1]; %defines third (of 3) C vectors 
Q=zeros(3,3); %sets dimension of estimated Q and fills out off-diag entries 
Qtrue=eye(3);
R=.01;
mu0=[0 0 0]';
P0=eye(3);
dimx=3;

%dQ=[eye(3); [1 2 3]]; dQ(:,:,1)%deriv. Q w.r.t. theta
dQ(:,:,1)=[1 0 0]'*[1 0 0];%deriv. Q w.r.t. theta TEMP for when calling StochGradient_batch.m 
dQ(:,:,2)=[0 1 0]'*[0 1 0];
dQ(:,:,3)=[0 0 1]'*[0 0 1];

D=[.5,3;.5,3;.5,3]; % 2-column matrix specifies the 
			%lower and upper limits for p=3 components of theta (2-column
			%matrix)
s_B=.05; %standard dev. of components in normal perturbations for alg. B  
z=zeros(ndata,1);
randn('seed',31111113) %seed 31111113 used consistently to produce data (same values of z across different programs)
x=sqrtm(P0)*randn(dimx,1); 
Qtrue_sr=sqrtm(Qtrue);
Rsr=sqrt(R);
for t=1:ndata %this loop generates data and stores in vector z
    x=F*x+Qtrue_sr*randn(dimx,1);%x
    z(t)=C(:,:,:,t-(floor((t-1)/3))*3)'*x+Rsr*randn; %recall Matlab treats C(:,:,:,t-(floor((t-1)/3))*3) as COLUMN vector!
end

%theta_0=[0.792988993498136   1.424129900977366   0.674294420581938]'; %latest i.c. from many runs of direct optimization of L via alg. B; gives loss= 1.007391658494420 (seems to be best possible, after conducting sensitivity studies); based on ndata=200 and seed=31111113; gives grad=[-0.002058844509795, -0.010219698111838, -0.001015063353503]' and grad norm = 0.010474322117648

%theta_0=[0.966461175107341   0.961655650866003   1.015048346337450] %temp??
theta_0=4*[1 1 1]'+[-1 -2 -3]'; %gives loss=0.839288823136975
theta_0=[0.994437830454378
   1.115959419758080
   1.116173252109350];
loss='Likelihood_Q_batch'; %this is the loss for use with random search (direct neg. log-likelihood)
grad='Gradient_batch'; %this is the gradient of neg. log-likelihood; use this with norm operator to do opt. of min. gradient value
theta=theta_0;
Q(1,1)=theta(1);
Q(2,2)=theta(2); 
Q(3,3)=theta(3);
Plag0=P0;
xlag0=mu0;
feval(loss,z,Q) %initial loss evaluation for comparison purposes; all evaluations of loss or gradient require resetting of Plag0 and xlag0, as in lines immediately above 
Plag0=P0;
xlag0=mu0;
feval(grad,z,Q)
for i=1:cases %set up to use same state-space data
     % temp1=zeros(3,1);%temp
   theta=theta_0;
   Q(1,1)=theta(1);
   Q(2,2)=theta(2);
   Q(3,3)=theta(3);
   Plag0=P0;
   xlag0=mu0;
   lossold=feval(loss,z,Q); 
   %First loop does direct optimization of loss function (neg. log-likelihood)
   for k=1:n_loss
      thetanew=theta+s_B*randn(p,1);
      thetanew=min(thetanew,D(:,2));
      thetanew=max(thetanew,D(:,1));
      Q(1,1)=thetanew(1);
      Q(2,2)=thetanew(2);
      Q(3,3)=thetanew(3);
      Plag0=P0; %for batch processing, need to re-set Plag0 at each iteration
      xlag0=mu0; %for batch processing, need to re-set xlag0 at each iteration
      lossnew=feval(loss,z,Q);%this is real "lossold" with actual likelihood fn.
      if lossnew < lossold  	
          theta=thetanew;
          lossold=lossnew;
      else
      end
   end
   theta
   Q(1,1)=theta(1);
   Q(2,2)=theta(2);
   Q(3,3)=theta(3);
   Plag0=P0;
   xlag0=mu0;
   feval(loss,z,Q) %check of loss value prior to going into grad. optimization part
   Plag0=P0;
   xlag0=mu0;
   feval(grad,z,Q) %check of grad value prior to going into grad. optimization part
   Plag0=P0;
   xlag0=mu0;
   %Second loop does minimization of gradient of loss function
   lossold=norm(feval(grad,z,Q));
   for k=1:n_grad
      thetanew=theta+s_B*randn(p,1);
      thetanew=min(thetanew,D(:,2));
      thetanew=max(thetanew,D(:,1));
      Q(1,1)=thetanew(1);
      Q(2,2)=thetanew(2);
      Q(3,3)=thetanew(3);
      Plag0=P0; %for batch processing, need to re-set Plag0 at each iteration
      xlag0=mu0; %for batch processing, need to re-set xlag0 at each iteration
      lossnew=norm(feval(grad,z,Q)); %TEMP with lossnew as norm of gradient
      if lossnew < lossold  	
          theta=thetanew;
          lossold=lossnew;
      else
      end
   end
end
%final evaluations
%theta
Q(1,1)=theta(1);
Q(2,2)=theta(2);
Q(3,3)=theta(3);
%Q %temp
Plag0=P0;
xlag0=mu0;
feval(loss,z,Q) %final loss value 
Plag0=P0;
xlag0=mu0;
feval(grad,z,Q) % final grad value