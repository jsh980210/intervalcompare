%Kalman_filter
clear all
%global F C R dimx dQ Plag0 xlag0 ndata; %,ti,tj;
ndata=200; %no. of outputs generated via state-space modeli
cases=1;
p=3; %fixed dimension for theta
%Steps below generate vector of output data
%F=eye(3);
%F=[1 1 0;0 1 1;0 0 1];
%F=[1 .5 0;0 1 .5;0 0 1];
%F=.5*[1 1 1;0 1 0;0 0 1]; %.5*[1 0 -1;0 1 0;0 0 1] works ok
F=.5*[1 1 0;0 1 1;0 0 1];
%C(:,:,:,1)=[1 1 0]; %defines first (of 3) C vectors; cycle through blocks of three C vectors in state-space model; ML treats as column vector! 
%C(:,:,:,2)=[0 1 1]; %defines second (of 3) C vectors
%C(:,:,:,3)=[1 0 1]; %defines third (of 3) C vectors 
C=[1 1 1]; %Same as traditional "H matrix"
%C=[1 0 0];
%C(:,:,:,1)=[1 0 0]; %defines first (of 3) C vectors; cycle through blocks of three C vectors in state-space model; ML treats as column vector! 
%C(:,:,:,2)=[0 1 0]; %defines second (of 3) C vectors
%C(:,:,:,3)=[0 0 1]; %defines third (of 3) C vectors 
Q=eye(3);
%Q=0*[1 .5 0;.5 1 .5;0 .5 1];
%Q=[1 .7071 0;.7071 1 .7071;0 .7071 1];
R=1;
mu0=[0 0 0]';
P0=eye(3);
dimx=3;
z=zeros(ndata,1); %sets dimension of data vector
x1=zeros(ndata,2); %arrays for storing true state and state estimate for use in plotting (when cases = 1); these variables are superfluous o.w.
x2=zeros(ndata,2);
x3=zeros(ndata,2);
randn('seed',31111113) %seed 31111113 used consistently to produce data (same values of z across different programs)
Qsr=sqrtm(Q);
Rsr=sqrt(R);
xaxis=zeros(ndata,1);
cumPlag0=zeros(3,3); %initializes and sets dimension of matrix of estimated error-cov matrices
for c=1:cases
   Plag0=P0;
   xlag0=mu0; 
   x=sqrtm(P0)*randn(dimx,1);
   for t=1:ndata %this loop generates data and stores in vector z
      x=F*x+Qsr*randn(dimx,1);%x
      x1(t,1)=x(1);
      x2(t,1)=x(2);
      x3(t,1)=x(3);
      % z(t)=C(:,:,:,t-(floor((t-1)/3))*3)'*x+Rsr*randn; %recall Matlab treats C(:,:,:,t-(floor((t-1)/3))*3) as COLUMN vector!
      z(t)=C*x+Rsr*randn;
   end
   %filter estimate below
   for t=1:ndata
      xaxis(t)=t; %for plotting
      Plag1=F*Plag0*F'+Q;%Kalman filter equations: P(t+1|t) update
     % CC=C(:,:,:,t-(floor((t-1)/3))*3)'; %cycles through 3 possible C values over each 3 data points; recall Matlab treats C(:,:,:,t-(floor((t-1)/3))*3) as COLUMN vector!
      CC=C;
      B=CC*Plag1*CC'+R; % B(t) update
      K=Plag1*(CC'/B); % K(t)
      Plag0=Plag1-K*CC*Plag1;Plag0% P(t|t) update
      xlag1=F*xlag0; % x(t|t-1) update
      e=z(t)-CC*xlag1; % residual (innovation)e(t|t-1)
      xlag0=xlag1+K*e; %xlag0 % x(t|t) update
      x1(t,2)=xlag0(1);
      x2(t,2)=xlag0(2);
      x3(t,2)=xlag0(3);
   end
   cumPlag0=((c-1)/c)*cumPlag0+(x-xlag0)*(x-xlag0)'/c; %empirical estimate of Plag0 at terminal time
end
%Lcum %temp
%Plag0 

    %computation of first derivatives for tj  
    %if ti==tj %avoid repeating the same computation if ti,tj are identical
    %    dPlag1j=dPlag1i;
    %    dBj=dBi;
    %    dPlag0j=dPlag0i;
    %    dxlag1j=dxlag1i;
    %    dej=dei;
    %    dxlag0j=dxlag0i;
    %else        
    %    dPlag1j=A*dPlag0j*A'+dQj;
    %    dBj=C*dPlag1j*C';
    %    dPlag0j=dPlag1j-dPlag1j*C'*B^(-1)*C*Plag1+Plag1*C'*B^(-1)*dBj*B^(-1)*C*Plag1-Plag1*C'*B^(-1)*C*dPlag1j;
    %    dxlag1j=A*dxlag0j;
    %    dej=-C*dxlag1j;
    %    dxlag0j=dxlag1j+dPlag1j*C'*B^(-1)*e-Plag1*C'*B^(-1)*dBj*B^(-1)*e+Plag1*C'*B^(-1)*dej;
    %end
    %Instantaneous gradient
%end