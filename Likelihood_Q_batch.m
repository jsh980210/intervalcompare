function Lcum=Likelihood_Q_batch(z,Q) %negative log-likelihood function
global F C R dimx dQ Plag0 xlag0 ndata; %,ti,tj;
Lcum=0;
% Note to reset Plag0 and xlag0 manually if calling independent of driver program

for t=1:ndata; %temp: should be ndata when finalized
    Plag1=F*Plag0*F'+Q;%Kalman filter equations: P(t+1|t) update
    CC=C(:,:,:,t-(floor((t-1)/3))*3)'; %cycles through 3 possible C values over each 3 data points; recall Matlab treats C(:,:,:,t-(floor((t-1)/3))*3) as COLUMN vector!
    B=CC*Plag1*CC'+R; % B(t) update
    K=Plag1*(CC'/B); % K(t)
    Plag0=Plag1-K*CC*Plag1;% P(t|t) update
    xlag1=F*xlag0; % x(t|t-1) update
    e=z(t)-CC*xlag1; % residual (innovation)e(t|t-1)
    xlag0=xlag1+K*e; %xlag0 % x(t|t) update
    Lcum=(t-1)*Lcum/t+0.5*((e^2)/B+log(B))/t;
end
%z(1) %temp
%z(2) %temp
%xlag0