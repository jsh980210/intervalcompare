function Gcum=Gradient_batch(z,Q) %ti, tj denote the element of the parameter vector
global F C R dimx dQ Plag0 xlag0 dPlag0 dxlag0 ndata; %,ti,tj;
%Gradient_batch
%G is output vector representing instantaneous gradient
%compute dQ/dti, dQ/dtj; loops over m = 1:dimx assume dim(theta) = dimx
for m=1:dimx
    dPlag0(:,:,m)=zeros(dimx,dimx);%initialization for first derivative for tm; this initialization should be placed in driver program for SG instantaneous
    dxlag0(:,m)=zeros(dimx,1);
end
G=zeros(dimx,1);
Gcum=zeros(dimx,1); %hard-coded for dim of theta = dimx
%Below reflects one epoch through one realization of measurements: Q is diag matrix of entries to be estimated
%dQ = zeros(3,3,3); %deriv. Q w.r.t. theta
%for m=1:dimx
%    dQ(m,m,m)=1; 
%end
%Plag0
for t=1:ndata %temp as 2 not ndata
    Plag1=F*Plag0*F'+Q;%Kalman filter equations: P(t+1|t) update
    CC=C(:,:,:,t-(floor((t-1)/3))*3)'; %cycles through 3 possible C values over each 3 data points; recall Matlab treats C(:,:,:,t-(floor((t-1)/3))*3) as COLUMN vector!
    B=CC*Plag1*CC'+R; % B(t) update
    K=Plag1*(CC'/B); % K(t)
    Plag0=Plag1-K*CC*Plag1;% P(t|t) update
    xlag1=F*xlag0; % x(t|t-1) update
    e=z(t)-CC*xlag1; % residual (innovation)e(t|t-1)
    xlag0=xlag1+K*e; %xlag0 % x(t|t) update
    
    %Plag1 %temp
    %z(1) %temp
    %z(2) %temp
    for m=1:dimx
        %computation of first derivatives for ti 
        dPlag1=F*dPlag0(:,:,m)*F'+dQ(:,:,m); %DP(t+1|t) (D = derivative w.r.t. theta component) (dQ(:,:,i) = zeros matrix except for 1 in the ii spot)
        dB=CC*dPlag1*CC';
        dK=(eye(dimx)-K*CC)*dPlag1*CC'/B;
        %dPlag0=dPlag1-dPlag1*CC'*(B\CC)*Plag1+Plag1*CC'*(B\dB)*(B\CC)*Plag1-Plag1*CC'*(B\CC)*dPlag1;dPlag0 %DP(t|t)
        %dPlag0=(eye(dimx)-K*CC)*dPlag1*(eye(dimx)-(CC'/B)*CC*Plag1);%dPlag0; formula produces identical results to above line; this version removed 1/31/14
        dPlag0(:,:,m)=(eye(dimx)-K*CC)*dPlag1-dK*CC*Plag1;
        dxlag1=F*dxlag0(:,m);%Dx(t|t-1)
        de=-CC*dxlag1;%De(t|t-1)
        %dxlag0=dxlag1+dPlag1*CC'*(B\e)-Plag1*CC'*(B\dB)*(B\e)+Plag1*CC'*(B\de);dxlag0 %Dx(t|t)
       % dxlag0=dxlag1+K*de+(eye(dimx)-K*CC)*dPlag1*(CC'/B)*e;%this version removed 1/31/14
        dxlag0(:,m)=dxlag1+K*de+dK*e;%updated filter derivative for mth component
        G(m)=(e'/B)*de-0.5*(e'/B)*dB*(e'/B)'+0.5*trace(B\dB);(e'/B)*de; %G(m)
    end
    Gcum=(t-1)*Gcum/t+G/t; 
end
%xlag0
