function L=likelihood(N,A,C,Q,R,mu0,P0,y)
Plag0temp=P0;
xlag0temp=mu0;
L=0;% initialization of the loss value
for t=1:N 
    Plag1temp=A*Plag0temp*A'+Q;%using kalman filter equations 39-43 in G & N 
    Ftemp=C*Plag1temp*C'+R;
    Ktemp=Plag1temp*C'*Ftemp^(-1);
    Plag0temp=Plag1temp-Ktemp*C*Plag1temp;
    xlag1temp=A*xlag0temp;
    etemp=y(t)-C*xlag1temp;
    xlag0temp=xlag1temp+Ktemp*etemp;
    L=L-0.5*(log(det(Ftemp))+etemp*Ftemp^(-1)*etemp);%using the likelihood function (18) in G & N
end