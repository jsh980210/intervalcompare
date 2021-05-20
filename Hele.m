function Hij=Hele(N,A,C,Q,R,mu0,P0,ti,tj,y) %ti, tj denote the element of the parameter vector
%compute dQ/dti, dQ/dtj
dQi=zeros(3,3);
dQj=zeros(3,3);
dQi(ti,ti)=1;
dQj(tj,tj)=1;
Plag0temp=P0; %info for starting point x0
xlag0temp=mu0;
dPlag0tempi=zeros(3,3);%initialization for first derivative for ti
dxlag0tempi=zeros(3,1);
dPlag0tempj=zeros(3,3);%initialization for first derivative for tj
dxlag0tempj=zeros(3,1);
ddPlag0tempij=zeros(3,3);%initializaing second derivative for ti tj
ddxlag0tempij=zeros(3,1);
Hij=0;%initialization of hessian element
for t=1:N
    Plag1temp=A*Plag0temp*A'+Q;%using kalman filter equations 39-43 in G & N
    Ftemp=C*Plag1temp*C'+R;
    Ktemp=Plag1temp*C'*Ftemp^(-1);
    Plag0temp=Plag1temp-Ktemp*C*Plag1temp;
    xlag1temp=A*xlag0temp;
    etemp=y(t)-C*xlag1temp;
    xlag0temp=xlag1temp+Ktemp*etemp;
    %computation of first derivatives for ti (formula derived by XC)
    dPlag1tempi=A*dPlag0tempi*A'+dQi;
    dFtempi=C*dPlag1tempi*C';        
    dPlag0tempi=dPlag1tempi-dPlag1tempi*C'*Ftemp^(-1)*C*Plag1temp+Plag1temp*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*C*Plag1temp-Plag1temp*C'*Ftemp^(-1)*C*dPlag1tempi;
    dxlag1tempi=A*dxlag0tempi;
    detempi=-C*dxlag1tempi;
    dxlag0tempi=dxlag1tempi+dPlag1tempi*C'*Ftemp^(-1)*etemp-Plag1temp*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*etemp+Plag1temp*C'*Ftemp^(-1)*detempi;
    %computation of first derivatives for tj  (formula derived by XC)
    if ti==tj %avoid repeating the same computation if ti,tj are identical
        dPlag1tempj=dPlag1tempi;
        dFtempj=dFtempi;
        dPlag0tempj=dPlag0tempi;
        dxlag1tempj=dxlag1tempi;
        detempj=detempi;
        dxlag0tempj=dxlag0tempi;
    else        
        dPlag1tempj=A*dPlag0tempj*A'+dQj;
        dFtempj=C*dPlag1tempj*C';
        dPlag0tempj=dPlag1tempj-dPlag1tempj*C'*Ftemp^(-1)*C*Plag1temp+Plag1temp*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*C*Plag1temp-Plag1temp*C'*Ftemp^(-1)*C*dPlag1tempj;
        dxlag1tempj=A*dxlag0tempj;
        detempj=-C*dxlag1tempj;
        dxlag0tempj=dxlag1tempj+dPlag1tempj*C'*Ftemp^(-1)*etemp-Plag1temp*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*etemp+Plag1temp*C'*Ftemp^(-1)*detempj;
    end
    %computation of second derivatives for ti tj (formula derived by XC)
    ddPlag1tempij=A*ddPlag0tempij*A';
    ddFtempij=C*ddPlag1tempij*C';
    ddPlag0tempij=ddPlag1tempij-ddPlag1tempij*C'*Ftemp^(-1)*C*Plag1temp+dPlag1tempi*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*C*Plag1temp...
                 -dPlag1tempi*C'*Ftemp^(-1)*C*dPlag1tempj+dPlag1tempj*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*C*Plag1temp...
                 -Plag1temp*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*dFtempi*Ftemp^(-1)*C*Plag1temp...
                 +Plag1temp*C'*Ftemp^(-1)*ddFtempij*Ftemp^(-1)*C*Plag1temp...
                 -Plag1temp*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*dFtempj*Ftemp^(-1)*C*Plag1temp...
                 +Plag1temp*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*C*dPlag1tempj-dPlag1tempj*C'*Ftemp^(-1)*C*dPlag1tempi...
                 +Plag1temp*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*C*dPlag1tempi-Plag1temp*C'*Ftemp^(-1)*C*ddPlag1tempij;
    ddxlag1tempij=A*ddxlag0tempij;
    ddetempij=-C*ddxlag1tempij;
    ddxlag0tempij=ddxlag1tempij+ddPlag1tempij*C'*Ftemp^(-1)*etemp-dPlag1tempi*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*etemp...
                 +dPlag1tempi*C'*Ftemp^(-1)*detempj-dPlag1tempj*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*etemp...
                 +Plag1temp*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*dFtempi*Ftemp^(-1)*etemp...
                 -Plag1temp*C'*Ftemp^(-1)*ddFtempij*Ftemp^(-1)*etemp...
                 +Plag1temp*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*dFtempj*Ftemp^(-1)*etemp...
                 -Plag1temp*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*detempj+dPlag1tempj*C'*Ftemp^(-1)*detempi...
                 -Plag1temp*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*detempi+Plag1temp*C'*Ftemp^(-1)*ddetempij;
    %computation of Hessian elements (formula derived by XC)
    Hij=Hij-0.5*((-Ftemp^(-1)*dFtempj*Ftemp^(-1)*dFtempi+Ftemp^(-1)*ddFtempij)*(1-Ftemp^(-1)*etemp^2))...
            -0.5*(Ftemp^(-1)*dFtempi*Ftemp^(-1)*dFtempj*Ftemp^(-1)*etemp^2)...
            +0.5*(Ftemp^(-1)*dFtempi*Ftemp^(-1)*(detempj*etemp+etemp*detempj))...
            -ddetempij*Ftemp^(-1)*etemp+detempi*dFtempj*Ftemp^(-2)*etemp-detempi*Ftemp^(-1)*detempj;
end