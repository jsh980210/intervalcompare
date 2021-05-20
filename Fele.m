function Fij=Fele(N,A,C,Q,R,mu0,P0,ti,tj)
%compute dQ/dti, dQ/dtj
dQi=zeros(3,3);
dQj=zeros(3,3);
dQi(ti,ti)=1;
dQj(tj,tj)=1;
%compute E(ykyl') matrix according to formula (3.14) in Cavanaugh
Eykyl=zeros(N+3,N+3);% the initial entries are of greater dimension, thus the matrix dimension exceeds N
for k=0:N
    for l=0:N
        if k==0
            if l==0
                Eykyl([1:3],[1:3])=mu0*mu0';
            else
                Eykyl([1:3],l+3)=mu0*mu0'*(A')^l*C';
            end
        else
            if l==0
                Eykyl(k+3,[1:3])=C*A^k*mu0*mu0';
            else
                if k==l
                    sum=0;
                    for s=0:k-1
                        sum=sum+A^s*Q*(A')^s;
                    end
                    Eykyl(k+3,l+3)=C*sum*C'+C*A^k*(P0+mu0*mu0')*(A')^k*C'+R;
                else if k<l
                        sum=0;
                        for s=0:k-1
                            sum=sum+A^s*Q*(A')^(s+l-k);
                        end
                        Eykyl(k+3,l+3)=C*sum*C'+C*A^k*(P0+mu0*mu0')*(A')^l*C';
                    else
                        sum=0;
                        for s=0:l-1
                            sum=sum+A^(s+k-l)*Q*(A')^s;
                        end
                        Eykyl(k+3,l+3)=C*sum*C'+C*A^k*(P0+mu0*mu0')*(A')^l*C';
                    end
                end
            end
        end
    end
end
Plag0temp=P0; %info for starting point x0
xlag0temp=mu0;%info for starting point x0
Rtemp=zeros(3,3+N);%3 by 3 followed by 3 by 1, initialize Rtemp with the maximum dimension when t=N
Ktemp=eye(3); %different dimension for initial Kt
dPlag0tempi=zeros(3,3);%initialization for first derivative for ti
dxlag0tempi=zeros(3,1);
dRtempi=zeros(3,3+N);
dKtempi=zeros(3,3);
dPlag0tempj=zeros(3,3);%initialization for first derivative for tj
dxlag0tempj=zeros(3,1);
dRtempj=zeros(3,3+N);
dKtempj=zeros(3,3);
Fij=0;%initialization of Fisher element
for t=1:N
    % computation of Rtemp and dRtemp for each value of t according to
    % Cavanaugh 3.4-6 and 3.10-12
    if t==1 %special case of t=1
        dRtempi(:,4)=0;
        dRtempi(:,[1:3])=A*dKtempi;
        dRtempj(:,4)=0;
        dRtempj(:,[1:3])=A*dKtempj;
        Rtemp(:,4)=1;
        Rtemp(:,[1:3])=A*Ktemp;
    else % general case when t>1
        dRtempi(:,t+3)=0;
        dRtempi(:,t+2)=A*dKtempi;
        dRtempj(:,t+3)=0;
        dRtempj(:,t+2)=A*dKtempj;
        Rtemp(:,t+3)=1;
        Rtemp(:,t+2)=A*Ktemp;
        for k=0:(t-2)
            if k==0
                dRtempi(:,[1:3])=-A*(dKtempi*C)*Rtemp(:,[1:3])+A*(eye(3)-Ktemp*C)*dRtempi(:,[1:3]);
                dRtempj(:,[1:3])=-A*(dKtempj*C)*Rtemp(:,[1:3])+A*(eye(3)-Ktemp*C)*dRtempj(:,[1:3]);
                Rtemp(:,[1:3])=A*(eye(3)-Ktemp*C)*Rtemp(:,[1:3]);
            else
                dRtempi(:,k+3)=-A*(dKtempi*C)*Rtemp(:,k+3)+A*(eye(3)-Ktemp*C)*dRtempi(:,k+3);
                dRtempj(:,k+3)=-A*(dKtempj*C)*Rtemp(:,k+3)+A*(eye(3)-Ktemp*C)*dRtempj(:,k+3);
                Rtemp(:,k+3)=A*(eye(3)-Ktemp*C)*Rtemp(:,k+3);
            end
        end
    end
    %computation of Kalman filter
    Plag1temp=A*Plag0temp*A'+Q;
    Ftemp=C*Plag1temp*C'+R;
    Ktemp=Plag1temp*C'*Ftemp^(-1);
    Plag0temp=Plag1temp-Ktemp*C*Plag1temp;
    %computation of first derivatives for ti
    dPlag1tempi=A*dPlag0tempi*A'+dQi;
    dFtempi=C*dPlag1tempi*C';
    dKtempi=dPlag1tempi*C'*Ftemp^(-1)-Plag1temp*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1);
    dPlag0tempi=dPlag1tempi-dPlag1tempi*C'*Ftemp^(-1)*C*Plag1temp+Plag1temp*C'*Ftemp^(-1)*dFtempi*Ftemp^(-1)*C*Plag1temp-Plag1temp*C'*Ftemp^(-1)*C*dPlag1tempi;
    %computation of first derivatives for tj
    if ti==tj %if ti tj are identical, copy everything from ti
        dPlag1tempj=dPlag1tempi;
        dFtempj=dFtempi;
        dKtempj=dKtempi;
        dPlag0tempj=dPlag0tempi;
    else        
        dPlag1tempj=A*dPlag0tempj*A'+dQj;
        dFtempj=C*dPlag1tempj*C';
        dKtempj=dPlag1tempj*C'*Ftemp^(-1)-Plag1temp*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1);
        dPlag0tempj=dPlag1tempj-dPlag1tempj*C'*Ftemp^(-1)*C*Plag1temp+Plag1temp*C'*Ftemp^(-1)*dFtempj*Ftemp^(-1)*C*Plag1temp-Plag1temp*C'*Ftemp^(-1)*C*dPlag1tempj;
    end
    %computation of Fisher elements
    ss=0;%ss denotes the double summation in (3.13)
    for k=0:t
        for l=0:t
            if k==0
                if l==0
                    ss=ss+C*dRtempi(:,[1:3])*Eykyl([1:3],[1:3])*dRtempj(:,[1:3])'*C';
                else
                    ss=ss+C*dRtempi(:,[1:3])*Eykyl([1:3],l+3)*dRtempj(:,l+3)'*C';
                end
            else
                if l==0
                    ss=ss+C*dRtempi(:,k+3)*Eykyl(k+3,[1:3])*dRtempj(:,[1:3])'*C';
                else
                    ss=ss+C*dRtempi(:,k+3)*Eykyl(k+3,l+3)*dRtempj(:,l+3)'*C';
                end
            end
        end
    end
    Fij=Fij+0.5*(Ftemp^(-1)*dFtempi*Ftemp^(-1)*dFtempj)+Ftemp^(-1)*ss;%according to (3.13) in Cavanaugh
end