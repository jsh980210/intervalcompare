A=[0,1,0;0,0,1;0.8,0.8,-0.8];
C=[1,0,0];
R=1;
mu=[0;0;0];
Sigma=[0,0,0;0,0,0;0,0,0];
N=100;
mu0=[0;0;0];
P0=[1,0,0;0,1,0;0,0,1];
Q=[1,0,0;0,1,0;0,0,1];
Q0=[0.95,0,0;0,0.95,0;0,0,0.95];
rep=1000;
errorstd=1;
M=10;

%V=covMLE(N,A,C,Q,R,mu0,P0,Q0,M,errorstd,rep)
%[Hinv,Finv]=HF(N,A,C,Q,R,mu0,P0,y)
%[MSEH,MSEF,ratio]=comparison(N,A,C,Q,R,mu0,P0,Q0,M,errorstd,V,rep)
MLE=zeros(rep,3);
Hinv=cell(rep,1);
Finv=cell(rep,1);
Htemp=zeros(3);
Ftemp=zeros(3);
i=1;
while i<=rep
    y=obs(N,A,C,Q,R,mu0,P0);%generate observation
    Qhat=algB(N,A,C,Q,R,mu0,P0,Q0,y,M,errorstd);%calculation of MLE Qhat
    
    [Htemp,Ftemp]=HF(N,A,C,Qhat,R,mu0,P0,y);
    if Htemp(1,1)>0 && Htemp(2,2)>0 && Htemp(3,3)>0
        fprintf('Current number is %d\n',i);
        Hinv{i}=Htemp;
        Finv{i}=Ftemp;
        MLE(i,:)=diag(Qhat)';
        i=i+1;
        
    end
    
end
V=cov(MLE);
MSEH=zeros(1,3);%mean squared error matrix for H
MSEF=zeros(1,3);%mean squared error matrix for F

for j=1:rep
    MSEH(1)=MSEH(1)+(2*normcdf(1.96*sqrt(abs(Hinv{j}(1,1))/V(1,1)))-1-0.95)^2;%calculate MSE componentwise
    MSEH(2)=MSEH(2)+(2*normcdf(1.96*sqrt(abs(Hinv{j}(2,2))/V(2,2)))-1-0.95)^2;%calculate MSE componentwise
    MSEH(3)=MSEH(3)+(2*normcdf(1.96*sqrt(abs(Hinv{j}(3,3))/V(3,3)))-1-0.95)^2;%calculate MSE componentwise
    MSEF(1)=MSEF(1)+(2*normcdf(1.96*sqrt(Finv{j}(1,1)/V(1,1)))-1-0.95)^2;%calculate MSE componentwise
    MSEF(2)=MSEF(2)+(2*normcdf(1.96*sqrt(Finv{j}(2,2)/V(2,2)))-1-0.95)^2;%calculate MSE componentwise
    MSEF(3)=MSEF(3)+(2*normcdf(1.96*sqrt(Finv{j}(3,3)/V(3,3)))-1-0.95)^2;%calculate MSE componentwise
end
MSEH=MSEH/rep;
MSEF=MSEF/rep;
MSEdiff=MSEH./MSEF%difference matrix

nf=zeros(1,rep);
nh=zeros(1,rep);

for k=1:rep
    nh(k)=norm(Hinv{k}-V,'fro');
    nf(k)=norm(Finv{k}-V,'fro');
end
[~, indices1] = sort(nh);
typical_H=Hinv{indices1(500)};

[~, indices2] = sort(nf);
typical_F=Finv{indices2(500)};
