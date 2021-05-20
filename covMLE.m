function c=covMLE(N,A,C,Q,R,mu0,P0,Q0,M,errorstd,rep)%rep stands for the number of MLEs generated for the approximation of cov(MLE)
MLE=zeros(rep,3);%store all MLEs
for i=1:rep
    y=obs(N,A,C,Q,R,mu0,P0);% for each independent replication, generate a new sequence of observations
    Qhat=algB(N,A,C,Q,R,mu0,P0,Q0,y,M,errorstd);
    MLE(i,:)=diag(Qhat)';%take the diagonal elements of Qhat as the MLE
   
end
c=cov(MLE);

