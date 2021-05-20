function Qhat=algB(N,A,C,Q,R,mu0,P0,Q0,y,M,errorstd)% M is the # of iterations for algorithm B, errorstd is the std for perturbation
i=0;
Qhat=Q0.^(0.5);%initialization of Qhat, which is the sqrt of Q matrix
while i<=M
    di=diag(normrnd(0,errorstd,1,3)); %diagonal perturbation matrix
    Qtemp=Qhat+di;%generate the candidate Qtemp
    if likelihood(N,A,C,Qtemp^2,R,mu0,P0,y)>likelihood(N,A,C,Qhat^2,R,mu0,P0,y)%notice Qtemp or Qhat is the sqrt of Q matrix
        Qhat=Qtemp;
    end
    i=i+1;
end
Qhat=Qhat^2;%square the Qhat up as an estimate of Q.   
    
