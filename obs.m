function y=obs(N,A,C,Q,R,mu0,P0) %mu0 P0 describe the starting point x0
y=zeros(1,N);
xtemp=mvnrnd(mu0,P0)';% xtemp is the current state vector
for i=1:N
    xtemp=A*xtemp+mvnrnd([0 0 0]',Q)';% state evolving step
    y(i)=C*xtemp+normrnd(0,sqrt(R)); % observation step
end