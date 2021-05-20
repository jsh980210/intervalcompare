function [Hinv,Finv]=HF(N,A,C,Q,R,mu0,P0,y)
H=zeros(3,3);
F=zeros(3,3);
for i=1:3 %generate H and F element by element (using symmetry)
    for j=i:3
        H(i,j)=Hele(N,A,C,Q,R,mu0,P0,i,j,y);
        F(i,j)=Fele(N,A,C,Q,R,mu0,P0,i,j);
    end
end
H=H+H'-diag(diag(H));
F=F+F'-diag(diag(F));
Hinv=(-H)^(-1);
Finv=F^(-1);





