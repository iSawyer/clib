m=50; n=200;
A = randn(m,n);	
nz=round(0.05*n);
z=n-nz;
ub= rand(n,1);
b=randperm(n);
ub(b(1:z))=0 ;	
f = A*ub;
fd = fopen('A','w');
for i = 1 : m
    for j = 1 : n
        fprintf(fd,"%f ",A(i,j));
    end
end
fclose(fd);

fd = fopen('y','w');
for i = 1 : m
    fprintf(fd,"%f ",f(i));
end
fclose(fd);

fd = fopen('x','w');
for i = 1 : n
    fprintf(fd,"%f ",ub(i));
end
fclose(fd);


