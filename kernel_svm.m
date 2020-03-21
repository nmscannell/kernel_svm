clear all;
load('SVM_data_nonlinear.mat', 'x', 'y');
[m, n] = size(x);
power = 3;

k = kernel(x, x, m, m, power);
H = diag(y)*k*diag(y);
f = -(ones(m, 1));
A = y';
b = 0;
l = zeros(m, 1);
alpha = quadprog(H,f,[],[],A,b,l,[]);
mask = alpha>10^(-6);
alpha = alpha.*mask;
indices = (1:m);
svs=indices(alpha>0)';

k = kernel(x, x(55,:), m, length(x(55)), power);
b = 0;
for i = 1:length(svs)
    b = b+alpha(svs(i))*y(svs(i))*k(svs(i));
end
b = 1-b;

x1range = [min(x(:,1))-1, max(x(:,1))+1];
x2range = [min(x(:,2))-1, max(x(:,2))+1];
d = 0.05;
[x1Grid,x2Grid] = meshgrid(x1range(1):d:x1range(2),...
    x2range(1):d:x2range(2));
xGrid = [x1Grid(:) x2Grid(:)];

k = kernel(x, xGrid, m, length(xGrid), power);
y_p = sum(diag(alpha.*y)*k,1)' + b*ones(length(xGrid),1);
figure()
contour(x1Grid,x2Grid,reshape(y_p,size(x1Grid)),[0 0],'k');
hold on;
set(gca, 'ydir', 'reverse');
plot(x(1:m/2,1), x(1:m/2,2),'+');
for i=1:length(svs)
    plot(x(svs(i),1), x(svs(i),2),'ko');
end
plot(x((m/2)+1:m,1),x((m/2)+1:m,2),'r*');
ylim([-5,4]);
xlim([-5,4]);
pbaspect([1 1 1])

function k = kernel(x1, x2, l1, l2, power)
    k = zeros(l1, l2);
    for i=1:l1
        for j=1:l2
            k(i,j) = (1+dot(x1(i,:),x2(j,:))).^power;
        end
    end
end