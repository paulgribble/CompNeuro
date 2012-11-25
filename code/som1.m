clear; clf;

% Self-organizing feature map in two dimensions
n_out= 20;  % number of nodes in the output layer
k=.01;      % learning rate
sig=1.5;      % width of neighborhood function
sig_inv=1/(2*sig^2);
[I,J] = meshgrid(1:n_out, 1:n_out);

r=zeros(n_out);

% design some random non-uniform clusters of data
R = [randn(2,100)*.08 + .3, randn(2,100)*.1 + .6, ...
    randn(2,100)*.05 + repmat([.2; .7],1,100)];
plot(R(1,:),R(2,:),'r.');
axis([-.1 1.1 -.1 1.1]);
hold on

% initialize weights on a regular grid
for i=1:n_out
    for j=1:n_out
        w1(i,j)=(i/n_out)*1;
        w2(i,j)=(j/n_out)*1;
    end;
end;

hold on;
wp1=plot(w1,w2,'k');
wp2=plot(w1',w2','k');
axis([-.1 1.1 -.1 1.1]); xlabel('w1'); ylabel('w2');
drawnow
pause

% iterate and adjust weights
for epochs=1:500;
    for i_example=1:size(R,2)
        r_in=R(:,i_example);
        % calculate winner
        r=exp(-(w1-r_in(1)).^2-(w2-r_in(2)).^2);
        [rmax,i_winner]=max(max(r));
        [rmax,j_winner]=max(max(r'));
        % update weight vectors using neighborhood function (sig_inv)
        r=exp(-((I-i_winner).^2+(J-j_winner).^2)*sig_inv);
        w1=w1+k*r.*(r_in(1)-w1);
        w2=w2+k*r.*(r_in(2)-w2);
    end;
    delete(wp1); delete(wp2);
    wp1=plot(w1,w2,'k');
    wp2=plot(w1',w2','k');
    axis([-.1 1.1 -.1 1.1]);
    title(num2str(epochs));
    drawnow
    pause(0.05);
end;

