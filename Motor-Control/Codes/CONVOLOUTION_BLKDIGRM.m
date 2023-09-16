close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 4);

F1 = [Null Learn unlearn Learn];

% Single state model
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n));
end

% Gain specific model
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x1_g(n)));
    x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x2_g(n)));
end
x_g = cconv(x1_g,x2_g,554)



unlearn = -ones(1, 5);
% Multi-rate model
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
k =4;
for n = 1:length(F1)-1
    
    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
    x_m(n+1) = k*x1_m(n+1) + x2_m(n+1);
end


% Plot the state over time for each model
figure;
subplot(3,1,1)
plot(F1, 'b', 'linewidth', 1);
hold on;
plot(x_s, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Single State Model');
legend('Input', 'State');

subplot(3,1,2)
plot(F1, 'b', 'linewidth', 1);
hold on;
plot(x1_g, 'g', 'linewidth', 1);
plot(x2_g, 'c', 'linewidth', 1);
plot(x_g, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Gain Specific Model');
legend('Input', 'x1_g', 'x2_g', 'State');

subplot(3,1,3)
plot(F1, 'b', 'linewidth', 1);
hold on;
plot(x1_m, 'g', 'linewidth', 1);
plot(x2_m, 'c', 'linewidth', 1);
plot(x_m, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model');
legend('Input', 'x1_m', 'x2_m', 'State');
