close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 50);

F1 = [Null Learn unlearn Learn];

% Single state model
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n));
end

% Gain specific model
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
x_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n)));
    x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n)));
    x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end

% Multi-rate model
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
    x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end

% Plot the state over time for each model
figure;
subplot(3,1,1)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+29 N+29 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+nd_s 2*N 2*N N+nd_s], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
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
plot(x_g, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Gain Specific Model');
legend('Input', 'State');

subplot(3,1,3)
plot(F1, 'b', 'linewidth', 1);
hold on;
plot(x_m, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model');
legend('Input', 'State');
%% SIMULATION MODELS
close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because paper did it and if we se the
% large value for it we goes to negetive value!!!
unlearn = -ones(1, 18);

F1 = [Null Learn unlearn Learn];

% Single state model
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n));
end

unlearn = -ones(1, 10);

% Gain specific model
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
x_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n)));
    x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n)));
    x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end


unlearn = -ones(1, 23);
% Multi-rate model
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
    x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end
unlearn = -ones(1, 29);
% Plot the state over time for each model
figure;
subplot(3,1,1)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x_s, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Single State Model');
legend('Input', 'Net Adaptation', 'location', 'southeast');
xlim([0,550])

unlearn = -ones(1, 10);
subplot(3,1,2)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x1_g,'--','color', 'g', 'linewidth', 1);
plot(x2_g,'--','color', 'b', 'linewidth', 1);
plot(x_g, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Gain Specific Model');
legend('Input', 'Up State', 'Down State', 'Net Adaptation', 'location', 'southeast');
xlim([0,550])

unlearn = -ones(1, 20);
subplot(3,1,3)
N = 276;
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'K', 'linewidth', 2);
hold on;
plot(x1_m,'--','color', 'g', 'linewidth', 1);
plot(x2_m,'--','color', 'b', 'linewidth', 1);
plot(x_m, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model');
legend('Input', 'Fast State', 'Slow State', 'Net Adaptation',  'location', 'southeast');
xlim([0,550])

%% DAYNAMIC PARAMETRS

%% SIMULATION MODELS - Learning and Re-Learning Plots
close all
clc

% Parameters
A_values = [0.95, 0.99, 1.03];  % New values for A
B_values = [0.01, 0.013, 0.016];  % New values for B
Af_values = [0.9, 0.92, 0.94];  % New values for Af
Bf_values = [0.02, 0.03, 0.04];  % New values for Bf
As_values = [0.992, 0.996, 1.0];  % New values for As
Bs_values = [0.002, 0.004, 0.006];  % New values for Bs

% Randomly choose 12 parameter combinations
rng(123);  % Set random seed for reproducibility
indices = randperm(numel(A_values)*numel(B_values)*numel(Af_values)*numel(Bf_values)*numel(As_values)*numel(Bs_values), 12);

figure_counter = 1;

for idx = indices
    [i, j, k, l, m, n] = ind2sub([numel(A_values), numel(B_values), numel(Af_values), numel(Bf_values), numel(As_values), numel(Bs_values)], idx);
    
    A = A_values(i);
    B = B_values(j);
    Af = Af_values(k);
    Bf = Bf_values(l);
    As = As_values(m);
    Bs = Bs_values(n);

    % Trial configuration
    null = 25;
    learn = 250;

    Null = zeros(1, null);
    Learn = ones(1, learn);
    unlearn = -ones(1, 18);

    F1 = [Null Learn unlearn Learn];

    % Single state model - Learning and Re-Learning
    x_s = zeros(size(F1));
    for p = 1:length(F1)-1
        x_s(p+1) = A * x_s(p) + B * (F1(p) - x_s(p));
    end

    % Gain specific model - Learning and Re-Learning
    x1_g = zeros(size(F1));
    x2_g = zeros(size(F1));
    x_g = zeros(size(F1));
    for p = 1:length(F1)-1
        x1_g(p+1) = min(0, A * x1_g(p) + B * (F1(p) - x_g(p)));
        x2_g(p+1) = max(0, A * x2_g(p) + B * (F1(p) - x_g(p)));
        x_g(p+1) = x1_g(p+1) + x2_g(p+1);
    end

    % Multi-rate model - Learning and Re-Learning
    x1_m = zeros(size(F1));
    x2_m = zeros(size(F1));
    x_m = zeros(size(F1));
    for p = 1:length(F1)-1
        x1_m(p+1) = Af * x1_m(p) + Bf * (F1(p) - x_m(p));
        x2_m(p+1) = As * x2_m(p) + Bs * (F1(p) - x_m(p));
        x_m(p+1) = x1_m(p+1) + x2_m(p+1);
    end

    % Plot the Learning and Re-Learning curves for each model
    figure(figure_counter);
    subplot(3,1,1)
    plot(x_s(26:275), 'b', 'linewidth', 1);
    hold on;
    plot(x_s(300:535),'--','color', 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Single State Model (A = %.2f, B = %.3f)', A, B));
    legend('Learning', 'Re-Learning', 'location', 'southeast');

    subplot(3,1,2)
    plot(x_g(26:275), 'b', 'linewidth', 1);
    hold on;
    plot(x_g(300:535), '--','color','r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Gain Specific Model (A = %.2f, B = %.3f)', A, B));
    legend('Learning', 'Re-Learning', 'location', 'southeast');

    subplot(3,1,3)
    plot(x_m(26:275), 'b', 'linewidth', 1);
    hold on;
    plot(x_m(300:535), '--','color','r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Multi Rate Model (Af = %.2f, Bf = %.3f, As = %.3f, Bs = %.3f)', Af, Bf, As, Bs));
    legend('Learning', 'Re-Learning', 'location', 'southeast');

    figure_counter = figure_counter + 1;
end

%% SIMULATION MODELS
close all
clc

% Parameters
A_values = [0.95, 0.99, 1.03];  % New values for A
B_values = [0.01, 0.013, 0.016];  % New values for B
Af_values = [0.9, 0.92, 0.94];  % New values for Af
Bf_values = [0.02, 0.03, 0.04];  % New values for Bf
As_values = [0.992, 0.996, 1.0];  % New values for As
Bs_values = [0.002, 0.004, 0.006];  % New values for Bs

% Randomly choose 12 parameter combinations
rng(123);  % Set random seed for reproducibility
indices = randperm(numel(A_values)*numel(B_values)*numel(Af_values)*numel(Bf_values)*numel(As_values)*numel(Bs_values), 12);

figure_counter = 1;

for idx = indices
    [i, j, k, l, m, n] = ind2sub([numel(A_values), numel(B_values), numel(Af_values), numel(Bf_values), numel(As_values), numel(Bs_values)], idx);
    
    A = A_values(i);
    B = B_values(j);
    Af = Af_values(k);
    Bf = Bf_values(l);
    As = As_values(m);
    Bs = Bs_values(n);

    % Trial configuration
    null = 25;
    learn = 250;

    Null = zeros(1, null);
    Learn = ones(1, learn);
    unlearn = -ones(1, 18);

    F1 = [Null Learn unlearn Learn];

    % Single state model
    x_s = zeros(size(F1));
    for p = 1:length(F1)-1
        x_s(p+1) = A * x_s(p) + B * (F1(p) - x_s(p));
    end

    unlearn = -ones(1, 10);

    % Gain specific model
    x1_g = zeros(size(F1));
    x2_g = zeros(size(F1));
    x_g = zeros(size(F1));
    for p = 1:length(F1)-1
        x1_g(p+1) = min(0, A * x1_g(p) + B * (F1(p) - x_g(p)));
        x2_g(p+1) = max(0, A * x2_g(p) + B * (F1(p) - x_g(p)));
        x_g(p+1) = x1_g(p+1) + x2_g(p+1);
    end

    unlearn = -ones(1, 29);

    % Multi-rate model
    x1_m = zeros(size(F1));
    x2_m = zeros(size(F1));
    x_m = zeros(size(F1));
    for p = 1:length(F1)-1
        x1_m(p+1) = Af * x1_m(p) + Bf * (F1(p) - x_m(p));
        x2_m(p+1) = As * x2_m(p) + Bs * (F1(p) - x_m(p));
        x_m(p+1) = x1_m(p+1) + x2_m(p+1);
    end

    % Plot the state over time for each model
    figure(figure_counter);
    subplot(3,1,1)
    N = 276;
    patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    hold on;
    patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none', 'handlevisibility', 'off')
    patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    plot(F1, 'k', 'linewidth', 2);
    hold on;
    plot(x_s, 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Single State Model (A = %.2f, B = %.3f)', A, B));
    legend('Input', 'Net Adaptation', 'location', 'southeast');
    xlim([0,550])

    subplot(3,1,2)
    N = 276;
    patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    hold on;
    patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none', 'handlevisibility', 'off')
    patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    plot(F1, 'k', 'linewidth', 2);
    hold on;
    plot(x1_g,'--','color', 'g', 'linewidth', 1);
    plot(x2_g,'--','color', 'b', 'linewidth', 1);
    plot(x_g, 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Gain Specific Model (A = %.2f, B = %.3f)', A, B));
    legend('Input', 'Up State', 'Down State', 'Net Adaptation', 'location', 'southeast');
    xlim([0,550])

    subplot(3,1,3)
    N = 276;
    patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    hold on;
    patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none', 'handlevisibility', 'off')
    patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    plot(F1, 'K', 'linewidth', 2);
    hold on;
    plot(x1_m,'--','color', 'g', 'linewidth', 1);
    plot(x2_m,'--','color', 'b', 'linewidth', 1);
    plot(x_m, 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Multi Rate Model (Af = %.2f, Bf = %.3f, As = %.3f, Bs = %.3f)', Af, Bf, As, Bs));
    legend('Input', 'Fast State', 'Slow State', 'Net Adaptation', 'location', 'southeast');
    xlim([0,550])

    figure_counter = figure_counter + 1;
end

%%
%% SIMULATION MODELS - Learning and Re-Learning Plots with Noise
close all
clc

% Parameters
A_values = [0.95, 0.99, 1.03];  % New values for A
B_values = [0.01, 0.013, 0.016];  % New values for B
Af_values = [0.9, 0.92, 0.94];  % New values for Af
Bf_values = [0.02, 0.03, 0.04];  % New values for Bf
As_values = [0.992, 0.996, 1.0];  % New values for As
Bs_values = [0.002, 0.004, 0.006];  % New values for Bs

% Randomly choose 12 parameter combinations
rng(123);  % Set random seed for reproducibility
indices = randperm(numel(A_values)*numel(B_values)*numel(Af_values)*numel(Bf_values)*numel(As_values)*numel(Bs_values), 12);

figure_counter = 1;

for idx = indices
    [i, j, k, l, m, n] = ind2sub([numel(A_values), numel(B_values), numel(Af_values), numel(Bf_values), numel(As_values), numel(Bs_values)], idx);
    
    A = A_values(i);
    B = B_values(j);
    Af = Af_values(k);
    Bf = Bf_values(l);
    As = As_values(m);
    Bs = Bs_values(n);

    % Trial configuration
    null = 25;
    learn = 250;

    Null = zeros(1, null);
    Learn = ones(1, learn);
    unlearn = -ones(1, 18);

    F1 = [Null Learn unlearn Learn];

    % Single state model with noise
    noise_s = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
    x_s = zeros(size(F1));
    for p = 1:length(F1)-1
        x_s(p+1) = A * x_s(p) + B * (F1(p) - x_s(p)) + noise_s(p);
    end

    unlearn = -ones(1, 10);

    % Gain specific model with noise
    noise_g = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
    x1_g = zeros(size(F1));
    x2_g = zeros(size(F1));
    x_g = zeros(size(F1));
    for p = 1:length(F1)-1
        x1_g(p+1) = min(0, A * x1_g(p) + B * (F1(p) - x_g(p))) + noise_g(p);
        x2_g(p+1) = max(0, A * x2_g(p) + B * (F1(p) - x_g(p))) + noise_g(p);
        x_g(p+1) = x1_g(p+1) + x2_g(p+1);
    end

    unlearn = -ones(1, 29);

    % Multi-rate model with noise
    noise_m = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
    x1_m = zeros(size(F1));
    x2_m = zeros(size(F1));
    x_m = zeros(size(F1));
    for p = 1:length(F1)-1
        x1_m(p+1) = Af * x1_m(p) + Bf * (F1(p) - x_m(p)) + noise_m(p);
        x2_m(p+1) = As * x2_m(p) + Bs * (F1(p) - x_m(p)) + noise_m(p);
        x_m(p+1) = x1_m(p+1) + x2_m(p+1);
    end

    % Plot the state over time for each model
    figure(figure_counter);
    subplot(3,1,1)
    N = 276;
    patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    hold on;
    patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none', 'handlevisibility', 'off')
    patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    plot(F1, 'k', 'linewidth', 2);
    hold on;
    plot(x_s, 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Single State Model (A = %.2f, B = %.3f) with Noise', A, B));
    legend('Input', 'Net Adaptation', 'location', 'southeast');
    xlim([0,550])

    subplot(3,1,2)
    N = 276;
    patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    hold on;
    patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none', 'handlevisibility', 'off')
    patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    plot(F1, 'k', 'linewidth', 2);
    hold on;
    plot(x1_g,'--','color', 'g', 'linewidth', 1);
    plot(x2_g,'--','color', 'b', 'linewidth', 1);
    plot(x_g, 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Gain Specific Model (A = %.2f, B = %.3f) with Noise', A, B));
    legend('Input', 'Up State', 'Down State', 'Net Adaptation', 'location', 'southeast');
    xlim([0,550])

    subplot(3,1,3)
    N = 276;
    patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    hold on;
    patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none', 'handlevisibility', 'off')
    patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none', 'handlevisibility', 'off')
    plot(F1, 'K', 'linewidth', 2);
    hold on;
    plot(x1_m,'--','color', 'g', 'linewidth', 1);
    plot(x2_m,'--','color', 'b', 'linewidth', 1);
    plot(x_m, 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Multi Rate Model (Af = %.2f, Bf = %.3f, As = %.3f, Bs = %.3f) with Noise', Af, Bf, As, Bs));
    legend('Input', 'Fast State', 'Slow State', 'Net Adaptation', 'location', 'southeast');
    xlim([0,550])

    figure_counter = figure_counter + 1;
end



%% SIMULATION MODELS- Sigmoidal decay:
% close all
% clc
% 
% % Original Parameters
% A_orig = 0.99; B_orig = 0.013; Af_orig = 0.92; Bf_orig = 0.03; As_orig = 0.996; Bs_orig = 0.004;
% 
% % Trial configuration
% null = 25;
% learn = 250;
% 
% Null = zeros(1, null);
% Learn = ones(1, learn);
% unlearn = -ones(1, 18);
% 
% F1 = [Null Learn unlearn Learn];
% 
% % Decay parameters
% k = 0; % decay rate for sigmoidal decay
% midpoint = 0; % trial at which decay is fastest
% 
% % Calculate sigmoidally decaying parameters
% A = A_orig ./ (1 + exp(k*((1:length(F1)) - midpoint)));
% B = B_orig ./ (1 + exp(k*((1:length(F1)) - midpoint)));
% Af = Af_orig ./ (1 + exp(k*((1:length(F1)) - midpoint)));
% Bf = Bf_orig ./ (1 + exp(k*((1:length(F1)) - midpoint)));
% As = As_orig ./ (1 + exp(k*((1:length(F1)) - midpoint)));
% Bs = Bs_orig ./ (1 + exp(k*((1:length(F1)) - midpoint)));
% 
% % Single state model
% x_s = zeros(size(F1));
% for n = 1:length(F1)-1
%     x_s(n+1) = A(n) * x_s(n) + B(n) * (F1(n) - x_s(n));
% end
% 
% unlearn = -ones(1, 10);
% 
% % Gain specific model
% x1_g = zeros(size(F1));
% x2_g = zeros(size(F1));
% x_g = zeros(size(F1));
% for n = 1:length(F1)-1
%     x1_g(n+1) = min(0, A(n) * x1_g(n) + B(n) * (F1(n) - x_g(n)));
%     x2_g(n+1) = max(0, A(n) * x2_g(n) + B(n) * (F1(n) - x_g(n)));
%     x_g(n+1) = x1_g(n+1) + x2_g(n+1);
% end
% 
% unlearn = -ones(1, 23);
% 
% % Multi-rate model
% x1_m = zeros(size(F1));
% x2_m = zeros(size(F1));
% x_m = zeros(size(F1));
% for n = 1:length(F1)-1
%     x1_m(n+1) = Af(n) * x1_m(n) + Bf(n) * (F1(n) - x_m(n));
%     x2_m(n+1) = As(n) * x2_m(n) + Bs(n) * (F1(n) - x_m(n));
%     x_m(n+1) = x1_m(n+1) + x2_m(n+1);
% end
% 
% 
% 


%% SIMULATION MODELS_ LINEAR decay:
close all
clc

% Original Parameters
A_orig = 0.99; B_orig = 0.013; Af_orig = 0.92; Bf_orig = 0.03; As_orig = 0.996; Bs_orig = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 18);

F1 = [Null Learn unlearn Learn];

% Decay parameters
k1 = 0.0001; % decay rate for linear decay
k2 = 0.000001; % decay rate for linear decay
k3 = 0.000001; % decay rate for linear decay


% Calculate linearly decaying parameters
A = max(0, A_orig + k1*(1:length(F1)));
B = max(0, B_orig + k2*(1:length(F1)));
Af = max(0, Af_orig + k1*(1:length(F1)));
Bf = max(0, Bf_orig + k1*(1:length(F1)));
As = max(0, As_orig + k3*(1:length(F1)));
Bs = max(0, Bs_orig + k2*(1:length(F1)));

% Single state model
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A(n) * x_s(n) + B(n) * (F1(n) - x_s(n));
end

unlearn = -ones(1, 10);

% Gain specific model
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
x_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A(n) * x1_g(n) + B(n) * (F1(n) - x_g(n)));
    x2_g(n+1) = max(0, A(n) * x2_g(n) + B(n) * (F1(n) - x_g(n)));
    x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end

unlearn = -ones(1, 23);

% Multi-rate model
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
    x1_m(n+1) = Af(n) * x1_m(n) + Bf(n) * (F1(n) - x_m(n));
    x2_m(n+1) = As(n) * x2_m(n) + Bs(n) * (F1(n) - x_m(n));
    x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end




% Plot the state over time for each model
figure;
subplot(3,1,1)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x_s, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Single State Model');
legend('Input', 'Net Adaptation', 'location', 'southeast');
xlim([0,550])

unlearn = -ones(1, 10);
subplot(3,1,2)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x1_g,'--','color', 'g', 'linewidth', 1);
plot(x2_g,'--','color', 'b', 'linewidth', 1);
plot(x_g, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Gain Specific Model');
legend('Input', 'Up State', 'Down State', 'Net Adaptation', 'location', 'southeast');
xlim([0,550])

unlearn = -ones(1, 20);
subplot(3,1,3)
N = 276;
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+19 N+19 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+19 2*N 2*N N+19], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'K', 'linewidth', 2);
hold on;
plot(x1_m,'--','color', 'g', 'linewidth', 1);
plot(x2_m,'--','color', 'b', 'linewidth', 1);
plot(x_m, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model');
legend('Input', 'Fast State', 'Slow State', 'Net Adaptation',  'location', 'southeast');
figure;
% single model

plot(x_s(26:275), 'b', 'linewidth', 1);
hold on;
plot(x_s(304:535),'--','color', 'r', 'linewidth',2);
xlabel('Time');
ylabel('State');

legend('Learning Phase', 'Re-learning Phase');
sgtitle('Learning and Re-learning Phases for single State Model ');
common_length = min(length(x_s(26:275)), length(x_s(304:535)));
area_between_curves = trapz( -x_s( 304:304+common_length-1) + x_s(26:26+common_length-1));
% Display the calculated area
disp(['Area between the curves: ' num2str(area_between_curves)]);

figure;

plot(x_g(26:275), 'b', 'linewidth', 1);
hold on;
plot(x_g(297:535),'--','color', 'r', 'linewidth', 2);
xlabel('Time');
ylabel('State');
title('Gain Specific Model - Learning Phase');
legend('Learning Phase', 'Re-learning Phase');
ylim([0,0.6])

xlim([0,550])
%% ERROR CLAMP
%% ERROR CLAPM
% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because paper did it and if we se the
% large value for it we goes to negetive value!!!
unlearn = -ones(1, 28);
rest = zeros(1, 250);
F1 = [Null Learn unlearn rest ];
N = 276;
h1 = plot(F1, 'k', 'linewidth', 2);
hold on;

h2 = patch([0 25 25 0], [-1 -1 1 1], [0 0 1], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off');
h3 = patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off');
h4 = patch([N+28 2*N 2*N N+28], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off');
h5 = patch([N N+28 N+28 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off');
h6 = patch([N+28 2*N 2*N N+28], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off');
xlim([0, 550]);
legend([h1 h2 h3 h4 h5 h6], 'Input', 'Null trials', 'Adaptation Trials', 'Error Clapm Trials', 'Deadaptation Trials');
sgtitle('Error-clamp / Relearning Expriment');
%% ERROR CLAMP - SINGLE MODEL
% %%
close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because paper did it and if we se the
% large value for it we goes to negetive value!!!
unlearn = -ones(1, 28);
rest = zeros(1, 250);
F1 = [Null Learn unlearn rest];

% Single state model
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n));
end


% Plot the state over time for each model
figure;
subplot(3,1,1)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N+28 2*N 2*N N+28], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N N+28 N+28 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')


plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x_s, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Single State Model');
legend('Input', 'Net Adaptation', 'location', 'southeast');
xlim([0,550])

%% ERROR CLAMP GAIN SPECIFIC AND MULTI-RATE

close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because paper did it and if we se the
% large value for it we goes to negetive value!!!
unlearn = -ones(1, 15);
rest = zeros(1, 350);
F1 = [Null Learn unlearn rest];


% Gain specific model
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
x_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n)));
    x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n)));
    x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end

% Multi-rate model
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
    x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end
unlearn = -ones(1, 10);
subplot(3,1,2)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N+15 2*N 2*N N+15], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N N+15 N+15 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N+15 2*N 2*N N+15], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x1_g,'--','color', 'g', 'linewidth', 1);
plot(x2_g,'--','color', 'b', 'linewidth', 1);
plot(x_g, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Gain Specific Model');
legend('Input', 'Up State', 'Down State', 'Net Adaptation', 'location', 'southeast');
xlim([0,550])

unlearn = -ones(1, 20);
subplot(3,1,3)
N = 276;
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N+15 2*N 2*N N+15], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N N+15 N+15 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N+15 2*N 2*N N+15], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'K', 'linewidth', 2);
hold on;
plot(x1_m,'--','color', 'g', 'linewidth', 1);
plot(x2_m,'--','color', 'b', 'linewidth', 1);
plot(x_m, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model');
legend('Input', 'Fast State', 'Slow State', 'Net Adaptation',  'location', 'southeast');
xlim([0,550])




%% ERROR CLAMP / RELEARNING
% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because paper did it and if we se the
% large value for it we goes to negetive value!!!
unlearn = -ones(1, 28);
rest = zeros(1, 100);
F1 = [Null Learn unlearn rest Learn];
N = 276;
h1 = plot(F1, 'k', 'linewidth', 2);
hold on;

h2 = patch([0 25 25 0], [-1 -1 1 1], [0 0 1], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off');
h3 = patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off');
h4 = patch([N+28 N+128 N+128 N+28], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off');
h5 = patch([N N+28 N+28 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off');
h6 = patch([N+28 2*N 2*N N+28], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off');

legend([h1 h2 h3 h4 h5 h6], 'Input', 'Null trials', 'Adaptation Trials', 'Error Clapm Trials', 'Deadaptation Trials', 'Readaptation Trials');
sgtitle('Error-clamp / Relearning Expriment');
xlim([0, 550]);

%% ERROR CLAMP / RELEARNING SINGLE MODEL
% %%
close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because paper did it and if we se the
% large value for it we goes to negetive value!!!
unlearn = -ones(1, 28);
rest = zeros(1, 100);
F1 = [Null Learn unlearn rest Learn];

% Single state model
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n));
end


% Plot the state over time for each model
figure;
subplot(3,1,1)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N+28 N+128 N+128 N+28], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N N+28 N+28 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N+28 2*N 2*N N+28], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x_s, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Single State Model');
legend('Input', 'Net Adaptation', 'location', 'southeast');
xlim([0,550])

%% ERROR CLAMP GAIN SPECIFIC AND MULTI-RATE

close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because paper did it and if we se the
% large value for it we goes to negetive value!!!
unlearn = -ones(1, 15);
rest = zeros(1, 100);
F1 = [Null Learn unlearn rest Learn];


% Gain specific model
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
x_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n)));
    x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n)));
    x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end

% Multi-rate model
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
    x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end
unlearn = -ones(1, 10);
subplot(3,1,2)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N+15 N+115 N+115 N+15], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N N+15 N+15 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N+15 2*N 2*N N+15], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x1_g,'--','color', 'g', 'linewidth', 1);
plot(x2_g,'--','color', 'b', 'linewidth', 1);
plot(x_g, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Gain Specific Model');
legend('Input', 'Up State', 'Down State', 'Net Adaptation', 'location', 'southeast');
xlim([0,550])

unlearn = -ones(1, 20);
subplot(3,1,3)
N = 276;
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N+15 N+115 N+115 N+15], [-1 -1 1 1], [0 1 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N N+15 N+15 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')

patch([N+15 2*N 2*N N+15], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'K', 'linewidth', 2);
hold on;
plot(x1_m,'--','color', 'g', 'linewidth', 1);
plot(x2_m,'--','color', 'b', 'linewidth', 1);
plot(x_m, 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model');
legend('Input', 'Fast State', 'Slow State', 'Net Adaptation',  'location', 'southeast');
xlim([0,550])


%%
% Calculate performance on each trial of initial learning and relearning for each model
initial_learning_performance_s = abs(F1(null+1:null+learn) - x_s(null+1:null+learn));
relearning_performance_s = abs(F1(2*learn+null+length(unlearn)+1:end) - x_s(2*learn+null+length(unlearn)+1:end));

initial_learning_performance_g = abs(F1(null+1:null+learn) - x_g(null+1:null+learn));
relearning_performance_g = abs(F1(2*learn+null+length(unlearn)+1:end) - x_g(2*learn+null+length(unlearn)+1:end));

initial_learning_performance_m = abs(F1(null+1:null+learn) - x_m(null+1:null+learn));
relearning_performance_m = abs(F1(2*learn+null+length(unlearn)+1:end) - x_m(2*learn+null+length(unlearn)+1:end));

% Calculate savings as percent improvement
savings_s = (initial_learning_performance_s - relearning_performance_s) ./ initial_learning_performance_s * 100;
savings_g = (initial_learning_performance_g - relearning_performance_g) ./ initial_learning_performance_g * 100;
savings_m = (initial_learning_performance_m - relearning_performance_m) ./ initial_learning_performance_m * 100;

% Plot savings over trials
figure;
plot(savings_s, 'r', 'linewidth', 1);
hold on;
plot(savings_g, 'g', 'linewidth', 1);
plot(savings_m, 'b', 'linewidth', 1);
xlabel('Trial');
ylabel('Savings (%)');
title('Savings over Trials');
legend('Single State Model', 'Gain Specific Model', 'Multi Rate Model', 'location', 'northeast');

%% single state _ Learning and Re learning 
close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 29);

F1 = [Null Learn unlearn Learn];

% Single state model
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n));
end

% Plot the state over time for each model (learning and re-learning phases)
figure;

% Single state model
subplot(2, 1, 1)
plot(x_s(26:275), 'b', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Single State Model - Learning Phase');


subplot(2, 1, 2)
plot(x_s(304:535), 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Single State Model - Re-learning Phase');

figure;
% single model

plot(x_s(26:275), 'b', 'linewidth', 1);
hold on;
plot(x_s(304:535),'--','color', 'r', 'linewidth',2);
xlabel('Time');
ylabel('State');

legend('Learning Phase', 'Re-learning Phase');
sgtitle('Learning and Re-learning Phases for single State Model ');
common_length = min(length(x_s(26:275)), length(x_s(304:535)));
area_between_curves = trapz( -x_s( 304:304+common_length-1) + x_s(26:26+common_length-1));
% Display the calculated area
disp(['Area between the curves: ' num2str(area_between_curves)]);

%% CONTOUR FOR  single model
%very sesible to changing A, B parameter!!
close all
clc

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 29);

F1 = [Null Learn unlearn Learn];

% Define the range of values for A and B
A_values = linspace(0.9, 1.0, 20);
B_values = linspace(0.01, 0.02, 20);

% Preallocate the matrix to store the areas
areas = zeros(length(A_values), length(B_values));

% Loop over different parameter values
for i = 1:length(A_values)
    for j = 1:length(B_values)
        A = A_values(i);
        B = B_values(j);
        
         % Single state model
            x_s = zeros(size(F1));
                for n = 1:length(F1)-1
                  x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n));
                end
        
        % Calculate the area between the curves
        common_length = min(length(x_s(26:275)), length(x_s(305:554)));
        area_between_curves = trapz(x_s(305:305+common_length-1) - x_s(26:26+common_length-1));
        
        % Store the area in the matrix
        areas(i, j) = area_between_curves;
    end
end

% Plot the area between curves as an image
figure;
imagesc( A_values,B_values, areas);
colorbar;
xlabel('B');
ylabel('A');
title('Area between Curves');


%% Gain specific _ 

close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 19);

F1 = [Null Learn unlearn Learn];
% Gain specific model
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
x_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n)));
    x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n)));
    x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end





% Gain specific model
figure;

plot(x_g(26:275), 'b', 'linewidth', 1);
hold on;
plot(x_g(297:535),'--','color', 'r', 'linewidth', 2);
xlabel('Time');
ylabel('State');
title('Gain Specific Model - Learning Phase');
legend('Learning Phase', 'Re-learning Phase');
ylim([0,0.6])


% Gain specific model
figure;
subplot(2, 1, 1)
plot(x_g(26:275), 'b', 'linewidth', 1);
xlabel('Time');
ylabel('State');
ylim([0,0.6])

subplot(2, 1, 2)
plot(x_g(297:535),'--','color', 'r', 'linewidth', 2);
xlabel('Time');
ylabel('State');
sgtitle('Learning and Re-learning Phases for Gain specific model');
ylim([0,0.6])
common_length = min(length(x_g(26:275)), length(x_g(297:535)));
area_between_curves = trapz( x_g(297:297+common_length-1) - x_g(26:26+common_length-1));
% Display the calculated area
disp(['Area between the curves: ' num2str(area_between_curves)]);
%% CONTOUR FOR GAIN SPECIFIC
%very sesible to changing A, B parameter!!
close all
clc

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 19);

F1 = [Null Learn unlearn Learn];

% Define the range of values for A and B
A_values = linspace(0.9, 1.0, 20);
B_values = linspace(0.01, 0.02, 20);

% Preallocate the matrix to store the areas
areas = zeros(length(A_values), length(B_values));

% Loop over different parameter values
for i = 1:length(A_values)
    for j = 1:length(B_values)
        A = A_values(i);
        B = B_values(j);
        
        % Gain specific model
        x1_g = zeros(size(F1));
        x2_g = zeros(size(F1));
        x_g = zeros(size(F1));
        for n = 1:length(F1)-1
            x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n)));
            x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n)));
             x_g(n+1) = x1_g(n+1) + x2_g(n+1);
        end
       
        
        % Calculate the area between the curves
        common_length = min(length(x_g(26:275)), length(x_g(297:535)));
        area_between_curves = trapz(x_g(297:297+common_length-1) - x_g(26:26+common_length-1));
        
        % Store the area in the matrix
        areas(i, j) = area_between_curves;
    end
end

% Plot the area between curves as an image
figure;
imagesc( A_values,B_values, areas);
colorbar;
xlabel('B');
ylabel('A');
title('Area between Curves');


%%  Multi-rate model AREA

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 20);


% Multi-rate model
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
    x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end

% Multi-rate model
figure;
subplot(2, 1, 1)
plot(x_m(26:275), 'b', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model - Learning Phase');

subplot(2, 1, 2)
plot(x_m(300:535),'--','color', 'r', 'linewidth', 2);
xlabel('Time');
ylabel('State');
title('Multi Rate Model - Re-learning Phase');

% Calculate the area between the curves
common_length = min(length(x_m(26:275)), length(x_m(310:544)));
area_between_curves = trapz( x_m(310:310 +common_length-1) - x_m(26:26+common_length-1));

% Multi-rate model
figure;
plot(x_m(26:275), 'b', 'linewidth', 1);
hold on;
plot(x_m(310:544),'--','color', 'r', 'linewidth', 2);
xlabel('Time');
ylabel('State');

legend('Learning Phase', 'Re-learning Phase');
sgtitle('Learning and Re-learning Phases for Multi-rate model');

% Display the calculated area
disp(['Area between the curves: ' num2str(area_between_curves)]);

%% AREA CONTOUR PLOT Af Bf
% Parameters
A = 0.99; B = 0.013; Af_values = [0.006:0.002:1.00]; Bf_values = [0.01:0.0005:0.07];
 As = 0.996; Bs = 0.004;
% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 20);

% Preallocate the matrix to store the calculated areas
areas = zeros(length(Af_values), length(Bf_values));

% Loop over different Af and Bf values
for i = 1:length(Af_values)
    for j = 1:length(Bf_values)
        Af = Af_values(i);
        Bf = Bf_values(j);
        
        % Multi-rate model
        x1_m = zeros(size(F1));
        x2_m = zeros(size(F1));
        x_m = zeros(size(F1));
        for n = 1:length(F1)-1
            x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
            x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
            x_m(n+1) = x1_m(n+1) + x2_m(n+1);
        end
        
        
        % Calculate the area between the curves
        common_length = min(length(x_m(26:275)), length(x_m(310:544)));
        area_between_curves = trapz(x_m(310:310+common_length-1) - x_m(26:26+common_length-1));
        
        % Store the calculated area
        areas(i, j) = area_between_curves;
    end
end

% Plot the areas using imagesc
figure;
imagesc(Af_values,Bf_values,  areas);
xlabel('A_f');
ylabel('Bـf');
title('Area between Curves for different Af and Bf');
colorbar;

% Display the calculated areas
disp('Areas between the curves for different Af and Bf:');
disp(areas);

%%

%% AREA CONTOUR PLOT As Bs
% Parameters

A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As_values = [0.998:0.0002:1.002]; Bs_values = [0.001:0.0005:0.009];
% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 20);

% Preallocate the matrix to store the calculated areas
areas = zeros(length(As_values), length(Bs_values));

% Loop over different Af and Bf values
for i = 1:length(As_values)
    for j = 1:length(Bs_values)
        As = As_values(i);
        Bs = Bs_values(j);
        
        % Multi-rate model
        x1_m = zeros(size(F1));
        x2_m = zeros(size(F1));
        x_m = zeros(size(F1));
        for n = 1:length(F1)-1
            x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
            x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
            x_m(n+1) = x1_m(n+1) + x2_m(n+1);
        end
        
        % Calculate the area between the curves
        common_length = min(length(x_m(26:275)), length(x_m(310:544)));
        area_between_curves = trapz(x_m(310:310+common_length-1) - x_m(26:26+common_length-1));
        
        % Store the calculated area
        areas(i, j) = area_between_curves;
    end
end

% Plot the areas using imagesc
figure;
imagesc(Bs_values, As_values, areas);
xlabel('B_s');
ylabel('A_s');
title('Area between Curves for different As and Bs');
colorbar;

% Display the calculated areas
disp('Areas between the curves for different As and Bs:');
disp(areas);
%% CHANGE Af, Bf, As, Bs AND PLOT CONTOUR  %OPTIONAL!!!!!!!!!!
% Parameters
A = 0.99; B = 0.013;
Af_values = [0.1, 0.94, 1.5];
Bf_values = [0.034, 0.035, 0.05];
As_values = [0.996, 0.998, 1.0];
Bs_values = [0.004, 0.005, 0.006];

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 20);

% Preallocate the matrix to store the calculated areas
areas = zeros(length(Af_values), length(Bf_values), length(As_values), length(Bs_values));

% Loop over different parameter values
for i = 1:length(Af_values)
    for j = 1:length(Bf_values)
        for k = 1:length(As_values)
            for l = 1:length(Bs_values)
                Af = Af_values(i);
                Bf = Bf_values(j);
                As = As_values(k);
                Bs = Bs_values(l);
                
        % Multi-rate model
        x1_m = zeros(size(F1));
        x2_m = zeros(size(F1));
        x_m = zeros(size(F1));
        for n = 1:length(F1)-1
            x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
            x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
            x_m(n+1) = x1_m(n+1) + x2_m(n+1);
        end
                
                % Calculate the area between the curves
                common_length = min(length(x_m(26:275)), length(x_m(300:535)));
                area_between_curves = trapz(x_m(300:300+common_length-1) - x_m(26:26+common_length-1));
                
                % Store the calculated area
                areas(i, j, k, l) = area_between_curves;
            end
        end
    end
end

% Create parameter grids
[Af_grid, Bf_grid] = meshgrid(linspace(min(Af_values), max(Af_values), 20), linspace(min(Bf_values), max(Bf_values), 20));

% Interpolate the areas to match the 20x20 grid
slice_areas_interp = interp2(Af_values, Bf_values, slice_areas, Af_grid, Bf_grid, 'linear');

% Create imagesc plot
figure;
imagesc(Af_grid(:), Bf_grid(:), slice_areas_interp);
xlabel('Af');
ylabel('Bf');
title('Areas between Curves (As = 0.998, Bs = 0.005)');
colorbar;


%%


% Parameters
A = 0.99; B = 0.013;
Af_values = linspace(0.9, 1, 20);
Bf_values = linspace(0.02, 0.04, 20);
As_values = linspace(0.99, 1.01, 20);
Bs_values = linspace(0.002, 0.006, 20);

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 20);

% Preallocate the matrix to store the areas
areas = zeros(length(As_values), length(Bs_values), length(Af_values), length(Bf_values));

% Loop over different parameter values
for i = 1:length(As_values)
    for j = 1:length(Bs_values)
        for k = 1:length(Af_values)
            for l = 1:length(Bf_values)
                As = As_values(i);
                Bs = Bs_values(j);
                Af = Af_values(k);
                Bf = Bf_values(l);
                
        % Multi-rate model
        x1_m = zeros(size(F1));
        x2_m = zeros(size(F1));
        x_m = zeros(size(F1));
        for n = 1:length(F1)-1
            x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n));
            x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n));
            x_m(n+1) = x1_m(n+1) + x2_m(n+1);
        end
                
                % Calculate the area between the curves
                common_length = min(length(x_m(26:275)), length(x_m(300:535)));
                area_between_curves = trapz(x_m(300:300+common_length-1) - x_m(26:26+common_length-1));
                
                % Store the area in the matrix
                areas(i, j, k, l) = area_between_curves;
            end
        end
    end
end

% Create a grid of parameter values
[Af_grid, Bs_grid, As_grid, Bf_grid] = ndgrid(Af_values, Bs_values, As_values, Bf_values);

% Reshape the parameter grids and areas matrix into vectors
Af_vec = reshape(Af_grid, [], 1);
Bs_vec = reshape(Bs_grid, [], 1);
As_vec = reshape(As_grid, [], 1);
Bf_vec = reshape(Bf_grid, [], 1);
areas_vec = reshape(areas, [], 1);

% Plot the points in 3D coordinate system
figure;
scatter3(As_vec, Bs_vec, Af_vec, 20*areas_vec, Bf_vec, 'filled');
xlabel('As');
ylabel('Bs');
zlabel('Af');
title('3D Coordinate System');
colorbar;
%%
% Parameters
A = 0.99; B = 0.013;
Af_values = linspace(0.9, 1, 20);
Bf_values = linspace(0.02, 0.04, 20);
As_values = linspace(0.99, 1.01, 20);
Bs_values = linspace(0.002, 0.006, 20);

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 20);

% Preallocate the matrix to store the areas
areas = zeros(length(As_values), length(Bs_values), length(Af_values), length(Bf_values));

% Loop over different parameter values
for i = 1:length(As_values)
    for j = 1:length(Bs_values)
        for k = 1:length(Af_values)
            for l = 1:length(Bf_values)
                As = As_values(i);
                Bs = Bs_values(j);
                Af = Af_values(k);
                Bf = Bf_values(l);
                
                % Multi-rate model
                x1_m = zeros(size(F1));
                x2_m = zeros(size(F1));
                for n = 1:length(F1)-1
                    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x1_m(n));
                    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x2_m(n));
                end
                x_m = x1_m + x2_m;
                
                % Calculate the area between the curves
                common_length = min(length(x_m(26:275)), length(x_m(300:535)));
                area_between_curves = trapz(x_m(300:300+common_length-1) - x_m(26:26+common_length-1));
                
                % Store the area in the matrix
                areas(i, j, k, l) = area_between_curves;
            end
        end
    end
end

% Create a grid of parameter values
[Af_grid, Bs_grid, As_grid] = meshgrid(Af_values, Bs_values, As_values);

% Reshape the parameter grids and areas matrix into vectors
Af_vec = reshape(Af_grid, [], 1);
Bs_vec = reshape(Bs_grid, [], 1);
As_vec = reshape(As_grid, [], 1);
areas_vec = reshape(areas, [], 1);

% Normalize the areas for color mapping
areas_norm = (areas_vec - min(areas_vec)) / (max(areas_vec) - min(areas_vec));

% Plot the points in 3D coordinate system
figure;
scatter3(As_vec, Bs_vec, Af_vec, 20*Bf_vec, areas_norm, 'filled');
xlabel('As');
ylabel('Bs');
zlabel('Af');
title('3D Coordinate System');
colormap('jet');
colorbar;










