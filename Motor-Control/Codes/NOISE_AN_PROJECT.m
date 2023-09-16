%% GAUSSIAN NOISE
close all
clc

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because the paper did it, and if we set
% a large value for it, we go to negative values!!!
unlearn = -ones(1, 29);

F1 = [Null Learn unlearn Learn];

% Single state model with noise
noise_s = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n)) + noise_s(n);
end

unlearn = -ones(1, 10);

% Gain specific model with noise
noise_g = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n))) + noise_g(n);
    x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n))) + noise_g(n);
    x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end


unlearn = -ones(1, 23);

% Multi-rate model with noise
noise_m = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
for n = 1:length(F1)-1
    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n)) + noise_m(n);
    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n)) + noise_m(n);
    x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end


unlearn = -ones(1, 29);

% Plot the state over time for each model with noise
figure;
subplot(3,1,1)

N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+29 N+29 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+29 2*N 2*N N+29], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x_s,'--','color', 'r', 'linewidth', 1.5);
xlim([0 550])
xlabel('Time');
ylabel('State');
title('Single State Model with Noise');
legend('Input', 'Net Adaptation');

unlearn = -ones(1, 10);
subplot(3,1,2)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+29 N+29 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+29 2*N 2*N N+29], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x1_g, 'g', 'linewidth', 1);
plot(x2_g, 'b', 'linewidth', 1);
plot(x_g,'--','color', 'r', 'linewidth', 1.5);
xlim([0 550])
xlabel('Time');
ylabel('State');
title('Gain Specific Model with Noise');
legend('Input', 'Down State', 'up State', 'Net Adaptation');

unlearn = -ones(1, 20);
subplot(3,1,3)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+29 N+29 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+29 2*N 2*N N+29], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x1_m, 'g', 'linewidth', 1);
plot(x2_m, 'b', 'linewidth', 1);
plot(x_m,'--','color', 'r', 'linewidth', 1.5);
xlim([0 550])
xlabel('Time');
ylabel('State');
title('Multi Rate Model with Noise');
legend('Input', 'Fast State', 'Slow State', 'Net Adaptation');

%% Dynamic
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

    % Plot the Learning and Re-Learning curves with noise for each model
    figure(figure_counter);
    subplot(3,1,1)
    plot(x_s(26:275), 'b', 'linewidth', 1);
    hold on;
    plot(x_s(300:535),'--','color', 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Single State Model (A = %.2f, B = %.3f) with Noise', A, B));
    legend('Learning', 'Re-Learning', 'location', 'southeast');

    subplot(3,1,2)
    plot(x_g(26:275), 'b', 'linewidth', 1);
    hold on;
    plot(x_g(300:535),'--','color', 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Gain Specific Model (A = %.2f, B = %.3f) with Noise', A, B));
    legend('Learning', 'Re-Learning', 'location', 'southeast');

    subplot(3,1,3)
    plot(x_m(26:275), 'b', 'linewidth', 1);
    hold on;
    plot(x_m(300:535),'--','color', 'r', 'linewidth', 1);
    xlabel('Time');
    ylabel('State');
    title(sprintf('Multi Rate Model (Af = %.2f, Bf = %.3f, As = %.3f, Bs = %.3f) with Noise', Af, Bf, As, Bs));
    legend('Learning', 'Re-Learning', 'location', 'southeast');

    figure_counter = figure_counter + 1;
end

%% PINK NOISE

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
% I set the Unlearn trials very small because the paper did it, and if we set
% a large value for it, we go to negative values!!!
unlearn = -ones(1, 29);

F1 = [Null Learn unlearn Learn];

% Generate the pink noise
num_points = length(F1);
pink_noise_s = 0.001 * pinknoise(num_points);
pink_noise_g = 0.001 * pinknoise(num_points);
pink_noise_m = 0.001 * pinknoise(num_points);

% Single state model with pink noise
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n)) + pink_noise_s(n);
end

% Gain specific model with pink noise
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
for n = 1:length(F1)-1
    x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n))) + pink_noise_g(n);
    x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n))) + pink_noise_g(n);
    x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end

% Multi-rate model with pink noise
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
for n = 1:length(F1)-1
    x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n)) + pink_noise_m(n);
    x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n)) + pink_noise_m(n);
    x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end

% Plot the state over time for each model with noise
figure;
subplot(3,1,1)

N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+29 N+29 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+29 2*N 2*N N+29], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x_s,'--','color', 'r', 'linewidth', 1.5);
xlim([0 550])
xlabel('Time');
ylabel('State');
title('Single State Model with Noise');
legend('Input', 'Net Adaptation');

unlearn = -ones(1, 10);
subplot(3,1,2)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+29 N+29 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+29 2*N 2*N N+29], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x1_g, 'g', 'linewidth', 1);
plot(x2_g, 'b', 'linewidth', 1);
plot(x_g,'--','color', 'r', 'linewidth', 1.5);
xlim([0 550])
xlabel('Time');
ylabel('State');
title('Gain Specific Model with Noise');
legend('Input', 'Down State', 'up State', 'Net Adaptation');

unlearn = -ones(1, 20);
subplot(3,1,3)
N = 276;
patch([0 N N 0], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor',...
    'none', 'handlevisibility', 'off')
hold on;
patch([N N+29 N+29 N], [-1 -1 1 1], [1 0 0], 'FaceAlpha', 0.2, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
patch([N+29 2*N 2*N N+29], [-1 -1 1 1], [0 0 0], 'FaceAlpha', 0.1, 'edgecolor', 'none'...
    , 'handlevisibility', 'off')
plot(F1, 'k', 'linewidth', 2);
hold on;
plot(x1_m, 'g', 'linewidth', 1);
plot(x2_m, 'b', 'linewidth', 1);
plot(x_m,'--','color', 'r', 'linewidth', 1.5);
xlim([0 550])
xlabel('Time');
ylabel('State');
title('Multi Rate Model with Noise');
legend('Input', 'Fast State', 'Slow State', 'Net Adaptation');

%%

%Single state model with noise
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

% Single state model with noise
noise_s = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.01
x_s = zeros(size(F1));
for n = 1:length(F1)-1
    x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n)) + noise_s(n);
end

% Plot the state over time for each model (learning and re-learning phases)
figure;

% Single state model
subplot(2, 1, 1)
plot(x_s(26:275), 'b', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Single State Model - Learning Phase');
grid on;


subplot(2, 1, 2)
plot(x_s(304:535),'--','color', 'r', 'linewidth', 1.5);
xlabel('Time');
ylabel('State');
title('Single State Model - Re-learning Phase');
grid on;
figure;
% Single state model
plot(x_s(26:275), 'b', 'linewidth', 1);
hold on;
plot(x_s(304:535),'--','color', 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
legend('Learning Phase', 'Re-learning Phase');
grid on;
sgtitle('Learning and Re-learning Phases for Single State Model');

% Calculate the area between the curves
common_length = min(length(x_s(26:275)), length(x_s(304:535)));
area_between_curves = trapz(-x_s(304:304+common_length-1) + x_s(26:26+common_length-1));

% Display the calculated area
disp(['Area between the curves: ' num2str(area_between_curves)]);

%%

%% Single State Model

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 29);

F1 = [Null Learn unlearn Learn];

% Define the range of values for A and B
A_values = linspace(0.9, 0.99, 20);
B_values = linspace(0.01, 0.14, 20);

% Preallocate the matrix to store the areas
areas = zeros(length(A_values), length(B_values));

% Loop over different parameter values
for i = 1:length(A_values)
    for j = 1:length(B_values)
        A = A_values(i);
        B = B_values(j);
        
        % Single state model with noise
        noise_s = 0.1 * rand(size(F1)); % Generate noise between 0 and 0.1
        x_s = zeros(size(F1));
        for n = 1:length(F1)-1
            x_s(n+1) = A * x_s(n) + B * (F1(n) - x_s(n)) + noise_s(n);
        end
        
        % Calculate the area between the curves
        common_length = min(length(x_s(26:275)), length(x_s(305:554)));
        area_between_curves = trapz(x_s(305:305+common_length-1) - x_s(26:26+common_length-1));
        
        % Store the area in the matrix
        areas(i, j) = area_between_curves;
    end
end

% Plot the area between curves as a contour plot
figure;
imagesc(A_values, B_values, areas);
colorbar;
xlabel('B');
ylabel('A');
title('Area between Curves - Single State Model');
%% Gain Specific Model
%% Gain Specific Model with Noise

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 19);

F1 = [Null Learn unlearn Learn];

% Gain specific model with noise
noise_g = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
x1_g = zeros(size(F1));
x2_g = zeros(size(F1));
x_g = zeros(size(F1));
for n = 1:length(F1)-1
x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n))) + noise_g(n);
x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n))) + noise_g(n);
x_g(n+1) = x1_g(n+1) + x2_g(n+1);
end


% Gain specific model with noise
figure;

plot(x_g(26:275), 'b', 'linewidth', 1);
hold on;
plot(x_g(296:535), '--', 'color', 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Gain Specific Model with Noise - Learning Phase');
legend('Learning Phase', 'Re-learning Phase');
grid on;
% Gain specific model with noise
figure;
subplot(2, 1, 1)
plot(x_g(26:275), 'b', 'linewidth', 1);
xlabel('Time');
ylabel('State');
grid on;
subplot(2, 1, 2)
plot(x_g(297:535),'--','color', 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
sgtitle('Learning and Re-learning Phases for Gain Specific Model with Noise');
grid on;
common_length = min(length(x_g(26:275)), length(x_g(297:535)));
area_between_curves = trapz(x_g(297:297+common_length-1) - x_g(26:26+common_length-1));

% Display the calculated area
disp(['Area between the curves: ' num2str(area_between_curves)]);

%% Gain Specific Model

% Parameters
A = 0.99; B = 0.013;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 19);

F1 = [Null Learn unlearn Learn];

% Define the range of values for A and B
A_values = linspace(0.98, 1.001, 20);
B_values = linspace(0.01, 0.005, 20);

% Preallocate the matrix to store the areas
areas = zeros(length(A_values), length(B_values));

% Loop over different parameter values
for i = 1:length(A_values)
    for j = 1:length(B_values)
        A = A_values(i);
        B = B_values(j);
        
        % Gain specific model with noise
        noise_g = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
        x1_g = zeros(size(F1));
        x2_g = zeros(size(F1));
        x_g = zeros(size(F1));
        for n = 1:length(F1)-1
            x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n)) + noise_g(n));
            x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n)) + noise_g(n));
            x_g(n+1) = x1_g(n+1) + x2_g(n+1);
        end
        
        
        % Calculate the area between the curves
        common_length = min(length(x_g(26:275)), length(x_g(297:535)));
        area_between_curves = trapz(x_g(297:297+common_length-1) - x_g(26:26+common_length-1));
        
        % Store the area in the matrix
        areas(i, j) = area_between_curves;
    end
end

% Plot the area between curves as a contour plot
figure;
contourf( A_values,B_values, areas, 20, 'LineColor', 'none');
colorbar;
xlabel('A');
ylabel('B');
title('Area between Curves - Gain Specific Model');

%%
% Parameters
A = 0.99; B = 0.013;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 19);

F1 = [Null Learn unlearn Learn];

% Define the range of values for A and B
A_values = linspace(0.99, 1.004, 20);
B_values = linspace( 0.006,0.007, 20);

% Preallocate the matrix to store the areas
areas = zeros(length(A_values), length(B_values));

% Loop over different parameter values
for i = 1:length(A_values)
    for j = 1:length(B_values)
        A = A_values(i);
        B = B_values(j);
        
        % Gain specific model with noise
        noise_g = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
        x1_g = zeros(size(F1));
        x2_g = zeros(size(F1));
        x_g = zeros(size(F1));
        for n = 1:length(F1)-1
            x1_g(n+1) = min(0, A * x1_g(n) + B * (F1(n) - x_g(n)) + noise_g(n));
            x2_g(n+1) = max(0, A * x2_g(n) + B * (F1(n) - x_g(n)) + noise_g(n));
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
imagesc(A_values, B_values, areas);
colorbar;
xlabel('A');
ylabel('B');
title('Area between Curves - Gain Specific Model');

%% Multi-rate Model with Noise

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03; As = 0.996; Bs = 0.004;

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 20);

F1 = [Null Learn unlearn Learn];

% Multi-rate model with noise
noise_m = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n)) + noise_m(n);
x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n)) + noise_m(n);
x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end


% Multi-rate model with noise
figure;
subplot(2, 1, 1)
plot(x_m(26:275), 'b', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model with Noise - Learning Phase');
grid on;

subplot(2, 1, 2)
plot(x_m(300:535),'--','color', 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
title('Multi Rate Model with Noise - Re-learning Phase');
grid on;
% Calculate the area between the curves
common_length = min(length(x_m(26:275)), length(x_m(310:544)));
area_between_curves = trapz(x_m(310:310+common_length-1) - x_m(26:26+common_length-1));

% Multi-rate model with noise
figure;
plot(x_m(26:275), 'b', 'linewidth', 1);
hold on;
plot(x_m(310:544), '--', 'color', 'r', 'linewidth', 1);
xlabel('Time');
ylabel('State');
legend('Learning Phase', 'Re-learning Phase');
sgtitle('Learning and Re-learning Phases for Multi-rate model with Noise');
grid on;
% Display the calculated area
disp(['Area between the curves: ' num2str(area_between_curves)]);
%%

% Parameters
A = 0.99; B = 0.013; Af = 0.92; Bf = 0.03;
As_values = linspace(0.99, 1.004, 20);
Bs_values = linspace(0.004, 0.005, 20);

% Trial configuration
null = 25;
learn = 250;

Null = zeros(1, null);
Learn = ones(1, learn);
unlearn = -ones(1, 20);

% Preallocate the matrix to store the calculated areas
areas = zeros(length(As_values), length(Bs_values));

% Loop over different As and Bs values
for i = 1:length(As_values)
    for j = 1:length(Bs_values)
        As = As_values(i);
        Bs = Bs_values(j);
        
% Multi-rate model with noise
noise_m = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.01
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n)) + noise_m(n);
x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n)) + noise_m(n);
x_m(n+1) = x1_m(n+1) + x2_m(n+1);
end
        
        
        % Calculate the area between the curves
        common_length = min(length(x_m(26:275)), length(x_m(310:544)));
        area_between_curves = trapz(x_m(310:310+common_length-1) - x_m(26:26+common_length-1));
        
        % Store the calculated area
        areas(i, j) = area_between_curves;
    end
end

% Plot the areas using contourf
figure;
imagesc(As_values, Bs_values, areas);
colorbar;
xlabel('As');
ylabel('Bs');
title('Area between Curves for different As and Bs');

%%
% Parameters
A = 0.99; B = 0.013;
Af_values = [0.1:0.005:0.95];
Bf_values = [0.01:0.0005:0.03];
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
        
% Multi-rate model with noise
noise_m = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n)) + noise_m(n);
x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n)) + noise_m(n);
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
imagesc(Af_values, Bf_values, areas);
xlabel('Af');
ylabel('Bf');
title('Area between Curves for different Af and Bf');
colorbar;

% Display the calculated areas
disp('Areas between the curves for different Af and Bf:');
disp(areas);
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
 % Multi-rate model with noise
noise_m = 0.01 * rand(size(F1)); % Generate noise between 0 and 0.1
x1_m = zeros(size(F1));
x2_m = zeros(size(F1));
x_m = zeros(size(F1));
for n = 1:length(F1)-1
x1_m(n+1) = Af * x1_m(n) + Bf * (F1(n) - x_m(n)) + noise_m(n);
x2_m(n+1) = As * x2_m(n) + Bs * (F1(n) - x_m(n)) + noise_m(n);
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
%% Function 

% Define the pink noise function
function pn = pinknoise(N)
    % Generate white noise
    wn = randn(1, N);
    % Generate 1/f noise
    a = [0.049922035 -0.095993537 0.050612699 -0.004408786];
    b = [1 -2.494956002   2.017265875  -0.522189400];
    nT60 = round(log(1000)/(1-max(abs(roots(a)))));
    v = randn(1,N+nT60); % Gaussian white noise: N(0,1)
    x = filter(b,a,v);    % Apply 1/f transfer function
    pn = x(nT60+1:end);
end
