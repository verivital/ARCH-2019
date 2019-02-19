%% Collect data Mountain Car
% Run Simulink model and collect data
% set the solver and simulation options 
% clc;clear
%opt = simset('solver','ode4','FixedStep',0.02);

%% Generate training data
s = 30; % number of simulation I want to record
for i=1:s
    rng(1); %set the seed for reproducible results
    v0 = 0.07*rand; %have a inital velocity between [-0.07,0.07]
    p0 = 0.2*rand - 0.6; %have a inital position between [-0.6,-0.4]
    % [T,X] = sim('Mountain_car',[0 15],opt); %simulate system and record data
    [T,X] = sim('Mountain_car',[0 100]); %simulate system and record data
    % Create data object for each iteration
    data = iddata(out1,in1,0.02);
    % Merge all simulations together
    if i>1
        dataf = merge(dataf,data);
    else
        dataf = merge(data);
    end
%     if i>1
%         in = catsamples(in,in1);
%         out = catsamples(out,out1);
%     else
%         in = in1;
%         out = out1;
%     end
end

clearvars -except dataf s;
%save('MC_data','dataf'); 

%% Train Neural Network controller with only relus and linear function
% Load data 
d = dataf';
out = d.y;
in = d.u;
for i=1:s
    in{i} = in{i}';
    out{i} = out{i}';
end
net = network(1,3,[1;1;1],[1;0;0],[0 0 0;1 0 0;0 1 0],[0,0,1]); %4 inputs,2 layers, 1 output

% add the rest of structure
net.inputs{1}.size = 2; % size of inputs
net.layers{1}.size = 16; %size of layers
net.layers{2}.size = 16;
net.layers{3}.size = 1;
net.layers{1}.transferFcn = 'poslin'; %poslin = relu
net.layers{2}.transferFcn = 'poslin'; %poslin = relu
net.layers{3}.transferFcn = 'purelin'; % purelin = linear
net.initFcn = 'initlay';
net.trainFcn = 'trainbr'; %Bayesian regularization
net.layers{1}.initFcn = 'initnw';
net.layers{2}.initFcn = 'initnw';
net.layers{3}.initFcn = 'initnw';
%net.inputWeights{1}.delays = 0:1;

%Store the output simulations
y1 = net(in);
net = init(net);
net = train(net,in,out);
y2 = net(in);
% out = cell2mat(out);

% Calculate fit percentage
% error = abs(y2-out);
% error = error./out*100;
% error = sum(error)/length(error);
% fit_percentage_cont = 100- error;

%% Save the net data in a dictionary
mxv = [-0.4 0.07];
mnv = [-0.6 -0.07];

W = {{net.IW{1} net.LW{2} net.LW{6}}};
w = W{1};
b = {{net.b{1} net.b{2} net.b{3}}};
ni = size(w{1},2);
no = net.output.size;
nl = length(net.layers);
nn = prod(size(w{1}))+prod(size(w{2}))+prod(size(w{3}));
ls = [net.layers{1}.size,net.layers{2}.size,net.layers{3}.size];

% Define a dictionary with net's info
nnetwork = struct('number_of_outputs',no,'number_of_inputs',ni,...
    'number_of_layers',nl,'number_of_weights',nn,'W',W,...
    'b',b,'layer_size',ls,'act_functions',{{'relu','relu','linear'}},...
    'number_of_neurons',sum(ls),'max',mxv,'min',mnv);
% save the net as Tran's tool
% save('MountainCar_ReluController','nnetwork');