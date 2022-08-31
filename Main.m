%% Main function for implementing SMO to solve DNN training problems
%% Programmer: Thang Le-Duc
%  Emails: thangld@sju.ac.kr; le.duc.thang0312@gmail.com
%% Begin main function
clear all, close all, clc
rng('default')

addpath('./SMO Package')

%% Initialize DNN models, algorithm settings and load dataset
epoch = 20
Funcname = 'MNIST';   % 'MNIST'
problem = DNNStructure(Funcname);
nn = DNN_Setup(problem);
lr = DNN_LearningRate;
valid.Active = 0;
[problem,nn] = ClassificationProblems(Funcname,valid,nn,problem);
smo = SMO_Setup(epoch);
batch = Batch_Setup;
rank = RankSetup(smo);

%% Training DNN model by SMO
[DNNparam,traintime,results,rank] = SMO_DNNTraining(smo,problem,nn,lr,batch,rank);

%% Collect statistical Results
if nn.TestPred == 0
    [TestLoss,TrainLoss,TestPred,TrainPred,TestEmpRisk,TrainEmpRisk,yfTestPred,StatsTest,...
            L0Norm,L1Norm,L2Norm] = DNNValidation(Best_x,problem,nn,0,rank);
    results.TestLoss = TestLoss;
    results.TrainLoss = TrainLoss;
    results.TestPred = TestPred;
    results.TrainPred = TrainPred;
    results.TestEmpRisk = TestEmpRisk;
    results.TrainEmpRisk = TrainEmpRisk;
    results.yfTestPred = yfTestPred;
    results.StatsTest = StatsTest;
    results.L0Norm = L0Norm;
    results.L1Norm = L1Norm;
    results.L2Norm = L2Norm;
end
TestLoss = results.TestLoss(epoch);
TestErr = 100 - results.TestPred(epoch);
fprintf('Test loss is: %s, \n',mat2str(TestLoss));
fprintf('Test error is: %s, \n',mat2str(TestErr));
fprintf('Training time (s) is: %s, \n',mat2str(traintime));

%% Save statistical results
save DNN.mat DNNparam smo problem nn batch rank
save results.mat results traintime