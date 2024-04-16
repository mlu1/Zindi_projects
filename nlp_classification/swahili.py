import os
import datetime
from tqdm import tqdm
import random
import shutil
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from nltk import word_tokenize

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel, BertConfig, AdamW, get_linear_schedule_with_warmup
def random_seed(seed_value, use_cuda): 
    np.random.seed(seed_value)
 #cpu vars
    torch.manual_seed(seed_value) 
# cpu  vars
    random.seed(seed_value)
 # Python 
    if use_cuda: 
        torch.cuda.manual_seed(seed_value) 
        torch.cuda.manual_seed_all(seed_value) 
# gpu vars
        torch.backends.cudnn.deterministic = True 
 #needed
        torch.backends.cudnn.benchmark = False 
#Remember to use num_workers=0 when creating the DataBunch.

random_seed(2023, True)
RANDOM_SEED = None
NUM_OF_CLASSES = 3
MAX_LENGTH = 277
TO_USE_POOLING_OUTPUT = False
RAITO_OF_EVAL_DATA = 0.1

MAX_EPOCH = 16
TRAIN_BATCH_SIZE =64
EVAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

LEARNING_RATE = 8e-6
WEIGHT_DECAY = 0.001
DROPOUT_RATE = 0.3
TO_USE_CLIP_GRAD = True



class Preprocessor:
    """
    preprocessing unit
    """
    INDEX = "ID"
    tweet = "Tweets"
    HYPOTHESIS = "hypothesis"
    LANG_AVB = "lang_avb"
    LANGUAGE = "language"
    label = "Labels"

    def __init__(self, trainDataPath='../data/Train.csv', testDataPath='../data/Test.csv', randomSeed=RANDOM_SEED):
        self.trainDataPath = trainDataPath
        self.testDataPath = testDataPath
        self.csvTrainData = None
        self.csvTestData = None
        self.random = random
        if randomSeed is not None:
            self.random.seed(randomSeed)


    def map_values(self,x):
        if x ==0:
            return x
        elif x==1:
            return 1
        else:
            return 2

    def prepareTrainAndEvalData(self):
        data = []
        self.loadTrainData()
        for index, item in self.csvTrainData.iterrows():
            texts = []
            texts.append(self.cleanTexts(item[self.tweet]))
            texts.append(self.map_values(item[self.label]))
            data.append(texts)
       
        
        lenTrainData = int(len(data) * (1 - RAITO_OF_EVAL_DATA))
        self.random.shuffle(data)
        trainData = data[:lenTrainData]
        evalData = data[lenTrainData:]

        return trainData, evalData

    def prepareTestData(self):
        data = []
        self.loadTestData()
        for index, item in self.csvTestData.iterrows():
            texts = []
            texts.append(item[self.INDEX])
            texts.append(self.cleanTexts(item[self.tweet]))
            data.append(texts)
        return data

    def loadTrainData(self):
        with open(self.trainDataPath) as file:
            self.csvTrainData = pd.read_csv(file,header=0)
    
    def loadTestData(self):
        with open(self.testDataPath) as file:
            self.csvTestData = pd.read_csv(file,header=0)
    
    # If you want to do additional preproccessings, write them in this unit. 
    def cleanTexts(self, text):
        text = text.replace('\t', ' ')
        text = text.replace('[^a-zA-Z]',' ') 
        text = text.lower()                #convert text to lower-case
        text = re.sub('â€˜','',text) 
        '''
        for puctuation in string.punctuation:
            if (puctuation == '.') or (puctuation == ','):
                continue
            else:
                text = text.replace(puctuation, ' ')

        '''
        return text
         
 
    
# For train data
class TrainDataSet(Dataset):
    def __init__(self, trainData, tokenizer):
        super(TrainDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.trainData = trainData

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, index):
        return self.getData(index)

    def encodeToTokenIds(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens.append(self.tokenizer.sep_token)
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)

    def getData(self, index):
        outputData = []
        premiseTokenIds = self.encodeToTokenIds(' '.join(self.trainData[index][0].split()))
        labelIndex = torch.tensor(self.trainData[index][1], dtype=torch.long)
        
        inputs = self.tokenizer.encode_plus(     
            premiseTokenIds,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            pad_to_max_length=True,
            truncation=True
        )

        outputData.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['token_type_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['attention_mask'], dtype=torch.long))

        return outputData, labelIndex

    def getTokenizer(self):
        return self.tokenizer

# For eval data
class EvalDataSet(Dataset):
    def __init__(self, evalData, tokenizer):
        super(EvalDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.evalData = evalData

    def __len__(self):
        return len(self.evalData)

    def __getitem__(self, index):
        return self.getData(index=index)

    def encodeToTokenIds(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens.append(self.tokenizer.sep_token)
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)
 

    def getData(self, index):
        outputData = []
        premiseTokenIds = self.encodeToTokenIds(' '.join(self.evalData[index][0].split()))
        labelIndex = torch.tensor(self.evalData[index][1], dtype=torch.long)
        
        inputs = self.tokenizer.encode_plus(     
            premiseTokenIds,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            pad_to_max_length=True,
            truncation=True
        )

        outputData.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['token_type_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['attention_mask'], dtype=torch.long))

        return outputData, labelIndex

# For test data
class TestDataSet(Dataset):
    def __init__(self, testData, tokenizer):
        super(TestDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.testData = testData
        self.labels = []

    def __len__(self):
        return len(self.testData)

    def __getitem__(self, index):
        return self.getData(index)

    def encodeToTokenIds(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens.append(self.tokenizer.sep_token)
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)

    def getData(self, index):
        outputData = []
        self.labels.append(self.testData[index][0])
        #premiseTokenIds = self.encodeToTokenIds(' '.join(self.testData[index]))
        print(testData[index][1])
        #premiseTokenIds = self.encodeToTokenIds(' '.join(self.testData[index][0]))
        #labelIndex = torch.tensor(self.evalData[index][1], dtype=torch.long)
        premiseTokenIds = self.encodeToTokenIds(' '.join(self.testData[index][1].split())) 


        inputs = self.tokenizer.encode_plus(     
            premiseTokenIds,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            pad_to_max_length=True,
            truncation=True
        )

        outputData.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['token_type_ids'], dtype=torch.long))
        outputData.append(torch.tensor(inputs['attention_mask'], dtype=torch.long))

        return outputData

# This method provides train and eval dataloaders. 
class BertDataLoader:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.preprocessor = Preprocessor()
        
    def getTrainAndEvalDataLoader(self):
        trainData, evalData = self.preprocessor.prepareTrainAndEvalData()

        trainDataSet = TrainDataSet(trainData=trainData, tokenizer=self.tokenizer)
        evalDataSet = EvalDataSet(evalData=evalData, tokenizer=self.tokenizer)
        
        trainDataLoader = DataLoader(trainDataSet, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        evalDataLoader = DataLoader(evalDataSet, batch_size=EVAL_BATCH_SIZE, shuffle=True)

        dataLoadersDict = {'train': trainDataLoader, 'eval': evalDataLoader}
        return dataLoadersDict

dataLoader = BertDataLoader()
dataLoadersDict = dataLoader.getTrainAndEvalDataLoader()


class Model(nn.ModuleList):
    def __init__(self):
        super(Model, self).__init__()
        self.bertModel = SingleBERT(toUsePooling=TO_USE_POOLING_OUTPUT)

        bertConfig = self.bertModel.getConfig()
        hiddenSize = bertConfig.hidden_size
        
        self.classifier = Classifier(hiddenSize=hiddenSize, nClasses=NUM_OF_CLASSES, dropoutRate=DROPOUT_RATE)

    def forward(self, input, tokenTypeIds, attentionMask):
        output = self.bertModel(input=input, tokenTypeIds=tokenTypeIds, attentionMask=attentionMask)
        output = self.classifier(output)
        return output 

class SingleBERT(nn.Module):
    def __init__(self, toUsePooling=False):
        super(SingleBERT, self).__init__()
        self.toUsePooling = toUsePooling
        self.bertConfig = BertConfig.from_pretrained('bert-base-multilingual-cased')
        self.bertModel = BertModel.from_pretrained('bert-base-multilingual-cased')

        for param in self.bertModel.parameters():
            param.requires_grad = True

    def forward(self, input, tokenTypeIds, attentionMask):
        lastLayerOutPut, poolingOutput = self.bertModel(input, attention_mask=attentionMask, token_type_ids=tokenTypeIds, return_dict=False)
        if self.toUsePooling:
            return poolingOutput
        return lastLayerOutPut[:, 0, :]

    def getConfig(self):
        return self.bertConfig

class Classifier(nn.Module):
    def __init__(self, hiddenSize, nClasses, dropoutRate):
        super(Classifier, self).__init__()
        self.dropout1 = nn.Dropout(p=dropoutRate)
        self.linear1 = nn.Linear(in_features=hiddenSize, out_features=hiddenSize)
        self.batchNorm = nn.BatchNorm1d(num_features=hiddenSize, eps=1e-05, momentum=0.1, affine=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout2 = nn.Dropout(p=dropoutRate)
        self.linear2 = nn.Linear(in_features=hiddenSize, out_features=nClasses)

        nn.init.normal_(self.linear1.weight, std=0.04)
        nn.init.normal_(self.linear2.weight, mean=0.5, std=0.04)
        nn.init.normal_(self.linear1.bias, 0)
        nn.init.normal_(self.linear2.bias, 0)

    def forward(self, input):
        output = self.dropout1(input)
        output = self.linear1(output)
        output = self.batchNorm(output)
        output = self.activation(output)
        output = self.dropout2(output)
        output = self.linear2(output)
        return output

model = Model()


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, inputTensor, lavelTensor):
        return self.loss(inputTensor, lavelTensor)


criterion = LossFunction()


def saveWeights(model):
    saveDirectoryPath = './weight'
    if not os.path.exists(saveDirectoryPath):
        os.makedirs(saveDirectoryPath)

    time_now = datetime.datetime.now()
    time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'
    savePath = saveDirectoryPath +'/'+ str(time_info) + '.pth'

    try:
        torch.save(model.state_dict(), savePath)
        print('Parameters were successfully saved!')
    except:
        print('Parameters were not successfully saved!')
    
    return None

# For drawing the loss and accuracy rate.
def saveLogs(logs):
    time_now = datetime.datetime.now()
    time_info = f'{time_now.year}-{time_now.month}-{time_now.day}_{time_now.hour}-{time_now.minute}-{time_now.second}'

    saveDirectoryPath = './logs/' + str(time_info)
    if not os.path.exists(saveDirectoryPath):
        os.makedirs(saveDirectoryPath)

    savePathLoss = saveDirectoryPath  + '/loss' + '.jpg'
    savePathAcurracy = saveDirectoryPath  + '/accuracy' + '.jpg'

    x = [num for num in range(MAX_EPOCH)]
    epochTrainLosses = logs[0].tolist()
    epochEvalLosses = logs[1].tolist()
    epochTrainAccuracies = logs[2].tolist()
    epochEvalAccuracies = logs[3].tolist()

    # Loss
    plt.plot(x, epochTrainLosses, color='red', label='Train Loss')
    plt.plot(x, epochEvalLosses, color='blue', label='Eval Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid()
    

    plt.savefig(savePathLoss)
    plt.clf()

    # Accuracy
    plt.plot(x, epochTrainAccuracies, color='red', label='Train Accuracy')
    plt.plot(x, epochEvalAccuracies, color='blue', label='Eval Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid()
 
    plt.savefig(savePathAcurracy)
    
    return None

# Check available device.
def checkDevice():
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("Device: ", device)
    return device

def trainModel(model, dataLoadedrsDict, criterion, optimizer):
    device = checkDevice()
    model.to(device)
    criterion.to(device)
    torch.backends.cudnn.benchmark = True

    total_steps = len(dataLoadedrsDict['train'].dataset) * MAX_EPOCH
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps = total_steps)
    
    # For progress record.
    trainLossLogs = np.zeros(shape=MAX_EPOCH, dtype=np.float)
    evalLossLogs = np.zeros(shape=MAX_EPOCH, dtype=np.float)
    trainAccuracyLogs = np.zeros(shape=MAX_EPOCH, dtype=np.float)
    evalAccuracyLogs = np.zeros(shape=MAX_EPOCH, dtype=np.float)
    
    # Core part of training
    for epoch in range(MAX_EPOCH):
        for phase in ['train', 'eval']:
            print('-------------------------------------------------------------------------------------------------------------------------------------')
            print("Phase: ", phase)

            if phase == 'train':
                model.train()
            else:
                model.eval()

            epochLoss = 0.0
            epochCorrects = 0
            
            for batch in tqdm(dataLoadedrsDict[phase]):
                inputs = batch[0][0].to(device)
                tokenTypeIds = batch[0][1].to(device)
                attentionMask = batch[0][2].to(device)
                labels = batch[1].to(device)

            
            
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(input=inputs, tokenTypeIds=tokenTypeIds, attentionMask=attentionMask)
                    loss = criterion(outputs, labels)
                    
                    _, predictions = torch.max(outputs, dim=1)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        if TO_USE_CLIP_GRAD:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    epochLoss += loss.item() * TRAIN_BATCH_SIZE
                    epochCorrects += torch.sum(predictions == labels, dim=0)

            epochLoss = epochLoss / len(dataLoadedrsDict[phase].dataset)
            epochAccuracy = epochCorrects.double() / len(dataLoadedrsDict[phase].dataset)

            if phase == 'train':
                trainLossLogs[epoch] = epochLoss
                trainAccuracyLogs[epoch] = epochAccuracy
            else:
                evalLossLogs[epoch] = epochLoss
                evalAccuracyLogs[epoch] = epochAccuracy

            print('Epoch: {}/{}  |  Loss: {:.4f}  |  Acc: {:.4f}'.format(epoch+1, MAX_EPOCH, epochLoss, epochAccuracy))

    saveWeights(model=model)
    logs = [trainLossLogs, evalLossLogs, trainAccuracyLogs, evalAccuracyLogs]
    saveLogs(logs=logs)

    return model


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, correct_bias=False)
trainedModel = trainModel(model=model, dataLoadedrsDict=dataLoadersDict, criterion=criterion, optimizer=optimizer)


def predict(model, testDataLoader):
    predictions = []
    device = checkDevice()
    model.to(device)

    model.eval()

    for batch in tqdm(testDataLoader):
        inputs = batch[0].to(device)
        tokenTypeIds = batch[1].to(device)
        attentionMask = batch[2].to(device)

        outputs = model(input=inputs, tokenTypeIds=tokenTypeIds, attentionMask=attentionMask)
        _, prediction = torch.max(outputs, dim=1)
        prediction = prediction.flatten().tolist()
        predictions += prediction
    
    return predictions

# functions for creating a submission file.
class Submitter:
    def __init__(self, dataSet):
        self.dataLoader = DataLoader(dataSet, batch_size=TEST_BATCH_SIZE, shuffle=False)
        self.ids = dataSet.labels

    def makeFile(self, model, weightPath):
        if weightPath is not None:
            weights = torch.load(weightPath, map_location={'cuda:0': 'cpu'})
            model.load_state_dict(weights)


        outputLabels = predict(model=model, testDataLoader=self.dataLoader)

        saveDirectoryPath = './'
        if not os.path.exists(saveDirectoryPath):
            os.makedirs(saveDirectoryPath)

        savePath =  'submission.csv'
        dataFrame = pd.DataFrame(list(zip(self.ids, outputLabels)), columns=['id', 'prediction'])

        try:
            dataFrame.to_csv(savePath, index=False)
            print("Successed !!")
        except FileNotFoundError:
            print("Failed !!")
# Prepare dataset for test. 
testData = Preprocessor().prepareTestData()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', local_files_only=True)
dataSet = TestDataSet(testData=testData, tokenizer=tokenizer)

# make a submittion file
submitter = Submitter(dataSet=dataSet)
submitter.makeFile(model=trainedModel, weightPath=None)

