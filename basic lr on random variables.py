import numpy as np
import pandas as pd
import torch
 
df = pd.read_csv('train/train.tsv', sep = '\t')
colnames = df.columns

ap = df.loc[df['Typ zabudowy'] == ' apartamentowiec']
a = ap['Rok budowy'].mean()
kam = df.loc[df['Typ zabudowy'] == ' kamienica']
b = kam['Rok budowy'].mean()
empty = df.loc[df['opis'].isnull()]
e = empty['cena'].mean()

variable1 = df['Rok budowy']
variable3 = df['Typ zabudowy']

for i in range(len(variable1):
        if np.isnan(variable1[i]):
            if (variable3[i] == ' apartamentowiec'):
                variable1[i] = a
            if (variable3[i] == ' kamienica'):
                variable1[i] = b

df['Rok budowy'] = variable1

df = df[-df['Rok budowy'].isnull()]
df = df[-df['opis'].isnull()]

df.head()

variable1 = df['Rok budowy']
variable2 = df['opis'].str.len()

target_variable = df['cena']
d = target_variable.mean()

inputs1 = np.array(variable1)
inputs2 = np.array(variable2)
targets = np.array(target_variable)
inputs1 = torch.from_numpy(inputs1)
inputs2 = torch.from_numpy(inputs2)
targets = torch.from_numpy(targets)
print(inputs1)
print(inputs2)
print(targets)

weight1 = torch.randn(1, requires_grad=True)
weight2 = torch.randn(1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)
print(weight1, weight2)
print(bias)

def lr(x, y):
    return x * weight1 + y * weight2 + bias

preds = lr(inputs1, inputs2)
print(preds)

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

for i in range(130):
    preds = lr(inputs1, inputs2)
    loss = mse(preds, targets)
    loss.backward()
    print(preds)
    with torch.no_grad():
        weight1 -= weight1.grad * 1e-7
        weight2 -= weight2.grad * 1e-7
        bias -= bias.grad * 1e-7
        weight1.grad.zero_()
        weight2.grad.zero_()
        bias.grad.zero_()
        
preds = lr(inputs1, inputs2)
loss = mse(preds, targets)
print(loss)
print(preds)
print(targets)

def predict(data):
    variable1 = data['Rok budowy']
    variable2 = data['opis'].str.len()
    variable3 = data['Typ zabudowy']
    target = []

    for i in range(len(variable1)):
            if np.isnan(variable1[i]) and not np.isnan(variable2[i]):
                if (variable3[i] == ' apartamentowiec'):
                    variable1[i] = a
                    target.append(variable1[i] * weight1.item() + variable2[i] * weight2.item() + bias.item())
                if (variable3[i] == ' kamienica'):
                    variable1[i] = b
                    target.append(variable1[i] * weight1.item() + variable2[i] * weight2.item() + bias.item())
                else:
                    target.append(d)
            elif np.isnan(variable1[i]) and np.isnan(variable2[i]):
                target.append(e*d/2)
            elif np.isnan(variable2[i]) and not np.isnan(variable1[i]):
                target.append(e)
            else:
                target.append(variable1[i] * weight1.item() + variable2[i] * weight2.item() + bias.item())
    return(target)

test = pd.read_csv('test-A/in.tsv', sep = '\t')
pd.DataFrame(predict(test)).to_csv('test-A/out.tsv', index=None, header=None, sep='\t')
        
dev = pd.read_csv('dev-0/in.tsv', sep = '\t')
pd.DataFrame(predict(dev)).to_csv('dev-0/out.tsv', index=None, header=None, sep='\t')