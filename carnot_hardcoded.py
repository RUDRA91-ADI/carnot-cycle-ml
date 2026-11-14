# simulate the carnot engine using the neural networks:
#input:(common for all 4 process:)
# NEURON FOR ISOTHERMAL./ADIABATIC PROCESS
import torch
P1 = torch.tensor(20.0)
V1 = torch.tensor(26.0)
V2= torch.tensor(30.0)
w1= torch.tensor(0.5, requires_grad=True)
w2 = torch.tensor(0.4,requires_grad=True)
w3 = torch.tensor(0.6,requires_grad=True)
w4 = torch.tensor(0.1,requires_grad=True)
w5=torch.tensor(0.2,requires_grad=True)
W6 = torch.tensor(0.5,requires_grad=True)
b1 = torch.tensor(0.0,requires_grad=True)
b2=torch.tensor(0.0,requires_grad=True)
gamma = 1.4
prediction_iso =  (P1*w1)+(V1*w2)+V2*w3+b1
target_iso = P1*V1/V2 
prediction_ADI = P1*w4 + V1*w5 + V2*W6 + b2
target_ADI = P1*(V1/V2)**gamma
optimizer = torch.optim.SGD([w1,w2,w3,w4,w5,W6,b1,b2,h1,hb,h2,hb2,h3,hb3,h4,hb4],lr=0.01)
for steps in range(1000):
    optimizer.zero_grad()
    prediction_iso =  (P1*w1)+(V1*w2)+V2*w3+b1
    prediction_ADI = P1*w4 + V1*w5 + V2*W6 + b2
    loss_iso = (target_iso-prediction_iso)**2
    loss_adi = (target_ADI-prediction_ADI)**2
    LOSS = loss_iso + loss_adi
    LOSS.backward()
    optimizer.step()


#HIDDEN LAYER NEURONS:
# ISOTHERMAL EXPANSION:
h1 = torch.tensor(0.5,requires_grad=True)
hb = torch.tensor(0.6,requires_grad=True)
prediction_iso_exp= torch.relu(prediction_iso*h1+hb)
#ISOTHERMAL COMPRESSION:
h2 = torch.tensor(0.5,requires_grad=True)
hb2=torch.tensor(0.6,requires_grad=True)
prediction_iso_compression=(torch.relu(prediction_iso*h2+hb2))
#ADIABATIC EXPANSION:
h3 = torch.tensor(0.5,requires_grad=True)
hb3 = torch.tensor(0.6,requires_grad=True)
prediction_ADI_expansion=torch.relu(prediction_ADI*h3+hb3)
#ADIABATIC COMPRESSION:
h4= torch.tensor(0.5,requires_grad=True)
hb4=torch.tensor(0.6,requires_grad=True)
prediction_ADI_compression=torch.relu(prediction_ADI*h4+hb4)

#output layer:
#output layer contains final 2 neuron : 
# whole isothermal and whole adiabaitic process in one :
# with their own sperate weigths and bias:
#ISOTHERMAL PROCESS:
O1 = torch.tensor(0.5,requires_grad=True)
O2 = torch.tensor(0.6,requires_grad=True)
OB1 = torch.tensor(0.5,requires_grad=True)
O3 = torch.tensor(0.5,requires_grad=True)
O4=torch.tensor(0.5,requires_grad=True)
OB2=torch.tensor(0.6,requires_grad=True)
whole_isothermal_process =(prediction_iso_exp*O1)+prediction_iso_compression*O2 +OB1
whole_adiabatic_process = (prediction_ADI_expansion*O3)+(prediction_ADI_compression*O4)+OB2
