# Schritt 2: Imports und Datenvorbereitung
# In diesem Schritt werden benötigte Bibliotheken importiert.
# torch stellt die Kernfunktionalitäten von PyTorch bereit, während torch.nn Module für den Aufbau von neuronalen Netzwerken und torch.optim Optimierungsalgorithmen enthält. torchvision wird für den Zugriff auf Datensätze und Bildverarbeitungsfunktionen verwendet.
# Der MNIST-Datensatz wird über torchvision.datasets.MNIST heruntergeladen und in Trainings- und Testdaten aufgeteilt.
# torch.utils.data.DataLoader erstellt Datenladeprogramme, die das Training des Modells in Mini-Batches ermöglichen.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Laden des Datensatzes MNIST und Vorbereitung der Trainings- und Testdaten
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Schritt 3: Aufbau des neuronalen Netzwerks
# Hier wird die Struktur des neuronalen Netzwerks definiert. Die Klasse SimpleNN erbt von nn.Module und definiert die Schichten des Netzwerks. In diesem Fall handelt es sich um ein einfaches Netzwerk mit drei vollständig verbundenen Schichten (Lineare Schichten), die durch ReLU-Aktivierungsfunktionen verbunden sind.


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten des Eingabebildes
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleNN()

# Schritt 4: Modelltraining
# Die Verlustfunktion (hier nn.CrossEntropyLoss) und der Optimierungsalgorithmus (optim.Adam) werden definiert. Das Modell wird über mehrere Epochen hinweg auf den Trainingsdaten trainiert. Im Trainingsschleifen-Codeabschnitt werden Mini-Batches von Daten geladen, das Modell wird vorwärts durchlaufen, der Verlust wird berechnet, Rückwärtsdurchläufe (Backpropagation) werden durchgeführt und die Gewichte des Modells werden aktualisiert.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 300 == 299:
            print(
                f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 300}')
            running_loss = 0.0

print("Training abgeschlossen!")

# Schritt 5: Modellbewertung
# Nach dem Training wird das trainierte Modell auf dem Testdatensatz evaluiert. Das Modell macht Vorhersagen auf den Testdaten, vergleicht diese mit den tatsächlichen Labels und berechnet die Genauigkeit des Modells, um zu überprüfen, wie gut es auf unbekannten Daten generalisiert.
# Dieses Tutorial bietet eine grundlegende Einführung in das Erstellen, Trainieren und Bewerten eines einfachen neuronalen Netzwerks mit PyTorch auf dem MNIST-Datensatz. Es ist wichtig zu beachten, dass für komplexe Anwendungen weitere Optimierungen, Hyperparameter-Anpassungen und Modifikationen am Modell erforderlich sein können.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    f"Genauigkeit des Modells auf dem Testdatensatz: {100 * correct / total}%")
