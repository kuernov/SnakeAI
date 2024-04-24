import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), self.learning_rate)
        self.loss_function = nn.MSELoss()
        self.mse_values = []

    def train_step(self, state, action, reward, new_state, done):
        # Konwersja danych wejściowych na tensory
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(new_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # Rozszerzenie wymiarów wejściowych, jeśli są jednowymiarowe
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Predykcja wartości Q dla stanu aktualnego
        prediction = self.model(state)
        target = prediction.clone()

        # Aktualizacja wartości Q na podstawie nowych danych
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                # Obliczenie maksymalnej wartości Q dla nowego stanu
                max_q_value = torch.max(self.model(next_state[i]))
                if len(max_q_value.shape) > 0:
                    max_q_value = max_q_value.unsqueeze(0)
                Q_new = reward[i] + self.gamma * max_q_value
            # Aktualizacja wartości Q w celu nauki
            target[i][torch.argmax(action).item()] = Q_new

        # Obliczenie straty i propagacja wsteczna
        self.optimizer.zero_grad()
        loss = self.loss_function(target, prediction)
        mse = loss.item()
        self.mse_values.append(mse)
        loss.backward()
        self.optimizer.step()