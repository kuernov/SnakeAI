import torch
import random
import numpy as np
from collections import deque
from main import Direction, SnakeGame, Point
from model import Linear_QNet
from trainer import QTrainer
import matplotlib.pyplot as plt
from IPython import display

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20

class Agent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, gamma, max_memory, batch_size):
        self.number_of_games = 0
        self.epsilon = 0
        self.gamma = gamma
        self.memory = deque(maxlen=max_memory)
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.trainer = QTrainer(self.model, learning_rate=learning_rate, gamma=gamma)
        self.mse_values = []

    def get_state(self, game):
        head = game.head

        directions = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]
        points = [
            Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)
            for dx, dy in [(1, 0), (-1, 0), (0, -1), (0, 1)]
        ]

        is_collision = lambda point: game.is_collision(point)
        is_collision_in_direction = lambda direction, point: direction == game.direction and is_collision(point)

        danger_straight = any(
            is_collision_in_direction(direction, point) for direction, point in zip(directions, points))
        danger_right = any(
            is_collision_in_direction(direction, point) for direction, point in
            zip(directions[-1:] + directions[:-1], points[-1:] + points[:-1])
        )
        danger_left = any(
            is_collision_in_direction(direction, point) for direction, point in
            zip(directions[1:] + directions[:1], points[1:] + points[:1])
        )

        state = [
            danger_straight,
            danger_right,
            danger_left,
            *map(lambda direction: direction == game.direction, directions),
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            sample = self.memory
        else:
            sample = random.sample(self.memory, BATCH_SIZE)
        self.mse_values.append(self.trainer.mse_values[-1])

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, nextState, done):
        self.trainer.train_step(state, action, reward, nextState, done)

    def get_action(self, state):
        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(agent, game):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    mse_games = []

    while True:
        # Pobierz stan początkowy
        state_old = agent.get_state(game)

        # Wykonaj ruch i uzyskaj nowy stan
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Trenuj pamięć krótkotrwałą
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Zapamiętaj ruch
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Trenuj pamięć długotrwałą, wyświetl wynik
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Gra', agent.number_of_games, 'Wynik', score, 'Rekord:', record)

            average_mse = sum(agent.mse_values) / len(agent.mse_values)
            mse_games.append(average_mse)
            print('Średnie MSE:', average_mse)
            agent.mse_values.clear()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


plt.ion()  # Włącz tryb interaktywny dla matplotlib

def plot(scores, mean_scores):
    # Wyczyść poprzednią wyświetlaną figurę
    display.clear_output(wait=True)

    # Utwórz nową figurę
    plt.figure()

    # Wykres dla wyników pojedynczych gier
    plt.plot(scores, label='Score')

    # Wykres dla średnich wyników
    plt.plot(mean_scores, label='Mean Score')

    # Ustawienia tytułu i osi
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Dodaj legendę
    plt.legend()

    # Wyświetl aktualną wartość wyników
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    # Wyświetl wykres
    plt.show()



if __name__ == '__main__':
    input_size = 11
    hidden_size = 256
    output_size = 3
    learning_rate = 0.001
    gamma = 0.9
    agent = Agent(input_size, hidden_size, output_size, learning_rate, gamma, MAX_MEMORY, BATCH_SIZE)
    game = SnakeGame()
    train(agent, game)