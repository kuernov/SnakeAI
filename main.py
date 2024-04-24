import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Inicjalizacja Pygame
pygame.init()

# Inicjalizacja czcionki
font = pygame.font.SysFont('arial', 25)

# Definicja kierunków ruchu węża
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Definicja punktu (x, y)
Point = namedtuple('Point', 'x, y')

# Kolory RGB
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 138, 0)
GREEN1 = (0, 255, 0)
GRAY1 = (169, 169, 169)
GRAY = (100, 100, 100)

# Rozmiar bloku oraz prędkość gry
BLOCK_SIZE = 20
SPEED = 100

# Wczytanie obrazków
food_image = pygame.image.load('img.png')
head_image_template = pygame.image.load('img_1.png')

class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))  # Inicjalizacja okna gry
        pygame.display.set_caption('Snake')  # Ustawienie tytułu okna
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Inicjalizacja stanu gry
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        # Losowe umieszczenie jedzenia na planszy
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)  # Wypełnienie ekranu kolorem czarnym
        # Rysowanie węża
        for i, pt in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.display, GRAY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GRAY1, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        # Rysowanie jedzenia
        scaled_food_image = pygame.transform.scale(food_image, (BLOCK_SIZE + 5, BLOCK_SIZE + 5))
        self.display.blit(scaled_food_image, (self.food.x, self.food.y))
        # Wyświetlanie wyniku
        text = font.render("Score: " + str(self.score), True, WHITE)
        text_rect = text.get_rect()
        text_rect.topleft = (10, 10)
        self.display.blit(text, text_rect)
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        elif np.array_equal(action, [0, 0, 1]):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
