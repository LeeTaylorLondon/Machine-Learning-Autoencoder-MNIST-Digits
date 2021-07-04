from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from model_functions import load_data, build_model
from typing import List, NoReturn
import numpy as np
import random
import pygame
import os


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
D_BLUE = (0, 32, 96)
WIDTH, HEIGHT = 1000, 500


def load_model() -> keras.Sequential:
    model = None
    # Try loading model structure and weights from dir
    try:
        model = keras.models.load_model('model')
    except (OSError, AttributeError):
        print(f"Encountered ERROR while loading model.\n"
              f"Building model from saved weights")
    if model is not None:
        print("Loaded model from directory '/model'.")
        return model
    # If above fails try building model layers then loading ONLY weights
    loading_weights_error = False
    model = Sequential(layers=[Dense(units=784, activation='sigmoid', input_dim=784),
                               Dense(units=500, activation='sigmoid'),
                               Dense(units=10, activation='sigmoid')])
    try:
        model.load_weights("model_weights/cp.cpkt")
    except (ImportError, ValueError):
        loading_weights_error = True
        print("Encountered ERROR while loading weights.\n"
              "Ensure module h5py is installed and directory to weights is correct.")
    if loading_weights_error is False:
        print("Created model layers and loaded weights.")
        return model
    # If all above fails then train a model, store it, and return it
    print("Loading model and loading weights failed. Proceeding to\n"
          "build a model, store it and it's weights in /model and /model_weights.")
    if len(os.listdir('/mnist_data')) != 2:
        raise TypeError("Cannot execute above. mnist_data folder does not contain training and test data.")
    return build_model()


def create_text(arr:List[str], font_size:int) -> List[pygame.font.SysFont]:
    rv = []
    font = pygame.font.SysFont('chalkduster.tff', font_size)
    for s in arr:
        rv.append(font.render(s, True, BLACK))
    return rv


class Window:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.run = True
        self.clock = pygame.time.Clock()
        # non pygame attrs
        self.pixels = []
        self.pixels_out = []
        self.text = create_text(["[Key T: Query Neural Network]",
                                 "[Key C: Clear drawing]", ""], 16)
        self.model = load_model()
        # continuous loop
        self.render()

    def clear_screen(self) -> NoReturn:
        self.pixels = [] #[[255 for _ in range(28)] for _ in range(28)]
        self.pixels_out = []

    def random_image(self) -> NoReturn:
        self.clear_screen()
        x_train, _, x_test, _ = load_data()
        i = random.randint(0, len(x_train) - 1)
        for vec in x_train[i]:
            self.pixels.append(list(np.array(255 - (vec * 255), dtype="int16")))
        return i

    def query_ae(self, i):
        x_train, _, x_test, _ = load_data()
        #ae = build_model()
        ae_out = self.model.predict([x_train[i].reshape(-1, 28, 28, 1)])[0]
        for vec in ae_out:
            for i, n in enumerate(vec):
                if n > 1: vec[i] = 1
        #self.pixels_out = 255 - (ae_out * 255.0)
        for y,vec in enumerate(ae_out):
            self.pixels_out.append([])
            vec = list(np.array(255 - (vec * 255), dtype="int16"))
            for x,n in enumerate(vec):
                self.pixels_out[y].append(int(n[0]))

    def render(self) -> NoReturn:
        while self.run:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.run = False
                    if event.key == pygame.K_t:
                        i = self.random_image()
                        self.query_ae(i)
                    if event.key == pygame.K_c:
                        self.clear_screen()
            self.screen.fill(WHITE)
            # --[render start]-- It might be slightly slower to put
            # the below code blocks i.e render pixels, render border, handle input
            # into functions or methods of class Window --[render start]--
            # Render pixels (left square)
            for y,vec in enumerate(self.pixels):
                for x,p in enumerate(vec):
                    pygame.draw.rect(self.screen, (p, p, p),
                                     [40 + (x * 10), 40 + (y * 10), 10, 10])
            # Render pixels (NN out)
            for y,vec in enumerate(self.pixels_out):
                for x,p in enumerate(vec):
                    if p > 255: vec[x] = 255
                    elif p < 0: vec[x] = 0
                    p = vec[x]
                    pygame.draw.rect(self.screen, (p, p, p),
                                     [450 + (x * 10), 40 + (y * 10), 10, 10])
            # Render text
            # self.screen.blit(self.text[0], (10, 10))
            # self.screen.blit(self.text[1], (180, 10))
            # self.screen.blit(self.text[2], (30, 300))
            # Render square left
            pygame.draw.rect(self.screen, D_BLUE, [40, 37, 283, 285], 5)
            # Render square right
            pygame.draw.rect(self.screen, D_BLUE, [450, 37, 283, 285], 5)
            # --[render end]--
            pygame.display.flip()
            self.clock.tick(144)
        pygame.quit()


if __name__ == '__main__':
    Window()