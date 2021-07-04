from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from model_functions import load_data, build_model
from typing import List, NoReturn, Tuple
import numpy as np
import random
import pygame
import os


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
D_BLUE = (0, 32, 96)
WIDTH, HEIGHT = 780, 380


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
        self.text = create_text(["Input to Auto Encoder",
                                 "Auto Encoder Output",
                                 "[Key D: Pass Input to Auto Encoder]",
                                 "[Key T: Load Random Image]",
                                 "[Key C: Clear Input]"],
                                16)
        self.model = load_model()
        self.x_test, _, _, _ = load_data()
        # continuous loop
        self.render()

    def clear_screen(self) -> NoReturn:
        self.pixels = [] #[[255 for _ in range(28)] for _ in range(28)]
        self.pixels_out = []

    def random_image(self) -> NoReturn:
        self.clear_screen()
        x_train, _, x_test, _ = load_data()
        i = random.randint(0, len(self.x_test) - 1)
        for vec in self.x_test[i]:
            self.pixels.append(list(np.array(255 - (vec * 255), dtype="int16")))
        return i

    def query_ae(self, i=0, mnist=True):
        if not self.pixels: return
        # load random image from mnist dataset
        if mnist:
            x_train, _, x_test, _ = load_data()
            ae_out = self.model.predict([self.x_test[i].reshape(-1, 28, 28, 1)])[0]
        # preprocess user drawing
        else:
            ae_out = self.model.predict(((255.0 - np.array(self.pixels, dtype="float32")
                                          )/ 255.0).reshape(-1, 28, 28, 1))[0]
        # process ae_out to be rendered
        for vec in ae_out:
            for i, n in enumerate(vec):
                if n > 1: vec[i] = 1
        self.pixels_out = []
        for y,vec in enumerate(ae_out):
            self.pixels_out.append([])
            vec = list(np.array(255 - (vec * 255), dtype="int16"))
            for x,n in enumerate(vec):
                self.pixels_out[y].append(int(n[0]))

    def draw_pixels(self, pixels:List[List[int]], x_offset:int, y_offset:int) -> NoReturn:
        for y, vec in enumerate(pixels):
            for x, p in enumerate(vec):
                if p > 255: vec[x] = 255
                elif p < 0: vec[x] = 0
                p = vec[x]
                pygame.draw.rect(self.screen, (p, p, p),
                                 [x_offset + (x * 10),
                                  y_offset + (y * 10), 10, 10])

    def draw_text(self, coords:List[Tuple[int, int]]) -> NoReturn:
        for text_obj, xy_pair in zip(self.text, coords):
            self.screen.blit(text_obj, xy_pair)

    def render(self) -> NoReturn:
        while self.run:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.run = False
                    if event.key == pygame.K_t:
                        i = self.random_image()
                        self.query_ae(i)
                    if event.key == pygame.K_d:
                        self.query_ae(mnist=False)
                    if event.key == pygame.K_c:
                        self.clear_screen()
            self.screen.fill(WHITE)
            # --[render start]--
            self.draw_text([(110,20), (530, 20), (50, 330), (50, 345), (50, 360)])
            self.draw_pixels(self.pixels, 40, 40) # input section
            self.draw_pixels(self.pixels_out, 450, 40) # output section
            pygame.draw.rect(self.screen, D_BLUE, [40, 37, 283, 285], 5) # input border
            pygame.draw.rect(self.screen, D_BLUE, [450, 37, 283, 285], 5) # output border
            # Handle mouse input (for drawing)
            if pygame.mouse.get_pressed(3)[0]:
                x, y = pygame.mouse.get_pos()
                x -= 40
                y -= 40
                # Checks if mouse is in drawing square
                if not(x >= 270 or x < 0 or y >= 270 or y <= 0):
                    if (o := x % 10) < 5: x -= o
                    else: x += (10 - o)

                    if (o := y % 10) < 5: y -= o
                    else: y += (10 - o)
                    try:
                        if (self.pixels[y//10][x//10]) != 0:
                            self.pixels[y//10][x//10] -= 51
                    except IndexError:
                        self.pixels = [[255 for _ in range(28)] for _ in range(28)]
            # --[render end]--
            pygame.display.flip()
            self.clock.tick(144)
        pygame.quit()


if __name__ == '__main__':
    Window()