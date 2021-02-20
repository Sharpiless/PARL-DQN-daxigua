from State2NN import AI_Board
import numpy as np
from nn import neural_net

NUM_INPUT = 11

def play(model):

    game_state = AI_Board()
    game_state.reset()

    # Do nothing to get initial.
    state,_ = game_state.next(0.5)

    # Move.
    while True:
        
        # Choose action.
        action = (np.argmax(model.predict(np.array([state]))[0]))

        # Take action.
        state,reward = game_state.frame_step(action)
        
        if reward == -1000:
            break


if __name__ == "__main__":
    saved_model = 'results/saved-models/256-256-512-50000-ver19-300000.h5'
    model = neural_net(NUM_INPUT, [256, 256], saved_model)
    play(model)
