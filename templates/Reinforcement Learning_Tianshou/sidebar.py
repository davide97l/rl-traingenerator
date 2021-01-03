import streamlit as st

# Define possible models in a dict.
# Format of the dict:
# option 1: model -> code
# option 2 – if model has multiple variants: model -> model variant -> code
POLICIES = {
        "DQN": {"Atari":
                    {"eps_test": 0.005, "eps_train": 1., "eps_train_final": 0.05, "linear_decay_steps": 1000000,
                     "buffer_size": 100000, "lr": 0.0001, "discount_factor": 0.99, "estimation_step": 3,
                     "target_update_freq": 500, "epoch": 100, "step_per_epoch": 10000, "collect_per_step": 10,
                     "batch_size": 32, "training_num": 16, "test_num": 10, "prioritized": False},
                "Classic control":
                    {"eps_test": 0.005, "eps_train": 0.1, "eps_train_final": 0.05, "linear_decay_steps": 50000,
                     "buffer_size": 20000, "lr": 0.0001, "discount_factor": 0.99, "estimation_step": 3,
                     "target_update_freq": 300, "epoch": 10, "step_per_epoch": 1000, "collect_per_step": 100,
                     "batch_size": 64, "training_num": 16, "test_num": 10, "prioritized": False},
                "Box2D":
                    {"eps_test": 0.005, "eps_train": 1., "eps_train_final": 0.05, "linear_decay_steps": 100000,
                     "buffer_size": 100000, "lr": 0.0001, "discount_factor": 0.99, "estimation_step": 3,
                     "target_update_freq": 500, "epoch": 10, "step_per_epoch": 50000, "collect_per_step": 10,
                     "batch_size": 64, "training_num": 16, "test_num": 10, "prioritized": False}
                },
        #"A2C": {},
        #"PPO": {},
}
EPS_DICT = {
    # eps
    "eps_test": {"text": "Epsilon test", "desc": "Value of fixed epsilon to use during testing"},
    "eps_train": {"text": "Epsilon train start", "desc": "Value of epsilon at the beginning of training"},
    "eps_train_final": {"text": "Epsilon train end", "desc": "Minimum value of epsilon during training"},
    "linear_decay_steps": {"text": "Linear Decay Steps", "desc": "How many steps it takes for epsilon to reach its minumum value"},
}
TRAINING_PARAMS_DICT = {
    # training params
    "buffer_size": {"text": "Buffer size", "desc": "How many observation to store in memory"},
    "lr": {"text": "Learning rate", "desc": None},
    "discount_factor": {"text": "Discount factor", "desc": "Higher values of discount_factor (gamma) give more importance to future states, usually between 0.9 and 0.99"},
    "estimation_step": {"text": "Estimation step", "desc": "Number of steps to look ahead. to estimate the reward"},
    "target_update_freq": {"text": "Target update frequency", "desc": "Number of steps between each target model update"},
    "batch_size": {"text": "Batch size", "desc": None},
    "prioritized": {"text": "Prioritized buffer", "desc": "With prioritized buffer more critical states have more probability to be sampled"},
}
PARALLELIZATION_DICT = {
    # parallelization
    "training_num": {"text": "Training environments", "desc": "Number of parallel environments during training"},
    "test_num": {"text": "Test environments", "desc": "Number of parallel environments during test"},
}
TRAINING_DURATION_DICT = {
    # training duration
    "epoch": {"text": "Epochs", "desc": "Training epochs"},
    "step_per_epoch": {"text": "Steps per epoch", "desc": "How many steps for each epoch"},
    "collect_per_step": {"text": "Collect per steps", "desc": "How many frames for each step"},
}
ENVIRONMENTS = {
    "Atari": {
        "Pong": "PongNoFrameskip-v4",
        "Breakout": "BreakoutNoFrameskip-v4",
        "AirRaid": "AirRaid-v0",
        "Alien": "Alien-v0",
        "Amidar": "Amidar-v4",
        "Assault": "Assault-v4",
        "Asterix": "Asterix-v4",
        "Asteroids": "Asteroids-v4",
        "Atlantis": "Atlantis-v4",
        "BankHeist": "BankHeist-v4",
        "BattleZone": "BattleZone-v4",
        "BeamRider": "BeamRider-v4",
        "Berzerk": "Berzerk-v4",
        "Bowling": "Bowling-v4",
        "Boxing": "Boxing-v4",
        "Carnival": "Carnival-v4",
        "Centipede": "Centipede-v4",
        "ChopperCommand": "ChopperCommand-v4",
        "CrazyClimber": "CrazyClimber-v4",
        "DemonAttack": "DemonAttack-v4",
        "DoubleDunk": "DoubleDunk-v4",
        "ElevatorAction": "ElevatorAction-v4",
        "Enduro": "EnduroNoFrameskip-v4",
        "FishingDerby": "FishingDerby-v4",
        "Freeway": "Freeway-v4",
        "Frostbite": "Frostbite-v4",
        "Gopher": "Gopher-v4",
        "Gravitar": "Gravitar-v4",
        "IceHockey": "IceHockey-v4",
        "Jamesbond": "Jamesbond-v4",
        "JourneyEscape": "JourneyEscape-v4",
        "Kangaroo": "Kangaroo-v4",
        "Krull": "Krull-v4",
        "KungFuMaster": "KungFuMaster-v4",
        "MontezumaRevenge": "MontezumaRevenge-v4",
        "MsPacman": "MsPacmanNoFrameskip-v4",
        "NameThisGame": "NameThisGame-v4",
        "Phoenix": "Phoenix-v4",
        "Pitfall": "Pitfall-v4",
        "Pooyan": "Pooyan-v4",
        "PrivateEye": "PrivateEye-v4",
        "Qbert": "QbertNoFrameskip-v4",
        "Riverraid": "Riverraid-v4",
        "RoadRunner": "RoadRunner-v4",
        "Robotank": "Robotank-v4",
        "Seaquest": "SeaquestNoFrameskip-v4",
        "Skiing": "Skiing-v4",
        "Solaris": "Solaris-v4",
        "SpaceInvaders": "SpaceInvadersNoFrameskip-v4",
        "StarGunner": "StarGunner-v4",
        "Tennis": "Tennis-v4",
        "TimePilot": "TimePilot-v4",
        "Tutankham": "Tutankham-v4",
        "UpNDown": "UpNDown-v4",
        "Venture": "Venture-v4",
        "VideoPinball": "VideoPinball-v4",
        "WizardOfWor": "WizardOfWor-v4",
        "YarsRevenge": "YarsRevenge-v4",
        "Zaxxom": "Zaxxom-v4",
    },
    "Classic control": {
        "CartPole": "CartPole-v1",
        "Acrobot": "Acrobot-v1",
    },
    "Box2D": {
        "BipedalWalker": "BipedalWalker-v2",
        "BipedalWalkerHardcore": "BipedalWalkerHardcore-v2",
        "CarRacing": "CarRacing-v0",
        "LunarLander": "LunarLander-v2",
    },
}


DISCRETE_POLICIES = ["DQN", "A2C", "PPO"]


def linear_decay(env_step):
    linear_decay.eps_train = 1
    linear_decay.eps_train_final = 0.05
    if env_step <= 1e6:
        eps = linear_decay.eps_train - env_step / 1e6 * \
              (linear_decay.eps_train - linear_decay.eps_train_final)
    else:
        eps = linear_decay.eps_train_final
    return eps


# not used for now
EPS_CURVE = {
    "linear_decay": {"text": "Linear decay", "func": linear_decay},
}

OPTIMIZERS = ["Adam", "Adadelta", "Adagrad", "Adamax", "RMSprop", "SGD"]


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {}

    with st.sidebar:
        st.write("## Policy :joystick:")
        policy = st.selectbox("Which policy do you want to train?", list(POLICIES.keys()))
        inputs["policy"] = policy

        st.write("## Environment :video_game:")
        inputs["env"] = st.selectbox("Which environment do you want to use?", list(ENVIRONMENTS.keys()))
        env = inputs["env"]

        env_list = ENVIRONMENTS[env].keys()
        if inputs["policy"] in DISCRETE_POLICIES:
            if env == "Classic control":
                env_list = ["CartPole", "Acrobot"]  # only CartPole can be solved by discrete policies
            if env == "Box2D":
                env_list = ["LunarLander"]  # only LunarLander can be solved by discrete policies

        if isinstance(ENVIRONMENTS[env], dict):  # different env tasks
            task = st.selectbox("Which task?", list(env_list))
            inputs["task"] = ENVIRONMENTS[env][task]
        else:  # only one variant
            inputs["task"] = ENVIRONMENTS[env]
        if env == "Atari":
            inputs["frames_stack"] = st.number_input("Frames stacking", 1, None, 4)
            st.markdown('<sup> Stacked frames composing an observation, used to represent object directions </sup>',
                        unsafe_allow_html=True)
        if env in ["Classic control", "Box2d"]:
            inputs["layer_num"] = st.number_input("Number of layers", 1, None, 3)
            st.markdown('<sup> Number of layers of dense network </sup>',
                        unsafe_allow_html=True)
        inputs["early_stop"] = st.checkbox("Early stop", False)
        if inputs["early_stop"]:
            inputs["target_reward"] = st.number_input("Target reward", None, None, 1000)
            st.markdown("<sup> Stop earlier if agent's reward is at least " + str(inputs["target_reward"]) + "</sup>",
                        unsafe_allow_html=True)

        inputs["prioritized"] = None

        section = ["Training duration :hourglass:", "Exploration level :world_map:",
                   "Training parameters :bookmark_tabs:", "Parallelization :fast_forward:"]
        for i, DICT in enumerate([TRAINING_DURATION_DICT, EPS_DICT, TRAINING_PARAMS_DICT, PARALLELIZATION_DICT]):
            st.write("## " + section[i])
            for k, v in POLICIES[policy][inputs["env"]].items():
                if k in DICT.keys():
                    if type(v) == float:
                        inputs[k] = st.number_input(DICT[k]["text"], 0.000, None, v, format="%f")
                    if type(v) == int:
                        inputs[k] = st.number_input(DICT[k]["text"], 1, None, v)
                    if type(v) == bool:
                        inputs[k] = st.checkbox(DICT[k]["text"], v)
                    if DICT[k]["desc"] is not None:
                        st.markdown('<sup>' + DICT[k]["desc"] + '</sup>', unsafe_allow_html=True)

                    if k == "prioritized" and inputs["prioritized"]:
                        inputs["alpha"] = st.number_input("Alpha", 0.000, 1., 0.5)
                        st.markdown(
                            '<sup> Determines how much prioritization is used, with α = 0 corresponding to the uniform case </sup>',
                            unsafe_allow_html=True)
                        inputs["beta"] = st.number_input("Beta", 0.000, 1., 0.5)
                        st.markdown(
                            "<sup> Importance sampling weights, reduce gradients according to samples' importance, higher values correspord to greater downscaling </sup>",
                            unsafe_allow_html=True)

            """if "Exploration level" in section[i]:
                import numpy as np, pandas as pd
                st.markdown('<sup> Epsilon curve </sup>', unsafe_allow_html=True)
                linear_decay.eps_train = inputs["eps_train"]
                linear_decay.eps_train_final = inputs["eps_train_final"]
                x = np.linspace(0, inputs["epoch"] * inputs["step_per_epoch"] * inputs["collect_per_step"], 100)
                y = np.vectorize(linear_decay)(x[:, np.newaxis])
                chart_data = pd.DataFrame(y, x, columns=['lr'])
                st.line_chart(chart_data)"""

            if "Training duration" in section[i]:
                st.markdown(
                    'Training frames: ' + str(inputs["epoch"] * inputs["step_per_epoch"] * inputs["collect_per_step"]),
                    unsafe_allow_html=True,
                )

        st.write("## Configuration :robot_face:")

        inputs["optimizer"] = st.selectbox("Optimizer", OPTIMIZERS)
        inputs["gpu"] = st.checkbox("Use GPU if available", True)
        inputs["tensorboard"] = st.checkbox("Tensorboard visualization", True)
        inputs["save"] = st.checkbox("Save best model", True)
        inputs["watch"] = st.checkbox("Test final performance", True)
        #inputs["args"] = st.checkbox("Use 'args' as input", True)
        inputs["seed"] = st.number_input("Random seed", 0, None, 0)

    return inputs


if __name__ == "__main__":
    show()


"""
streamlit run app/main.py

heroku create
git push heroku main
heroku open
"""
