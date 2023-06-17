"""This file just serves as an example on how to configure the zoo
using python scripts instead of yaml files."""
import torch

act_fn = {
    "tanh": torch.nn.Tanh, 
    "relu": torch.nn.ReLU, 
    "elu": torch.nn.ELU, 
    "leaky_relu": torch.nn.LeakyReLU
    }
net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
    "big": dict(pi=[512, 512], vf=[512, 512]),
    "large": dict(pi=[640, 640], vf=[640, 640]),
    }

hyperparams = {
    "SumoEnv-v0": dict(
        normalize=True,
        n_envs=71,
        n_timesteps=int(3.0e7),
        policy="MlpPolicy",
        batch_size=64,
        n_steps=2048,
        gamma=0.98,
        learning_rate=1.1557922545723131e-05,
        ent_coef=1.299211827860584e-08,
        clip_range=0.4,
        n_epochs=5,
        gae_lambda=0.92,
        max_grad_norm=1,
        vf_coef=0.3979058828581206,
        use_sde=False,
        policy_kwargs=dict(
            activation_fn=act_fn["leaky_relu"],
            net_arch=net_arch["medium"]
        ),
    )
}
