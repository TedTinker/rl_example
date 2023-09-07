#%%
import torch.nn as nn
from torchinfo import summary as torch_summary

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) Model. Q^\pi approximates Q^* by estimating r + \gamma Q^\pi (s_t+1, \pi(s_t+1))
    """
    def __init__(self, n_states=4, n_actions=2):
        """
        Initialize the DQN model.
        
        Args:
        - n_states (int): Number of state inputs.
        - n_actions (int): Number of action outputs.
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state):
        """
        Forward pass through the network.
        """
        return self.network(state)

if __name__ == "__main__":
    # Initialize and print the DQN model
    dqn = DQN()
    print(dqn)
    
    # Print the model summary
    print("\nModel Summary:")
    print(torch_summary(dqn, (3, 4)))
# %%
