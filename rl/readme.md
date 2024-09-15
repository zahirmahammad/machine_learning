### Q Learning

Q table

1. Rows - States

2. Columns - Actions

Update Rule - $Q(s, a) = Q(s, a) + \alpha[R + \gamma*max_aQ(s`, a)-Q(s, a)]$
    
- where $\alpha$ is the learning rate and $\gamma$ is the discount factor

Epsilon Greedy $\epsilon = 0.9$

- Choose greedy action from Q table with a probability of $(1-\epsilon)$
- Decrease the values of $\epsilon$ as number of episodes increases, using a `decay_rate`