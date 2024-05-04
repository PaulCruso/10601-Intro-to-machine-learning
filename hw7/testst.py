values = {'SA': 30, 'SB': 36, 'SC': 0, 'SD': 25, 'SE': 0}

# Rewards as defined earlier
rewards = {
    ('SB', 'SE'): 50,  # Reward for moving from SB to SE
    ('SB', 'SC'): -65, # Penalty for moving from SB to SC
    ('SD', 'SE'): 25   # Reward for moving from SD to SE (though not specified, assuming it's a reward)
}
gamma = 0.9
# Asynchronous value iteration for SA
# SA to SB
values['SD'] = max(0.8 * (rewards[('SD', 'SE')] + gamma * values['SE']) + 0.2 * (gamma * values['SD']),
                   0.8 * gamma * values['SA'] + 0.1 * rewards[('SD', 'SE')] + 0.1 * gamma * values['SD'])

values['SB'] = 0.8 * (rewards[('SB', 'SE')] + gamma * values['SE']) + 0.1 * (gamma * values['SA']) + 0.1 * (gamma * values['SE'])

values['SA'] = 0.8 * (gamma * values['SB']) + 0.2 * (gamma * values['SA'])  # 10% left slip + 10% right slip

# Asynchronous value iteration for SB
# Best action is to move up to SE which gives a reward

# Asynchronous value iteration for SD
# Only possible beneficial action is to move right towards SE
 # 10% left slip + 10% down slip


# Round the values to the first decimal place as instructed
final_values = {state: round(value, 1) for state, value in values.items()}
print(final_values)