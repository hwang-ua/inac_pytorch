import numpy as np

optimal_value = np.zeros((48,4))

# state 0
# optimal_value[0,0] = -10
optimal_value[0,0] = -1
optimal_value[0,1] = 0.99**13
optimal_value[0,2] = 0.99**13
optimal_value[0,3] = 0.99**12

# state 12-23
optimal_value[23,0] = 0.99
optimal_value[23,1] = 1
optimal_value[23,2] = 0.99**2
optimal_value[23,3] = 0.99**2

optimal_value[12,0] = 0.99**11
optimal_value[12,1] = 0.99**13
optimal_value[12,2] = 0.99**12
optimal_value[12,3] = 0.99**13

for i in range(13,23):
    # optimal_value[i,1] = -10
    optimal_value[i,1] = -1

right = 0.99
# down = -10
down = -1
left = 0.99**3
up = 0.99**3
for i in reversed(range(13,23)):
    optimal_value[i,0] = right
    optimal_value[i,1] = down
    optimal_value[i,2] = left
    optimal_value[i,3] = up

    right *= 0.99
    left *= 0.99
    up *= 0.99

# state 24-35
optimal_value[24,0] = optimal_value[12,0] * 0.99
optimal_value[24,1] = optimal_value[24,0]
optimal_value[24,2] = optimal_value[24,1] * 0.99
optimal_value[24,3] = optimal_value[24,2] * 0.99

optimal_value[35,1] = optimal_value[23,1] * 0.99
optimal_value[35,0] = optimal_value[35,1] * 0.99
optimal_value[35,2] = optimal_value[35,0] * 0.99
optimal_value[35,3] = optimal_value[35,0] * 0.99

right = 0.99**2
down = right
left = 0.99**4
up = left
for i in reversed(range(25,35)):
    optimal_value[i,0] = right
    optimal_value[i,1] = down
    optimal_value[i,2] = left
    optimal_value[i,3] = up

    right *= 0.99
    down *= 0.99
    left *= 0.99
    up *= 0.99

# state 36-47
optimal_value[36,0] = optimal_value[24,0] * 0.99
optimal_value[36,1] = optimal_value[24,1] * 0.99
optimal_value[36,2] = optimal_value[24,2] * 0.99
optimal_value[36,3] = optimal_value[24,3]

optimal_value[47,0] = optimal_value[35,0] * 0.99
optimal_value[47,1] = optimal_value[35,1] * 0.99
optimal_value[47,2] = optimal_value[35,2] * 0.99
optimal_value[47,3] = optimal_value[35,3]

right = 0.99**3
down = right
left = 0.99**5
up = 0.99**4
for i in reversed(range(37,47)):
    optimal_value[i,0] = right
    optimal_value[i,1] = down
    optimal_value[i,2] = left
    optimal_value[i,3] = up

    right *= 0.99
    down *= 0.99
    left *= 0.99
    up *= 0.99

# print(optimal_value)

# optimal_action = np.argmax(optimal_value,axis=1)

# count how many actions are correct for a given Q value array

def argmax(q_values):
    """argmax with random tie-breaking
    Args:
        q_values (Numpy array): the array of action-values
    Returns:
        action (int): an action with the highest value
    """
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)

def count_correct(q_value_array):
    correct_count = 0
    total_state = 48-10
    wrong_action_list = []
    # print(q_value_array)
    for i in range(48):
        optimal_action = argmax(q_value_array[i])
        if i ==0:
            if optimal_action == 3:
                correct_count += 1
            else:
                wrong_action_list.append(i)
        elif i in range(1,12):
            pass
        elif i in range(12,23):
            if optimal_action == 0:
                correct_count += 1
            else:
                wrong_action_list.append(i)
        elif i in [23,35,47]:
            if optimal_action == 1:
                correct_count += 1
            else:
                wrong_action_list.append(i)
        else:
            if optimal_action == 0 or optimal_action == 1:
                correct_count += 1
            else:
                wrong_action_list.append(i)

    return correct_count,wrong_action_list

optimal_path_length = {0:13} # the length of path start from a state under the optimal policy

length = 12
for i in range(12,24):
    optimal_path_length[i] = length # state 12-23
    optimal_path_length[i+12] = length+1 # state 24-35
    optimal_path_length[i+24] = length+2 # state 36-47
    length -= 1


optimal_policy = []
for i in range(48):
    if i in range(1,12):
        optimal_policy.append(-1)
    else:
        optimal_action = argmax(optimal_value[i])
        optimal_policy.append(optimal_action)

if __name__ == '__main__':
    # print(np.flipud(np.array(optimal_policy).reshape(4,12)))
    print(np.delete(optimal_value,range(1,12),0))
    print(optimal_value)
    print(count_correct(optimal_value))


