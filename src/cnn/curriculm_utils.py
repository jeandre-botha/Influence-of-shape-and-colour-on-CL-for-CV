import math

def calculate_available_colours(t, t_g, c_0, total_colours):
    c_t = min(1, t*((1-c_0)/t_g)+c_0)
    return math.ceil(c_t*total_colours)

def calculate_num_easiest_examples(t, t_g, u_0, p, total_examples):
    u_root_p = t*((1-u_0**p)/t_g) + u_0**p
    u_root_p = u_root_p**(1/p)
    u_root_p = min(1, u_root_p)
    return math.ceil(u_root_p*total_examples)