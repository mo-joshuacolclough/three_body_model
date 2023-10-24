from model import Model, State

import matplotlib.pyplot as plt
import numpy as np

model = Model(0.01)
Q_start = np.array([[0.9, 0.0], [0.0, 0.201843]])
P_start = np.array([[0.0829942, 0.0], [0.0, 0.0884979]])

x0 = State(Q_start, P_start)

step_num = 100

states = [x0]

for i in range(step_num):
    xi, _M = model.step(states[-1])
    print("State %d: %s" % (i, xi))
    states.append(xi)

sun_pos = np.array([ xi.q_s() for xi in states ])
earth_pos = np.array([ xi.q_e() for xi in states ])
moon_pos = np.array([ xi.q_m() for xi in states ])
#moon_pos_rel_to_planet = np.array([ xi.get_Q()[1] for xi in states ])

print("Moon pos: ", moon_pos)
print("Earth pos: ", earth_pos)
print("Sun pos: ", sun_pos)

plt.scatter(sun_pos[:,0], sun_pos[:,1], label="Sun")
plt.scatter(earth_pos[:,0], earth_pos[:,1], label="Earth")
plt.scatter(moon_pos[:,0], moon_pos[:,1], label="Moon")
plt.legend()
plt.show()



