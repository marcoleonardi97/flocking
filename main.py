from analysis import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Adjust simulation parameters
box_size = 100
num_obstacles = 0            
free_will = 0                
wind_func = None
fov = 2*np.pi
perc_rad = 20
align_strength = 0.02
cohesion_strength = 0.02
separation_strength = 0.01
num_boids = 200
num_predators = 0

# Initialise simulation
sim = Flock(n=num_boids, boxsize=box_size, num_predators=num_predators,obstacles=num_obstacles, wind=wind_func, \
 perception_radius=perc_rad, free_will=free_will, perception_fov=fov, \
 align_strength=align_strength, cohesion_strength=cohesion_strength, separation_strength=separation_strength)



# Animate the simulation
#sim.animate(n_frames=200, name="flocking_simulation_200.gif")



# Analyse equilibrium based on different parameters (this function accepts "free will", "perception radius", "predators")
# This function sometimes does not update the simulation properly, since it does not reset it every loop step.
# It is much more convenient, but for more accurate result use the manual loop below

#sim_100 = Flock(n=100, boxsize=100, obstacles=0, wind=None)
#sim_100.analyse({'free will': range(0,10,1)})



# Other values can be analyse manually  ---- Add value in the if statement and change it in the init

fws = np.linspace(0, 1, 2)
pers = np.linspace(0,50,20)
predators = range(0,20,2)
psis = []

variable = 'Free Will' # Free Will or Perception Radius

if variable == 'Free Will':
    array = fws
elif variable == 'Perception Radius':
    array = pers
elif variable == 'Predators':
	array = predators

for a in array:
    if variable == 'Free Will':
        sim = Flock(n=num_boids, boxsize=100,num_predators = num_predators, obstacles=0, wind=None, perception_radius=20, free_will=a,
        perception_fov=fov, align_strength=align_strength, cohesion_strength=cohesion_strength, separation_strength=separation_strength)
    elif variable == 'Perception Radius':
        sim = Flock(n=num_boids, boxsize=100,num_predators = num_predators, obstacles=0, wind=None, perception_radius=a, free_will=0,
        perception_fov=fov, align_strength=align_strength, cohesion_strength=cohesion_strength, separation_strength=separation_strength)
    elif variable == 'Predators':
        sim = Flock(n=num_boids, boxsize=100,num_predators = a, obstacles=0, wind=None, perception_radius=20, free_will=0,
        perception_fov=fov, align_strength=align_strength, cohesion_strength=cohesion_strength, separation_strength=separation_strength)


    curr_psi = 0
    for i in range(200):
        sim.update()
        if i >= 150:
            curr_psi += sim.psi_tot
    curr_psi /= 50
    psis.append(curr_psi)

    fig, ax = plt.subplots()
    plt.title("Flocking Simulation last frame")
    plt.xlabel("x")
    plt.ylabel("y")
    positions, pred_positions = sim.get_positions()
    velocities, pred_velocities = sim.get_velocities()
    scat = ax.quiver(positions[:, 0], positions[:, 1],
                    velocities[:, 0], velocities[:, 1], 
                    angles="xy", scale_units="xy", scale=1/2)

    if num_predators > 0:
        pred_scat = ax.quiver(pred_positions[:, 0], pred_positions[:, 1],
                            pred_velocities[:, 0], pred_velocities[:, 1],
                            color='red', angles="xy", scale_units="xy", scale=1/2)

    textstr = '\n'.join((
    f'Number of boids = {num_boids}',
    f'Wind = {wind_func}',
    f'Alignment strength = {align_strength}',
    f'Cohesion strength = {cohesion_strength}',
    f'Separation strength = {separation_strength}',
    f'Free will = {free_will}',
    f'Number of predators = {num_predators}', 
    f'Field of view = {fov/np.pi}*$\pi$ rad',))

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.show()

    angles = np.array(sim.vel_angles)
    num_bins = 16
    counts, bin_edges = np.histogram(angles, bins=num_bins, range=(-np.pi, np.pi))
    bin_width = bin_edges[1] - bin_edges[0]

    # Normalize counts to make area = 1 (i.e., probability density)
    probs = counts / (len(angles))

    # Plot manually with bar()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, probs, width=bin_width, alpha=0.3, label=f"{variable} = {a}")

    plt.title(f"Velocity Angle Distribution for {variable} = {round(a,2)}")
    plt.xlabel('Velocity angle')
    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    plt.ylim(0, 1)
    plt.ylabel('Percentage of boids')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.savefig("monit_pr.png")
    plt.show()
        
np.save(f"psis_{variable}.npy", psis) 

plt.scatter(pers, psis)
plt.plot(pers, psis, alpha=0.7)
plt.title(f"Equilibrium Alignment vs. {variable}")
plt.xlabel(f"{variable}")
plt.ylabel(r"$\psi$")
plt.ylim(0,1)

textstr = '\n'.join((
    f'Number of boids = {num_boids}',
    f'Wind = {wind_func}',
    f'Alignment strength = {align_strength}',
    f'Cohesion strength = {cohesion_strength}',
    f'Separation strength = {separation_strength}',
    f'Free will = {free_will}',
    f'Number of predators = {num_predators}', 
    f'Field of view = {fov/np.pi}*$\pi$ rad',))

props = dict(boxstyle='square', facecolor='white', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(f"psis_{variable}.png")
plt.show()
