# Flocking simulation 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Boid:
    """
    Class for individual boid objects.
    """
    def __init__(self, perception_radius=1, perception_fov = 2*np.pi, max_speed=2.5):	
        """
        Initialize the boid.

        Args:
            perception_radius (float): The radius of the perception sphere
            perception_fov (float): The field of view of the perception sphere
            max_speed (float): The maximum speed of the boid
        """
        self.position = np.random.random(2) * 100  
        self.velocity = np.random.uniform(-1, 1, 2) * 2    
        self.acceleration = np.zeros(2)
        self.perception_radius = perception_radius
        self.perception_fov = perception_fov
        self.max_speed = max_speed


    def in_fov(self,other): 
        """
        Check if the other boid is in the field of view of the current boid.

        Args:
            other (Boid): The other boid

        Returns:
            bool: True if the other boid is in the field of view of the current boid
        """
        offset = other.position - self.position
        distance = np.linalg.norm(offset)
        if (distance > self.perception_radius):
            return False

        own_direction = self.velocity / np.linalg.norm(self.velocity)
        direction_to_other = offset / np.linalg.norm(offset)
        angle = (np.arccos(np.dot(own_direction, direction_to_other))) % (np.pi)

        return angle < self.perception_fov / 2

    def align(self, others, max_force=0.01):
        """
        Align the velocity of the boid with the average velocity of its neighbours.

        Args:
            others (list): A list of other boids
            max_force (float): The maximum force to apply
        """
        avg = np.zeros(2) # Average velocity of neighbours
        neighbours = 0    # Initial neigbhbours
        for other in others: 
            if other is not self:
                distance = np.linalg.norm(self.position - other.position) # Check distance to other animals
                if self.in_fov(other) and distance < self.perception_radius: # Check if within perception radius
                    avg += other.velocity        # Add their velocity
                    neighbours += 1             # Count neighbours

        if neighbours > 0:              # If animal has neighbours its velocity is affected
            desired = avg / neighbours          
            desired = desired/np.linalg.norm(desired) * self.max_speed #Average velocity normalized
            steer = desired - self.velocity # Difference with own velocity
            # Limit the steering force
            steer_mag = np.linalg.norm(steer + 1e-5 )
            if steer_mag > max_force:
                steer = steer / steer_mag * max_force
            self.acceleration += steer


    def cohesion(self, others, max_force=0.01):
        """
        Cohere the position of the boid with the average position of its neighbours.

        Args:
            others (list): A list of other boids
            max_force (float): The maximum force to apply
        """
        avg = np.zeros(2)
        neighbours = 0
        for other in others:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if self.in_fov(other) and distance < self.perception_radius * 0.5:
                    avg += other.position
                    neighbours += 1

        if neighbours > 0:
            desired = avg / neighbours          
            #desired = desired/np.linalg.norm(desired) * self.max_speed #Average velocity normalized
            steer = desired - self.position
            # Limit the steering force
            steer_mag = np.linalg.norm(steer + 1e-5)
            if steer_mag > max_force:
                steer = steer / steer_mag * max_force
            self.acceleration += steer 


    def separation(self, others, max_force=0.01):
        """
        Separate the position of the boid with the average position of its neighbours.

        Args:
            others (list): A list of other boids
            max_force (float): The maximum force to apply
        """
        avg = np.zeros(2)
        neighbours = 0
        for other in others:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if self.in_fov(other) and distance < self.perception_radius:
                    diff = self.position - other.position
                    diff /= distance
                    avg += diff
                    neighbours += 1

        if neighbours > 0:
            avg /= neighbours
            steer = avg
            # Limit the steering force
            steer_mag = np.linalg.norm(steer + 1e-5)
            #steer -= self.velocity #not sure if i need this
            if steer_mag > max_force:
                steer = steer / steer_mag * max_force
            self.acceleration += steer 

    def avoid_predators(self, predators, avoidance_radius=8, max_force=0.3):
        """
        Avoid nearby predators.

        Args:
            predators (list): A list of predator boids
            avoidance_radius (float): The radius of the perception sphere
            max_force (float): The maximum force to apply
        """
        total_avoidance = np.zeros(2)
        predators_detected = 0
        
        for predator in predators:
            distance = np.linalg.norm(self.position - predator.position)
            if distance < avoidance_radius:  # Don't require field of view - fear response is 360°
                # Calculate escape direction (away from predator)
                escape_dir = self.position - predator.position
                if distance > 0:
                    escape_dir = escape_dir / distance
                    # Weight by inverse distance - closer predators cause stronger response
                    weight = avoidance_radius / (distance + 0.1)
                    total_avoidance += escape_dir * weight
                    predators_detected += 1

        if predators_detected > 0:
            # Normalize and apply strong avoidance force
            desired_velocity = total_avoidance / predators_detected
            desired_velocity = desired_velocity / (np.linalg.norm(desired_velocity) + 1e-5) * self.max_speed
            steer = desired_velocity - self.velocity
            
            # Limit the steering force
            steer_mag = np.linalg.norm(steer)
            if steer_mag > max_force:
                steer = steer / steer_mag * max_force
            self.acceleration += steer

    def calc_order_param(self, others):
        """
        Calculate equilibrium parameter for the indiviudal boid

        Args:
            others (list): other boids
        """
        avg = np.zeros(2) # Average velocity of neighbours
        neighbours = 0    # Initial neigbhbours
        for other in others: 
            if other is not self:
                distance = np.linalg.norm(self.position - other.position) # Check distance to other animals
                if distance < self.perception_radius: # Check if within perception radius
                    avg += other.velocity/np.linalg.norm(other.velocity)        # Add their velocity
                    neighbours += 1             # Count neighbours

        if neighbours > 0:              
            psi = np.linalg.norm(avg) / neighbours  
            return psi
        else:
            return 0


    def limit_velocity(self, max_speed):
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed
            
    def get_predator_stress(self, predators, stress_radius=10):
        """
        Calculate stress level based on nearby predators
        """
        stress = 0
        for predator in predators:
            distance = np.linalg.norm(self.position - predator.position)
            if distance < stress_radius:
                stress += (stress_radius - distance) / stress_radius
        return min(1.0, stress)  # Cap at 1.0

class Flock:
    """
    This class governs the behaviour of the flock of boids, and functions as our simulation interface
    """
    def __init__(self, n=100, boxsize=100, num_predators = 0, obstacles=0, wind = None, 
    perception_radius=1, free_will = 0, perception_fov = 2*np.pi,
    align_strength=0.02, cohesion_strength=0.02, separation_strength=0.01):
        """
        Initialise the flock

        Args:

        n (int): number of boids
        boxsize (int): simulation boxsize
        num_predators (int): number of predator boids
        obstacles (int): number of obstacle objects
        wind (function): wind function
        perception_radius (float): perception radius
        free_will (float): free will
        perception_fov (float): perception field of view
        align_strength (float): alignment strength
        cohesion_strength (float): cohesion strength
        separation_strength (float): separation strength
        """
        
        self.flock = [Boid(perception_radius=perception_radius, perception_fov = perception_fov) for _ in range(n)]
        self.vel_angles = [np.arctan2(boid.velocity[1] , boid.velocity[0]) for boid in self.flock]
        self.box_size = 100
        self.frame = 0
        self.n = n
        self.perception_radius = perception_radius
        self.perception_fov = perception_fov
        self.wind_func = wind if wind is not None else None
        self.free_will = free_will
        self.psi_tot = 0
        self.align_strength = align_strength
        self.cohesion_strength = cohesion_strength
        self.separation_strength = separation_strength
        self.predators = [Predator() for _ in range(num_predators)]
        self.num_predators = num_predators


        # If obstacles is an integer, create that many obstacle objects
        if isinstance(obstacles, int):
            self.obstacles = [Circ_object() for _ in range(obstacles)]
        else:
            self.obstacles = obstacles

    def rotate_vector(self, v, angle_rad):
    # Rotate a 2D vector `v` by `angle_rad` radians
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]])

        return rotation_matrix @ v

    
    def update(self):
        """
        Update the position and velocity of all boids and predators each simulation frame
        """
        self.psi_tot = 0

        # Update all boids first
        for boid in self.flock:
            boid.acceleration = np.zeros(2)
            boid.position[0] += boid.velocity[0] 
            boid.position[1] += boid.velocity[1]
            boid.position = np.mod(boid.position, self.box_size) # Periodic boundary conditions

            self.psi_tot += boid.calc_order_param(self.flock)

            # Handle obstacles
            for obj in self.obstacles:
                ox = obj.x
                oy = obj.y
                radius = obj.radius

                # Check if boid is inside obstacle
                to_center = boid.position - np.array([ox, oy])
                dist = np.linalg.norm(to_center)

                if dist < radius:
                    # Normalize the vector from obstacle center to boid
                    normal = to_center / dist

                    # Reflect velocity across the normal
                    boid.velocity = boid.velocity - 2 * np.dot(boid.velocity, normal) * normal

                    # Move boid just outside the obstacle
                    boid.position = np.array([ox, oy]) + normal * radius

            # Apply flocking behaviors (reduce influence when predators nearby)
            predator_stress = boid.get_predator_stress(self.predators)
            stress_factor = max(0.1, 1.0 - predator_stress)  # Reduce flocking when stressed
            
            boid.align(self.flock, max_force = self.align_strength * stress_factor) 
            boid.cohesion(self.flock, max_force = self.cohesion_strength * stress_factor)
            boid.separation(self.flock, max_force = self.separation_strength)
            
            # Apply wind
            if self.wind_func is not None:
                boid.velocity += self.wind_func(boid.position)

            # Avoid predators (high priority)
            boid.avoid_predators(self.predators)

            # Apply free will and acceleration
            boid.velocity += boid.acceleration * (1-self.free_will) + self.free_will * np.random.uniform(-1, 1, 2)
            boid.limit_velocity(max_speed=2.5)

        # Update predators separately (only once per frame)
        if self.num_predators > 0:
            for predator in self.predators:
                predator.acceleration = np.zeros(2)
                predator.chase(self.flock)
                predator.velocity += predator.acceleration
                predator.position[0] += predator.velocity[0]
                predator.position[1] += predator.velocity[1]
                predator.position = np.mod(predator.position, self.box_size)
                predator.limit_velocity(max_speed=2.0)  # Slightly slower than boids

        self.frame += 1
        
        # Update the angles of the velocities
        self.vel_angles = [np.arctan2(boid.velocity[1] , boid.velocity[0]) for boid in self.flock]
        self.psi_tot /= len(self.flock)

    def analyse(self, parameter):
        """
        Plots alignment equilibrium (psi) against the given parameter.

        Args:
        paramter (dictionary): name and list of values for your desired parameter
        
        Accepted values: 'free will', 'perception radius', 'predators'

        Example use: {'free will': np.linspace(0, 1, 10)}
        """
        curr_psi = 0
        psis = []
        for value in list(parameter.values())[0]:
            if 'free will' in parameter:
                self.free_will = value
            elif 'perception radius' in parameter:
                self.perception_radius = value
            elif 'predators' in parameter:
                self.num_predators = value
                self.predators = [Predator() for _ in range(self.num_predators)]
            else:
                print('Accepted values: \'free will\', \'perception radius\'')
                return 
            for i in range(200):
                self.update()
                if i >= 150:
                    curr_psi += self.psi_tot
            curr_psi /= 50
            psis.append(curr_psi)
            rgb =  1-value/max(list(parameter.values())[0])
            plt.hist(self.vel_angles, label=f"{list(parameter.keys())[0]} = {value:.2f}", 
            color=(rgb, rgb * 0.5, rgb),
            alpha=0.3, bins=20)

        plt.title(f"Velocity Angle Distribution for {list(parameter.keys())[0]}")
        plt.xlabel('Velocity angle')
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
        plt.ylim(0, 50)
        plt.ylabel('Number of boids')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"histogram_{list(parameter.keys())[0]}.png")
        plt.show()
                    
        plt.scatter(list(parameter.values())[0], psis)
        plt.plot(list(parameter.values())[0], psis, alpha=0.7)
        plt.title(f"Equilibrium Alignment vs. {list(parameter.keys())[0]}")
        plt.xlabel(f"{list(parameter.keys())[0]}")
        plt.ylabel(r"$\psi$")
        plt.ylim(0,1)
        plt.savefig(f"analysis_{list(parameter.keys())[0]}.png")
        plt.show()
    
    
    def animate(self, n_frames=200, name="flocking_animation.gif"):
		
        fig, ax = plt.subplots()
        fig.set_size_inches(8,6)
        for obj in self.obstacles:
            ox = obj.x
            oy = obj.y
            r = obj.radius
            circle = plt.Circle((ox, oy), r, color='black', alpha=0.3)
            ax.add_patch(circle)
        
        plt.title("Flocking Simulation")
        plt.xlabel("x")
        plt.ylabel("y")
        positions, pred_positions = self.get_positions()
        velocities, pred_velocities = self.get_velocities()
        scat = ax.quiver(positions[:, 0], positions[:, 1],
                         velocities[:, 0], velocities[:, 1], 
                         angles="xy", scale_units="xy", scale=1/2)
        
        if self.num_predators > 0:
            pred_scat = ax.quiver(pred_positions[:, 0], pred_positions[:, 1],
                                  pred_velocities[:, 0], pred_velocities[:, 1],
                                  color='red', angles="xy", scale_units="xy", scale=1/2)

        textstr = '\n'.join((
        f'Number of boids = {self.n}',
        f'Wind = {self.wind_func}',
        f'Alignment strength = {self.align_strength}',
        f'Cohesion strength = {self.cohesion_strength}',
        f'Separation strength = {self.separation_strength}',
        f'Free will = {self.free_will}',
        f'Number of predators = {self.num_predators}', 
        f'Field of view = {self.perception_fov/np.pi}*$\pi$ rad',))
    
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        
        # place a text box in upper left in axes coords
        ax.text(1.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.tight_layout()

        # Finally using a normal animation function instead of printing a million pngs 
        def update(frame):
            self.update()
            positions, pred_positions = self.get_positions()
            velocities, pred_velocities = self.get_velocities()
            scat.set_offsets(positions)
            scat.set_UVC(velocities[:, 0], velocities[:, 1])
            if self.num_predators > 0:
                pred_scat.set_offsets(pred_positions)
                pred_scat.set_UVC(pred_velocities[:, 0], pred_velocities[:, 1])
            return scat,
        
        ani = FuncAnimation(fig, update, frames=200, blit=True, interval=50)
        ani.save(name)
        plt.show()


    def get_positions(self):
        return np.array([boid.position for boid in self.flock]), np.array([predator.position for predator in self.predators])

    def get_velocities(self):
        return np.array([boid.velocity for boid in self.flock]), np.array([predator.velocity for predator in self.predators])

class Predator(Boid):
    """
    A predator that chases boids
    """
    def __init__(self):
        super().__init__()
        # Remove the velocity multiplier that was causing speed issues
        self.velocity = np.random.uniform(-1, 1, 2) * 1.5  # Slightly faster than boids but not too much
        self.position = np.random.random(2) * 100
        
    def chase(self, boids, perception_radius=15, max_force=0.03):  # Reduced max_force
        """
        Steer the predator towards the closest boid

        Args:
        boids (list): list of boids
        perception_radius (float): radius of perception
        max_force (float): maximum force of the predator
        """
        closest_prey = None
        min_distance = np.inf

        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if distance < perception_radius and distance < min_distance:
                min_distance = distance
                closest_prey = boid

        if closest_prey:
            direction = closest_prey.position - self.position
            steer = direction / np.linalg.norm(direction + 1e-5) * max_force
            self.acceleration += steer

    def limit_velocity(self, max_speed):
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed

class Circ_object:
    """
    A circular obstacle
    """
    def __init__(self, x=None, y=None, radius=None):
        if x == None and y == None and radius == None:
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            radius = np.random.uniform(1, 10)
        self.x = x
        self.y = y
        self.radius = radius
        self.features = np.array([x, y, radius])

def uniform_wind(pos):
    windpower = np.array([0,1]) # For uniform wind
    return windpower

def wavy_wind(pos):
    return np.array([0.5 * np.sin(pos[1] / 10),
            0.5 * np.cos(pos[0] / 10)])

def vortex_wind(pos):
    center = np.array([50, 50])
    offset = pos - center
    perp = np.array([-offset[1], offset[0]])  # Perpendicular to offset (counterclockwise)
    strength = 0.05
    return strength * perp / (np.linalg.norm(offset) + 1e-5)



