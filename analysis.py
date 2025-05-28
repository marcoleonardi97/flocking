# The error occurs because in the `Flock` class initialization, obstacles were set as an integer,
# and later an attempt was made to iterate over it. Below is the fixed code:

# Flocking simulation 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Boid:
    def __init__(self, perception_radius=1):
        self.position = np.random.random(2) * 100  # Start position
        self.velocity = np.random.uniform(-1, 1, 2) * 2    # Start velocity
        self.acceleration = np.zeros(2)
        self.perception_radius = perception_radius

    def align(self, others, max_force=0.01):
        avg = np.zeros(2) # Average velocity of neighbours
        neighbours = 0    # Initial neigbhbours
        for other in others: 
            if other is not self:
                distance = np.linalg.norm(self.position - other.position) # Check distance to other animals
                if distance < self.perception_radius: # Check if within perception radius
                    avg += other.velocity        # Add their velocity
                    neighbours += 1             # Count neighbours

        if neighbours > 0:              # If animal has neighbours its velocity is affected
            avg /= neighbours           # Average velocity of neighbours
            steer = avg - self.velocity # Difference with own velocity

            # Limit the steering force
            steer_mag = np.linalg.norm(steer)
            if steer_mag > max_force:
                steer = steer / steer_mag * max_force
            self.acceleration += steer 


    def cohesion(self, others, max_force=0.01):
        avg = np.zeros(2)
        neighbours = 0
        for other in others:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < self.perception_radius * 0.5:
                    avg += other.position
                    neighbours += 1

        if neighbours > 0:
            avg /= neighbours
            steer = avg - self.position
            # Limit the steering force
            steer_mag = np.linalg.norm(steer)
            steer -= self.velocity #not sure if i need this
            if steer_mag > max_force:
                steer = steer / steer_mag * max_force
            self.acceleration += steer 


    def separation(self, others, max_force=0.01):
        avg = np.zeros(2)
        neighbours = 0
        for other in others:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < self.perception_radius:
                    diff = self.position - other.position
                    diff /= distance
                    avg += diff
                    neighbours += 1

        if neighbours > 0:
            avg /= neighbours
            steer = avg
            # Limit the steering force
            steer_mag = np.linalg.norm(steer)
            #steer -= self.velocity #not sure if i need this
            if steer_mag > max_force:
                steer = steer / steer_mag * max_force
            self.acceleration += steer 

    def calc_order_param(self, others, max_force=0.01):
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


    def limit_velocity(self, max_speed=1):
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed

class Flock:
    def __init__(self, n=100, boxsize=100, obstacles=0, wind = None, perception_radius=1, free_will = None):
        self.flock = [Boid(perception_radius=perception_radius) for _ in range(n)]
        self.vel_angles = [np.arctan2(boid.velocity[1] , boid.velocity[0]) for boid in self.flock]
        self.box_size = 100
        self.frame = 0
        self.perception_radius = perception_radius
        self.wind_func = wind if wind is not None else None
        self.free_will = free_will
        self.psi_tot = 0

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
        I think here we update the position and acceleration of each boid one per one
        but maybe we should first update the acceleration of each boid and after that
        calculate the new positions of each boid.
        Won't make a big difference but I think that is more correct
        """

        self.psi_tot = 0

        for boid in self.flock:
            boid.acceleration = 0
            boid.position[0] += boid.velocity[0] 
            boid.position[1] += boid.velocity[1]
            boid.position = np.mod(boid.position, self.box_size) # Periodic boundary conditions

            self.psi_tot += boid.calc_order_param(self.flock)


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

            boid.align(self.flock, max_force=0.02) 
            boid.cohesion(self.flock, max_force=0.02)
            boid.separation(self.flock, max_force =0.01)

            boid.velocity += boid.acceleration * (1-self.free_will) + self.free_will * np.random.uniform(-1, 1, 2)

            if self.wind_func is not None:
                boid.velocity += self.wind_func(boid.position)

            """
            if self.frame % 20 == 0:
                random_angle = np.random.uniform(-self.free_will * np.pi, self.free_will * np.pi)
                boid.velocity = self.rotate_vector(boid.velocity, random_angle)
                free_will_change = np.random.uniform(-0.5, 1) * self.free_will
                #boid.velocity += free_will_change
                boid.velocity += boid.velocity * free_will_change
            """

            boid.limit_velocity(max_speed=2.5)

            # Update the angles of the velocities
            self.vel_angles = [np.arctan2(boid.velocity[1] , boid.velocity[0]) for boid in self.flock]

            self.frame += 1
        
        self.psi_tot /= len(self.flock)

    def get_positions(self):
        return np.array([boid.position for boid in self.flock])

    def get_velocities(self):
        return np.array([boid.velocity for boid in self.flock])


class Circ_object:

    def __init__(self, x=None, y=None, radius=None):
        if x == None and y == None and radius == None:
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            radius = np.random.uniform(1, 10)
        self.x = x
        self.y = y
        self.radius = radius
        self.features = np.array([x, y, radius])

box_size = 100

num_obstacles = 0             # Number of circular obstacle
free_will = 1        # Strength of free will

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


