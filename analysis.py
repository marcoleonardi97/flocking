import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
from collections import defaultdict
import matplotlib.patches as patches


class Boid:
    def __init__(self, position=None, velocity=None):
        self.position = position if position is not None else np.random.random(2) * 100
        self.velocity = velocity if velocity is not None else np.random.random(2) * 2 - 1
        self.acceleration = np.zeros(2)
        self.id = np.random.randint(0, 1000000)  # Unique ID for tracking

    def align(self, others, perception_radius=15, max_force=0.02):
        """Steer towards the average heading of neighbors"""
        avg_velocity = np.zeros(2)
        count = 0
        
        for other in others:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < perception_radius:
                    avg_velocity += other.velocity
                    count += 1

        if count > 0:
            avg_velocity /= count
            if np.linalg.norm(avg_velocity) > 0:
                desired = avg_velocity / np.linalg.norm(avg_velocity) * 2.5
                steer = desired - self.velocity
                steer = self._limit_force(steer, max_force)
                return steer
        return np.zeros(2)

    def cohesion(self, others, perception_radius=15, max_force=0.02):
        """Steer towards the average position of neighbors"""
        avg_position = np.zeros(2)
        count = 0
        
        for other in others:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if distance < perception_radius:
                    avg_position += other.position
                    count += 1

        if count > 0:
            avg_position /= count
            desired = avg_position - self.position
            if np.linalg.norm(desired) > 0:
                desired = desired / np.linalg.norm(desired) * 2.5
                steer = desired - self.velocity
                steer = self._limit_force(steer, max_force)
                return steer
        return np.zeros(2)

    def separation(self, others, perception_radius=5, max_force=0.03):
        """Steer away from neighbors to avoid crowding"""
        steer = np.zeros(2)
        count = 0
        
        for other in others:
            if other is not self:
                distance = np.linalg.norm(self.position - other.position)
                if 0 < distance < perception_radius:
                    diff = self.position - other.position
                    diff = diff / distance
                    steer += diff
                    count += 1

        if count > 0:
            steer /= count
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * 2.5
                steer = steer - self.velocity
                steer = self._limit_force(steer, max_force)
                return steer
        return np.zeros(2)

    def avoid_predators(self, predators, perception_radius=20, max_force=0.1):
        """Flee from nearby predators"""
        steer = np.zeros(2)
        count = 0
        
        for predator in predators:
            distance = np.linalg.norm(self.position - predator.position)
            if distance < perception_radius:
                diff = self.position - predator.position
                if distance > 0:
                    diff = diff / (distance * distance)
                    steer += diff
                    count += 1

        if count > 0:
            steer /= count
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * 3.0
                steer = steer - self.velocity
                steer = self._limit_force(steer, max_force)
                return steer
        return np.zeros(2)

    def _limit_force(self, force, max_force):
        """Limit the magnitude of a force vector"""
        force_mag = np.linalg.norm(force)
        if force_mag > max_force:
            return force / force_mag * max_force
        return force

    def limit_velocity(self, max_speed=2.5):
        """Limit the velocity to max_speed"""
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = self.velocity / speed * max_speed


class Predator(Boid):
    def __init__(self, position=None):
        super().__init__(position)
        self.velocity *= 1.2
        
    def chase(self, boids, perception_radius=25, max_force=0.08):
        """Chase the closest boid within perception radius"""
        closest_prey = None
        min_distance = np.inf

        for boid in boids:
            distance = np.linalg.norm(self.position - boid.position)
            if distance < perception_radius and distance < min_distance:
                min_distance = distance
                closest_prey = boid

        if closest_prey:
            desired = closest_prey.position - self.position
            if np.linalg.norm(desired) > 0:
                desired = desired / np.linalg.norm(desired) * 3.0
                steer = desired - self.velocity
                steer = self._limit_force(steer, max_force)
                return steer
        return np.zeros(2)


class CircularObstacle:
    def __init__(self, x=None, y=None, radius=None, box_size=100):
        self.x = x if x is not None else np.random.uniform(10, box_size-10)
        self.y = y if y is not None else np.random.uniform(10, box_size-10)
        self.radius = radius if radius is not None else np.random.uniform(3, 8)


class EquilibriumTracker:
    """Tracks system equilibrium using various metrics"""
    
    def __init__(self, max_history=500):
        self.max_history = max_history
        self.history = {
            'positions': [],
            'velocities': [],
            'speeds': [],
            'alignment': [],
            'cohesion': [],
            'nearest_neighbor_distances': [],
            'cluster_sizes': [],
            'num_clusters': [],
            'largest_cluster_fraction': []
        }
        
    def update(self, flock):
        """Update tracking data with current flock state"""
        positions = np.array([boid.position for boid in flock.flock])
        velocities = np.array([boid.velocity for boid in flock.flock])
        
        # Store raw data (keep only recent history)
        self.history['positions'].append(positions.copy())
        self.history['velocities'].append(velocities.copy())
        
        if len(self.history['positions']) > self.max_history:
            self.history['positions'].pop(0)
            self.history['velocities'].pop(0)
        
        # Calculate derived metrics
        speeds = np.linalg.norm(velocities, axis=1)
        self.history['speeds'].append(np.mean(speeds))
        
        # Alignment: average dot product of normalized velocities
        normalized_vels = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-8)
        alignment = np.mean([np.dot(normalized_vels[i], normalized_vels[j]) 
                           for i in range(len(normalized_vels)) 
                           for j in range(i+1, len(normalized_vels))])
        self.history['alignment'].append(alignment)
        
        # Cohesion: inverse of average distance from center of mass
        center_of_mass = np.mean(positions, axis=0)
        distances_to_com = np.linalg.norm(positions - center_of_mass, axis=1)
        cohesion = 1.0 / (np.mean(distances_to_com) + 1e-8)
        self.history['cohesion'].append(cohesion)
        
        # Nearest neighbor distances
        if len(positions) > 1:
            dist_matrix = cdist(positions, positions)
            np.fill_diagonal(dist_matrix, np.inf)
            nn_distances = np.min(dist_matrix, axis=1)
            self.history['nearest_neighbor_distances'].append(np.mean(nn_distances))
        
        # Friends-of-friends clustering
        clusters = self.friends_of_friends_clustering(positions, threshold=8.0)
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        self.history['cluster_sizes'].append(cluster_sizes)
        self.history['num_clusters'].append(len(clusters))
        
        if cluster_sizes:
            largest_cluster_fraction = max(cluster_sizes) / len(positions)
            self.history['largest_cluster_fraction'].append(largest_cluster_fraction)
        else:
            self.history['largest_cluster_fraction'].append(0)
        
        # Trim history
        for key in ['speeds', 'alignment', 'cohesion', 'nearest_neighbor_distances', 
                   'cluster_sizes', 'num_clusters', 'largest_cluster_fraction']:
            if len(self.history[key]) > self.max_history:
                self.history[key].pop(0)
    
    def friends_of_friends_clustering(self, positions, threshold=8.0):
        """Friends-of-friends clustering algorithm"""
        n_boids = len(positions)
        visited = [False] * n_boids
        clusters = []
        
        def dfs(boid_idx, current_cluster):
            visited[boid_idx] = True
            current_cluster.append(boid_idx)
            
            for other_idx in range(n_boids):
                if not visited[other_idx]:
                    distance = np.linalg.norm(positions[boid_idx] - positions[other_idx])
                    if distance <= threshold:
                        dfs(other_idx, current_cluster)
        
        for i in range(n_boids):
            if not visited[i]:
                cluster = []
                dfs(i, cluster)
                clusters.append(cluster)
        
        return clusters
    
    def compute_autocorrelation(self, data, max_lag=50):
        """Compute autocorrelation function for a time series"""
        if len(data) < max_lag * 2:
            return np.array([]), np.array([])
        
        data = np.array(data)
        n = len(data)
        data = data - np.mean(data)  # Remove mean
        
        # Compute autocorrelation using FFT (more efficient)
        padded = np.zeros(2 * n)
        padded[:n] = data
        
        fft_data = np.fft.fft(padded)
        autocorr = np.fft.ifft(fft_data * np.conj(fft_data)).real
        autocorr = autocorr[:max_lag] / autocorr[0]  # Normalize
        
        lags = np.arange(max_lag)
        return lags, autocorr
    
    def get_equilibrium_metrics(self):
        """Calculate equilibrium-related metrics"""
        if len(self.history['speeds']) < 50:
            return {}
        
        metrics = {}
        
        # Autocorrelation times (time to decay to 1/e)
        for key in ['speeds', 'alignment', 'cohesion']:
            if len(self.history[key]) > 100:
                lags, autocorr = self.compute_autocorrelation(self.history[key])
                if len(autocorr) > 0:
                    # Find where autocorrelation drops below 1/e ≈ 0.368
                    decay_idx = np.where(autocorr < 1/np.e)[0]
                    if len(decay_idx) > 0:
                        metrics[f'{key}_autocorr_time'] = lags[decay_idx[0]]
                    else:
                        metrics[f'{key}_autocorr_time'] = len(lags)
        
        # Stability metrics (coefficient of variation)
        recent_window = 50
        for key in ['speeds', 'alignment', 'cohesion', 'num_clusters']:
            if len(self.history[key]) >= recent_window:
                recent_data = self.history[key][-recent_window:]
                mean_val = np.mean(recent_data)
                std_val = np.std(recent_data)
                if mean_val > 0:
                    metrics[f'{key}_stability'] = std_val / mean_val
        
        return metrics


class Flock:
    def __init__(self, n_boids=50, n_predators=0, box_size=100, obstacles=0, 
                 wind_func=None, free_will=None):
        self.flock = [Boid() for _ in range(n_boids)]
        self.predators = [Predator() for _ in range(n_predators)]
        self.box_size = box_size
        self.wind_func = wind_func
        self.free_will = free_will
        
        # Initialize equilibrium tracker
        self.eq_tracker = EquilibriumTracker()

        # Create obstacles
        if isinstance(obstacles, int):
            self.obstacles = [CircularObstacle(box_size=box_size) for _ in range(obstacles)]
        else:
            self.obstacles = obstacles if obstacles else []

    def _rotate_vector(self, v, angle_rad):
        """Rotate a 2D vector by angle_rad radians"""
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([cos_a * v[0] - sin_a * v[1],
                        sin_a * v[0] + cos_a * v[1]])

    def _handle_obstacle_collision(self, boid, obstacle):
        """Handle collision between boid and circular obstacle"""
        to_center = boid.position - np.array([obstacle.x, obstacle.y])
        dist = np.linalg.norm(to_center)

        if dist < obstacle.radius:
            if dist > 0:
                normal = to_center / dist
            else:
                normal = np.random.random(2) - 0.5
                normal = normal / np.linalg.norm(normal)
            
            boid.position = np.array([obstacle.x, obstacle.y]) + normal * (obstacle.radius + 1)
            boid.velocity = boid.velocity - 2 * np.dot(boid.velocity, normal) * normal

    def update(self):
        """Update all boids and predators"""
        # First pass: calculate accelerations
        for boid in self.flock:
            boid.acceleration = np.zeros(2)
            
            # Flocking behaviors
            align_force = boid.align(self.flock, self.box_size * 0.15, 0.02)
            cohesion_force = boid.cohesion(self.flock, self.box_size * 0.15, 0.02)
            separation_force = boid.separation(self.flock, self.box_size * 0.05, 0.03)
            
            # Predator avoidance
            if self.predators:
                avoid_force = boid.avoid_predators(self.predators, self.box_size * 0.2, 0.1)
                boid.acceleration += avoid_force

            # Apply forces
            boid.acceleration += align_force + cohesion_force + separation_force
            
            # Wind effect
            if self.wind_func:
                boid.acceleration += self.wind_func(boid.position)

        # Update predators
        for predator in self.predators:
            predator.acceleration = np.zeros(2)
            chase_force = predator.chase(self.flock)
            predator.acceleration += chase_force

        # Second pass: update velocities and positions
        for boid in self.flock:
            boid.velocity += boid.acceleration
            
            # Add free will (random steering)
            if self.free_will:
                random_angle = np.random.uniform(-self.free_will, self.free_will)
                boid.velocity = self._rotate_vector(boid.velocity, random_angle)
            
            boid.limit_velocity(max_speed=2.5)
            boid.position += boid.velocity
            
            # Handle boundaries (wrap around)
            boid.position = np.mod(boid.position, self.box_size)
            
            # Handle obstacle collisions
            for obstacle in self.obstacles:
                self._handle_obstacle_collision(boid, obstacle)

        for predator in self.predators:
            predator.velocity += predator.acceleration
            predator.limit_velocity(max_speed=3.0)
            predator.position += predator.velocity
            predator.position = np.mod(predator.position, self.box_size)
        
        # Update equilibrium tracking
        self.eq_tracker.update(self)

    def get_positions(self):
        """Get positions of boids and predators"""
        boid_positions = np.array([boid.position for boid in self.flock])
        predator_positions = np.array([pred.position for pred in self.predators]) if self.predators else np.empty((0, 2))
        return boid_positions, predator_positions

    def get_velocities(self):
        """Get velocities of boids and predators"""
        boid_velocities = np.array([boid.velocity for boid in self.flock])
        predator_velocities = np.array([pred.velocity for pred in self.predators]) if self.predators else np.empty((0, 2))
        return boid_velocities, predator_velocities

    def get_clusters(self, threshold=8.0):
        """Get current clusters using friends-of-friends algorithm"""
        positions = np.array([boid.position for boid in self.flock])
        return self.eq_tracker.friends_of_friends_clustering(positions, threshold)


# Wind functions
def uniform_wind(strength=0.05):
    return lambda pos: np.array([strength, 0])

def wavy_wind(pos):
    return np.array([0.3 * np.sin(pos[1] / 10), 0.3 * np.cos(pos[0] / 10)])

def vortex_wind(pos, center=None, strength=0.05):
    if center is None:
        center = np.array([50, 50])
    offset = pos - center
    distance = np.linalg.norm(offset)
    if distance < 1:
        return np.zeros(2)
    perp = np.array([-offset[1], offset[0]])
    return strength * perp / distance


# Simulation parameters
box_size = 100
n_boids = 40
n_predators = 1
num_obstacles = 2
free_will_strength = np.pi / 8

# Create simulation
sim = Flock(n_boids=n_boids, n_predators=n_predators, box_size=box_size, 
           obstacles=num_obstacles, wind_func=None, free_will=free_will_strength)

# Set up visualization with subplots
fig = plt.figure(figsize=(16, 12))

# Main simulation plot
ax_sim = plt.subplot(2, 3, (1, 4))
ax_sim.set_xlim(0, box_size)
ax_sim.set_ylim(0, box_size)
ax_sim.set_aspect('equal')
ax_sim.set_title("Flocking Simulation with Equilibrium Tracking", fontsize=12)
ax_sim.set_xlabel("X Position")
ax_sim.set_ylabel("Y Position")

# Equilibrium metrics plots
ax_speed = plt.subplot(2, 3, 2)
ax_speed.set_title("Speed & Alignment")
ax_speed.set_ylabel("Speed")

ax_cohesion = plt.subplot(2, 3, 3)
ax_cohesion.set_title("Cohesion & Clustering")
ax_cohesion.set_ylabel("Cohesion")

ax_autocorr = plt.subplot(2, 3, 5)
ax_autocorr.set_title("Autocorrelation Functions")
ax_autocorr.set_xlabel("Lag")
ax_autocorr.set_ylabel("Autocorrelation")

ax_clusters = plt.subplot(2, 3, 6)
ax_clusters.set_title("Cluster Analysis")
ax_clusters.set_xlabel("Time")
ax_clusters.set_ylabel("Number of Clusters")

# Draw obstacles
for obstacle in sim.obstacles:
    circle = plt.Circle((obstacle.x, obstacle.y), obstacle.radius, 
                       color='gray', alpha=0.5, zorder=1)
    ax_sim.add_patch(circle)



# Initialize plots
boid_positions, pred_positions = sim.get_positions()
boid_velocities, pred_velocities = sim.get_velocities()

boid_quiver = ax_sim.quiver(boid_positions[:, 0], boid_positions[:, 1],
                           boid_velocities[:, 0], boid_velocities[:, 1],
                           color='blue', angles='xy', scale_units='xy', scale=0.5,
                           alpha=0.7, width=0.003, zorder=2)

pred_quiver = None
if len(pred_positions) > 0:
    pred_quiver = ax_sim.quiver(pred_positions[:, 0], pred_positions[:, 1],
                               pred_velocities[:, 0], pred_velocities[:, 1],
                               color='red', angles='xy', scale_units='xy', scale=0.5,
                               alpha=0.8, width=0.005, zorder=3)

# Initialize empty lines for tracking plots
speed_line, = ax_speed.plot([], [], 'b-', label='Speed', alpha=0.7)
alignment_line, = ax_speed.plot([], [], 'r-', label='Alignment', alpha=0.7)
ax_speed.legend()

cohesion_line, = ax_cohesion.plot([], [], 'g-', label='Cohesion', alpha=0.7)
cluster_frac_line, = ax_cohesion.plot([], [], 'm-', label='Largest Cluster %', alpha=0.7)
ax_cohesion.legend()

cluster_num_line, = ax_clusters.plot([], [], 'c-', label='Num Clusters', alpha=0.7)
ax_clusters.legend()

def animate(frame):
    """Animation function with equilibrium tracking"""
    sim.update()
    
    # Update main simulation
    boid_positions, pred_positions = sim.get_positions()
    boid_velocities, pred_velocities = sim.get_velocities()
    
    boid_quiver.set_offsets(boid_positions)
    boid_quiver.set_UVC(boid_velocities[:, 0], boid_velocities[:, 1])
    
    if pred_quiver and len(pred_positions) > 0:
        pred_quiver.set_offsets(pred_positions)
        pred_quiver.set_UVC(pred_velocities[:, 0], pred_velocities[:, 1])
    
    # Highlight clusters with different colors
    if frame > 50:  # Allow some time for data collection
        clusters = sim.get_clusters(threshold=10.0)
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        
        # Clear previous cluster highlights
        for patch in ax_sim.patches[len(sim.obstacles):]:
            patch.remove()
        
        # Add cluster highlights
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:  # Only highlight multi-boid clusters
                cluster_positions = boid_positions[cluster]
                center = np.mean(cluster_positions, axis=0)
                max_dist = np.max([np.linalg.norm(pos - center) for pos in cluster_positions])
                circle = plt.Circle(center, max_dist + 2, color=colors[i], 
                                  alpha=0.2, zorder=0)
                ax_sim.add_patch(circle)
    
    # Update tracking plots
    if frame > 10 and len(sim.eq_tracker.history['speeds']) > 10:
        time_steps = range(len(sim.eq_tracker.history['speeds']))
        
        # Speed and alignment
        ax_speed.clear()
        ax_speed.plot(time_steps, sim.eq_tracker.history['speeds'], 'b-', 
                     label='Speed', alpha=0.7)
        if len(sim.eq_tracker.history['alignment']) > 0:
            ax_speed.plot(time_steps, sim.eq_tracker.history['alignment'], 'r-', 
                         label='Alignment', alpha=0.7)
        ax_speed.set_title("Speed & Alignment")
        ax_speed.legend()
        ax_speed.grid(True, alpha=0.3)
        
        # Cohesion and clustering
        ax_cohesion.clear()
        ax_cohesion.plot(time_steps, sim.eq_tracker.history['cohesion'], 'g-', 
                        label='Cohesion', alpha=0.7)
        if len(sim.eq_tracker.history['largest_cluster_fraction']) > 0:
            ax_cohesion.plot(time_steps, sim.eq_tracker.history['largest_cluster_fraction'], 
                           'm-', label='Largest Cluster %', alpha=0.7)
        ax_cohesion.set_title("Cohesion & Clustering")
        ax_cohesion.legend()
        ax_cohesion.grid(True, alpha=0.3)
        
        # Number of clusters
        ax_clusters.clear()
        ax_clusters.plot(time_steps, sim.eq_tracker.history['num_clusters'], 'c-', 
                        label='Num Clusters', alpha=0.7)
        ax_clusters.set_title("Cluster Analysis")
        ax_clusters.set_xlabel("Time")
        ax_clusters.legend()
        ax_clusters.grid(True, alpha=0.3)
        
        # Autocorrelation (update less frequently)
        if frame % 50 == 0 and len(sim.eq_tracker.history['speeds']) > 100:
            ax_autocorr.clear()
            
            for key, color, label in [('speeds', 'blue', 'Speed'), 
                                    ('alignment', 'red', 'Alignment'),
                                    ('cohesion', 'green', 'Cohesion')]:
                if len(sim.eq_tracker.history[key]) > 100:
                    lags, autocorr = sim.eq_tracker.compute_autocorrelation(
                        sim.eq_tracker.history[key], max_lag=50)
                    if len(autocorr) > 0:
                        ax_autocorr.plot(lags, autocorr, color=color, 
                                       label=f'{label}', alpha=0.7)
            
            ax_autocorr.axhline(y=1/np.e, color='black', linestyle='--', 
                              alpha=0.5, label='1/e decay')
            ax_autocorr.set_title("Autocorrelation Functions")
            ax_autocorr.set_xlabel("Lag")
            ax_autocorr.legend()
            ax_autocorr.grid(True, alpha=0.3)
    
    # Display equilibrium metrics in title
    if frame > 100:
        metrics = sim.eq_tracker.get_equilibrium_metrics()
        if metrics:
            metric_str = f"Stability - Speed: {metrics.get('speeds_stability', 0):.3f}, "
            metric_str += f"Alignment: {metrics.get('alignment_stability', 0):.3f}"
            ax_sim.set_title(f"Flocking Simulation - {metric_str}", fontsize=10)
    
    return [boid_quiver] + ([pred_quiver] if pred_quiver else [])

# Create and run animation
anim = FuncAnimation(fig, animate, frames=1000, interval=100, blit=False, repeat=True)
anim.save("test2.gif")
plt.tight_layout()
plt.savefig("monitoring.png")
plt.show()

# Print final equilibrium analysis
def print_equilibrium_analysis():
    if len(sim.eq_tracker.history['speeds']) > 100:
        print("\n=== Equilibrium Analysis ===")
        metrics = sim.eq_tracker.get_equilibrium_metrics()
        
        print(f"System Stability (Coefficient of Variation):")
        for key in ['speeds', 'alignment', 'cohesion', 'num_clusters']:
            stability_key = f'{key}_stability'
            if stability_key in metrics:
                print(f"  {key.capitalize()}: {metrics[stability_key]:.4f}")
        
        print(f"\nAutocorrelation Decay Times:")
        for key in ['speeds', 'alignment', 'cohesion']:
            autocorr_key = f'{key}_autocorr_time'
            if autocorr_key in metrics:
                print(f"  {key.capitalize()}: {metrics[autocorr_key]} time steps")
        
        # Recent cluster statistics
        recent_clusters = sim.eq_tracker.history['cluster_sizes'][-50:]
        if recent_clusters:
            avg_clusters = np.mean([len(clusters) for clusters in recent_clusters])
            avg_largest = np.mean(sim.eq_tracker.history['largest_cluster_fraction'][-50:])
            print(f"\nRecent Clustering (last 50 steps):")
            print(f"  Average number of clusters: {avg_clusters:.2f}")
            print(f"  Average largest cluster fraction: {avg_largest:.3f}")

# Uncomment to run analysis after simulation
print_equilibrium_analysis()
