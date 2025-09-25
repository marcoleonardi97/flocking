# Flocking Simulation

An interactive simulation of boid flocking behavior modeling swarm intelligence, predator-prey dynamics, and emergent collective motion using simple rules. This project implements both Python analysis tools and a web-based interactive simulation.
[Run the simulation](https://marcoleonardi97.github.io/flocking)

## Features

### Interactive Web Simulation
- **Real-time Boid Flocking**: Watch hundreds of boids exhibit emergent behavior
- **Multiple Animal Presets**: Fish, Birds, Herds, Insects, and Migrating Birds with realistic parameters
- **Environmental Controls**: Add obstacles, wind effects, and predators
- **Real-time Metrics**: Monitor speed, alignment, local equilibrium, and directional entropy
- **Visual Options**: Trails, velocity vectors, perception radius visualization
- **Interactive Controls**: Click to place/remove obstacles, adjustable parameters

### Python Analysis Tools
- **Parameter Analysis**: Study how free will, perception radius, and predator count affect flocking
- **Equilibrium Monitoring**: Calculate and visualize the ψ (psi) alignment parameter
- **Statistical Analysis**: Generate velocity angle distributions and equilibrium plots
- **Animation Export**: Create GIF animations of simulations
- **Data Export**: Save analysis results as NumPy arrays and PNG plots

## Live Demo

🌐 **[Try the Interactive Simulation](https://marcoleonardi97.github.io/flocking/)**

## Project Structure

```
flocking/
├── index.html              # Interactive web simulation
├── main.py                 # Python analysis and parameter studies  
├── analysis.py             # Core flocking simulation classes (Python)
├── README.md               # This file
└── output/                 # Generated plots and data
    ├── psis_*.npy          # Saved equilibrium data
    └── *.png               # Analysis plots
```

## Requirements

### For Python Analysis
```
numpy
matplotlib
scipy (optional, for advanced analysis)
```

### For Web Simulation
- Modern web browser with JavaScript enabled
- No additional dependencies required

## Installation & Usage

### Python Analysis

1. Clone the repository:
```bash
git clone https://github.com/marcoleonardi97/flocking.git
cd flocking
```

2. Install Python dependencies:
```bash
pip install numpy matplotlib
```

3. Run parameter analysis:
```bash
python main.py
```

### Web Simulation

Simply open `index.html` in a web browser or visit the live demo link above.

## Simulation Parameters

### Core Flocking Rules (Reynolds 1987)
- **Alignment**: Boids steer towards average heading of neighbors
- **Cohesion**: Boids steer towards average position of neighbors  
- **Separation**: Boids avoid crowding local flockmates

### Advanced Parameters
- **Perception Radius**: Distance within which boids can sense neighbors
- **Field of View**: Angular range of perception (0 to 2π radians)
- **Free Will**: Random movement component (0 = pure flocking, 1 = random walk)
- **Wind Effects**: Uniform flow or vortex patterns
- **Predator Avoidance**: Escape behavior when predators are present

## Animal Behavior Presets

The simulation includes scientifically-inspired parameter sets:

- **🐟 Fish**: Tight schooling with vortex currents
- **🕊️ Birds**: Balanced flocking with moderate alignment
- **🦌 Herd**: Ground-based grouping with close cohesion  
- **🐝 Insects**: Swarm behavior with limited alignment
- **🦆 Migrating Birds**: Directional movement with wind assistance

## Scientific Metrics

### Real-time Monitoring
- **Average Speed**: Mean velocity magnitude of the flock
- **Alignment**: Directional coherence (dot product of velocity vectors)
- **Local Equilibrium (ψ)**: Order parameter measuring neighbor alignment
- **Average Neighbors**: Mean number of perceived neighbors per boid
- **Directional Entropy**: Measure of velocity direction randomness
![monit](https://github.com/user-attachments/assets/1a6b1777-7ed9-4381-b5ae-f6537c31d0ca)

![monit2](https://github.com/user-attachments/assets/1b2fb80a-9dfa-4513-b92b-6e945a3bca4d)

### Analysis Outputs
- Velocity angle distributions
- Equilibrium vs parameter plots
- Time series of order parameters
- Statistical correlations

## Technical Implementation

### Web Simulation (JavaScript)
- **Vector2 Class**: 2D vector mathematics
- **Boid Class**: Individual agent with flocking behaviors
- **Predator Class**: Hunting behavior and boid avoidance
- **Obstacle Class**: Static environmental barriers
- **Real-time Rendering**: HTML5 Canvas with 60fps animation
- **Chart.js Integration**: Live entropy plotting

### Python Analysis
- **Flock Class**: Simulation engine with configurable parameters
- **Metric Calculation**: Automated equilibrium and alignment analysis
- **Visualization**: Matplotlib-based plotting and animation
- **Data Export**: NumPy arrays for further analysis

## Research Applications

This simulation is designed for:
- **Educational Demonstrations**: Understanding emergence and self-organization
- **Parameter Studies**: Investigating phase transitions in collective behavior
- **Behavioral Modeling**: Comparing different animal movement patterns
- **Algorithm Development**: Testing swarm intelligence applications

## Controls

### Web Interface
- **Mouse**: Click to add/remove obstacles
- **Sliders**: Adjust all flocking parameters in real-time
- **Buttons**: Add predators, change presets, toggle visual options
- **Presets**: Quick setup for different animal behaviors

### Python Analysis
- **Variable Selection**: Choose 'Free Will', 'Perception Radius', or 'Predators'
- **Parameter Ranges**: Customize analysis ranges in the script
- **Output Control**: Enable/disable animations, plots, and data saving

## Scientific Background

Based on Craig Reynolds' 1987 boids model and extended with:
- **Viscek Model**: Statistical physics approach to flocking
- **Predator-Prey Dynamics**: Ecological interactions
- **Active Matter Physics**: Non-equilibrium collective behavior
- **Information Theory**: Entropy measures of organization

## Performance

### Web Simulation
- **Optimized Rendering**: Efficient Canvas API usage
- **Scalable**: Handles 10-200 boids smoothly
- **Responsive**: Real-time parameter updates

### Python Analysis  
- **Batch Processing**: Automated parameter sweeps
- **Data Management**: Efficient NumPy operations
- **Visualization**: High-quality matplotlib outputs

## Contributing

Contributions welcome! Areas for development:
- [ ] 3D flocking simulation
- [ ] Machine learning behavior analysis  
- [ ] Additional animal behavior presets
- [ ] Performance optimizations
- [ ] Mobile touch controls

## Authors

- **Marco Leonardi** - Core simulation development
- **Rudi Van Velzen** - Web interface and interactive features

## License

[Add your chosen license here]

## References

1. Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model
2. Vicsek, T. et al. (1995). Novel type of phase transition in a system of self-driven particles
3. Ballerini, M. et al. (2008). Interaction ruling animal collective behavior

## Acknowledgments

- Inspired by natural flocking behaviors in birds, fish, and mammals
- Built with modern web technologies for accessibility and education
- Designed for both research and public engagement with complex systems
