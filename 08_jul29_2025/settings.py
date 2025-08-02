import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.animation as animation


class EulerAngles:
    """Crystal orientation in Euler angles"""
    def __init__(self, phi1: float, PHI: float, phi2: float):
        self.phi1 = phi1
        self.PHI = PHI
        self.phi2 = phi2
    
    def to_array(self):
        return np.array([self.phi1, self.PHI, self.phi2])

class Cell:
    """Individual cell in the CA grid"""
    def __init__(self, grain_id: int, orientation: EulerAngles):
        self.grain_id = grain_id
        self.orientation = orientation
        self.is_recrystallized = False
        self.stored_energy = np.random.uniform(0.8, 1.5)  # Initial stored energy
        self.dislocation_density = np.random.uniform(1e14, 1e15)  # m^-2
        
class CAGrid:
    """3D Cellular Automaton grid"""
    def __init__(self, nx: int, ny: int, nz: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.cells = np.empty((nx, ny, nz), dtype=object)
        self.initialize_microstructure()
        
    def initialize_microstructure(self):
        """Initialize each cell with random orientation"""
        grain_id = 0
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Random Euler angles
                    phi1 = np.random.uniform(0, 360)
                    PHI = np.random.uniform(0, 180)
                    phi2 = np.random.uniform(0, 360)
                    
                    orientation = EulerAngles(phi1, PHI, phi2)
                    self.cells[i, j, k] = Cell(grain_id, orientation)
                    grain_id += 1
    
    def get_neighbors(self, i: int, j: int, k: int) -> List[Tuple[int, int, int]]:
        """Get Moore neighborhood (26 neighbors in 3D)"""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    
                    # Periodic boundary conditions
                    ni = (i + di) % self.nx
                    nj = (j + dj) % self.ny
                    nk = (k + dk) % self.nz
                    
                    neighbors.append((ni, nj, nk))
        return neighbors

class EnergyCalculator:
    """Calculate various energy terms"""
    
    @staticmethod
    def calculate_misorientation(orient1: EulerAngles, orient2: EulerAngles) -> float:
        """Simplified misorientation calculation"""
        diff = np.abs(orient1.to_array() - orient2.to_array())
        # Wrap angles
        diff = np.minimum(diff, 360 - diff)
        return np.linalg.norm(diff)
    
    @staticmethod
    def calculate_gb_energy(misorientation: float) -> float:
        """Grain boundary energy as function of misorientation"""
        # Read-Shockley relationship (simplified)
        theta_max = 15.0  # degrees
        if misorientation < theta_max:
            return 0.5 * misorientation / theta_max
        else:
            return 0.5  # High angle grain boundary energy
    
    @staticmethod
    def calculate_driving_force(cell1: Cell, cell2: Cell) -> float:
        """Calculate driving force for grain boundary migration"""
        # Stored energy difference
        energy_diff = cell1.stored_energy - cell2.stored_energy
        
        # Add curvature effect (simplified)
        curvature_effect = 0.1 * np.random.uniform(-1, 1)
        
        return energy_diff + curvature_effect

class RecrystallizationCA:
    """Main CA simulation class"""
    
    def __init__(self, grid: CAGrid, temperature: float = 800):
        self.grid = grid
        self.temperature = temperature  # Kelvin
        self.time = 0.0
        self.dt = 0.1
        self.energy_calc = EnergyCalculator()
        
        # Physical parameters
        self.mobility_0 = 1e-6  # m^4/(J·s)
        self.activation_energy = 200000  # J/mol
        self.gas_constant = 8.314  # J/(mol·K)
        self.critical_stored_energy = 0.8
        self.nucleation_rate_constant = 1e-3
        
    def calculate_mobility(self, misorientation: float) -> float:
        """Temperature and misorientation dependent mobility"""
        mobility = self.mobility_0 * np.exp(-self.activation_energy / (self.gas_constant * self.temperature))
        
        # Misorientation dependency
        if misorientation < 15:  # Low angle boundary
            mobility *= misorientation / 15
            
        return mobility
    
    def nucleation_probability(self, cell: Cell) -> float:
        """Calculate nucleation probability"""
        if cell.is_recrystallized:
            return 0.0
            
        if cell.stored_energy < self.critical_stored_energy:
            return 0.0
            
        # Temperature and energy dependent
        prob = self.nucleation_rate_constant * np.exp(
            -self.activation_energy / (2 * self.gas_constant * self.temperature)
        ) * (cell.stored_energy / self.critical_stored_energy) ** 2
        
        return min(prob * self.dt, 1.0)
    
    def growth_probability(self, driving_force: float, mobility: float) -> float:
        """Calculate growth probability"""
        if driving_force <= 0:
            return 0.0
            
        velocity = mobility * driving_force
        prob = velocity * self.dt / 1e-6  # Normalize by cell size
        
        return min(prob, 1.0)
    
    def nucleate(self, i: int, j: int, k: int):
        """Create new recrystallized grain"""
        cell = self.grid.cells[i, j, k]
        
        # New random orientation
        phi1 = np.random.uniform(0, 360)
        PHI = np.random.uniform(0, 180)
        phi2 = np.random.uniform(0, 360)
        
        cell.orientation = EulerAngles(phi1, PHI, phi2)
        cell.is_recrystallized = True
        cell.stored_energy = 0.0  # Strain-free
        cell.dislocation_density = 1e10  # Very low
        
        # Assign new grain ID
        max_id = np.max([[c.grain_id for c in row] for row in self.grid.cells])
        cell.grain_id = max_id + 1
    
    def simulate_step(self):
        """One CA time step"""
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        
        # Create random update order
        indices = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
        random.shuffle(indices)
        
        # Nucleation phase
        for i, j, k in indices:
            cell = self.grid.cells[i, j, k]
            if np.random.random() < self.nucleation_probability(cell):
                self.nucleate(i, j, k)
        
        # Growth phase
        cells_changed = 0
        for i, j, k in indices:
            cell = self.grid.cells[i, j, k]
            neighbors = self.grid.get_neighbors(i, j, k)
            
            # Find best neighbor to switch to
            max_prob = 0.0
            best_neighbor = None
            
            for ni, nj, nk in neighbors:
                neighbor = self.grid.cells[ni, nj, nk]
                
                if neighbor.grain_id == cell.grain_id:
                    continue
                
                # Calculate driving force
                driving_force = self.energy_calc.calculate_driving_force(cell, neighbor)
                
                # Calculate mobility
                misorientation = self.energy_calc.calculate_misorientation(
                    cell.orientation, neighbor.orientation
                )
                mobility = self.calculate_mobility(misorientation)
                
                # Growth probability
                prob = self.growth_probability(driving_force, mobility)
                
                if prob > max_prob:
                    max_prob = prob
                    best_neighbor = neighbor
            
            # Probabilistic switching
            if best_neighbor and np.random.random() < max_prob:
                cell.grain_id = best_neighbor.grain_id
                cell.orientation = best_neighbor.orientation
                cell.is_recrystallized = best_neighbor.is_recrystallized
                cell.stored_energy = best_neighbor.stored_energy
                cells_changed += 1
        
        # Recovery: reduce stored energy
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    cell = self.grid.cells[i, j, k]
                    if not cell.is_recrystallized:
                        recovery_rate = 1e-3 * np.exp(-self.activation_energy / (3 * self.gas_constant * self.temperature))
                        cell.stored_energy *= (1 - recovery_rate * self.dt)
        
        self.time += self.dt
        return cells_changed
    
    def get_grain_ids(self):
        """Extract grain ID array for visualization"""
        grain_ids = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz))
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    grain_ids[i, j, k] = self.grid.cells[i, j, k].grain_id
        return grain_ids
    
    def get_recrystallized_fraction(self):
        """Calculate fraction of recrystallized cells"""
        rx_count = 0
        total = 0
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    if self.grid.cells[i, j, k].is_recrystallized:
                        rx_count += 1
                    total += 1
        return rx_count / total

def visualize_3d_slice(ca_sim: RecrystallizationCA, slice_index: int = 5):
    """Visualize a 2D slice of the 3D microstructure"""
    grain_ids = ca_sim.get_grain_ids()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # XY slice at given Z
    slice_xy = grain_ids[:, :, slice_index]
    im1 = ax1.imshow(slice_xy, cmap='tab20', interpolation='nearest')
    ax1.set_title(f'XY Slice at Z={slice_index}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # XZ slice at middle Y
    slice_xz = grain_ids[:, ca_sim.grid.ny//2, :]
    im2 = ax2.imshow(slice_xz, cmap='tab20', interpolation='nearest')
    ax2.set_title(f'XZ Slice at Y={ca_sim.grid.ny//2}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    
    plt.tight_layout()
    return fig

def plot_evolution(ca_sim: RecrystallizationCA, steps: int = 50):
    """Plot evolution of recrystallization"""
    rx_fractions = []
    times = []
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_steps = [0, 10, 20, 30, 40, 50]
    plot_index = 0
    
    for step in range(steps + 1):
        if step > 0:
            cells_changed = ca_sim.simulate_step()
            print(f"Step {step}: {cells_changed} cells changed")
        
        rx_frac = ca_sim.get_recrystallized_fraction()
        rx_fractions.append(rx_frac)
        times.append(ca_sim.time)
        
        # Plot microstructure at specific steps
        if step in plot_steps and plot_index < 6:
            grain_ids = ca_sim.get_grain_ids()
            slice_data = grain_ids[:, :, 5]
            
            im = axes[plot_index].imshow(slice_data, cmap='tab20', interpolation='nearest')
            axes[plot_index].set_title(f'Step {step}, RX: {rx_frac:.2%}')
            axes[plot_index].set_xlabel('X')
            axes[plot_index].set_ylabel('Y')
            plot_index += 1
    
    plt.tight_layout()
    
    # Plot recrystallization kinetics
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times, rx_fractions, 'b-', linewidth=2)
    ax.set_xlabel('Time (arbitrary units)')
    ax.set_ylabel('Recrystallized Fraction')
    ax.set_title('Recrystallization Kinetics')
    ax.grid(True)
    ax.set_ylim([0, 1.05])
    
    return fig, fig2