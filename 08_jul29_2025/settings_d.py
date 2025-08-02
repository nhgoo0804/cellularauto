import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import random
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
    
    def copy(self):
        return EulerAngles(self.phi1, self.PHI, self.phi2)

class Cell:
    """Individual cell in the CA grid"""
    def __init__(self, grain_id: int, orientation: EulerAngles):
        self.grain_id = grain_id
        self.orientation = orientation
        self.is_recrystallized = False
        self.stored_energy = np.random.uniform(0.8, 1.0)
        self.dislocation_density = np.random.uniform(1e14, 1e15)
        
class CAGrid:
    """3D Cellular Automaton grid"""
    def __init__(self, nx: int, ny: int, nz: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.cells = np.empty((nx, ny, nz), dtype=object)
        self.initialize_microstructure()
        
    def initialize_microstructure(self):
        """Initialize with proper grain structure"""
        # Step 1: Create a deformed single crystal or few large grains
        # This represents the cold-worked material before recrystallization
        
        # Option 1: Single crystal (most common for RX studies)
        base_orientation = EulerAngles(45.0, 30.0, 60.0)
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Small orientation variations to represent deformation
                    phi1 = base_orientation.phi1 + np.random.uniform(-5, 5)
                    PHI = base_orientation.PHI + np.random.uniform(-5, 5)
                    phi2 = base_orientation.phi2 + np.random.uniform(-5, 5)
                    
                    orientation = EulerAngles(phi1, PHI, phi2)
                    self.cells[i, j, k] = Cell(0, orientation)  # All grain_id = 0 initially
                    
                    # High stored energy (deformed state)
                    self.cells[i, j, k].stored_energy = np.random.uniform(0.9, 1.2)
                    self.cells[i, j, k].dislocation_density = np.random.uniform(1e14, 5e15)
        
        # Step 2: Add some regions with very high stored energy (deformation bands)
        self.add_deformation_bands()
    
    def add_deformation_bands(self):
        """Add deformation bands with higher stored energy"""
        # Create 3-4 deformation bands
        for band in range(3):
            # Random band direction
            if band % 2 == 0:
                # Horizontal band
                y_center = np.random.randint(2, self.ny-2)
                for i in range(self.nx):
                    for j in range(max(0, y_center-1), min(self.ny, y_center+2)):
                        for k in range(self.nz):
                            self.cells[i, j, k].stored_energy = np.random.uniform(1.5, 2.0)
                            self.cells[i, j, k].dislocation_density = np.random.uniform(5e15, 1e16)
            else:
                # Vertical band
                x_center = np.random.randint(2, self.nx-2)
                for i in range(max(0, x_center-1), min(self.nx, x_center+2)):
                    for j in range(self.ny):
                        for k in range(self.nz):
                            self.cells[i, j, k].stored_energy = np.random.uniform(1.5, 2.0)
                            self.cells[i, j, k].dislocation_density = np.random.uniform(5e15, 1e16)
    
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
        """Calculate misorientation angle between two orientations"""
        # Simplified calculation
        diff = orient1.to_array() - orient2.to_array()
        # Handle angle wrapping
        diff = np.abs(diff)
        diff = np.minimum(diff, 360 - diff)
        # Return magnitude
        return np.sqrt(np.sum(diff**2))
    
    @staticmethod
    def calculate_gb_energy(misorientation: float) -> float:
        """Grain boundary energy as function of misorientation"""
        # Read-Shockley for low angle, constant for high angle
        theta_max = 15.0  # degrees
        if misorientation < theta_max:
            return 0.5 * (misorientation / theta_max) * (1 - np.log(misorientation / theta_max))
        else:
            return 0.5  # High angle grain boundary energy

class RecrystallizationCA:
    """Main CA simulation class"""
    
    def __init__(self, grid: CAGrid, temperature: float = 823):  # 550°C
        self.grid = grid
        self.temperature = temperature  # Kelvin
        self.time = 0.0
        self.dt = 1.0
        self.energy_calc = EnergyCalculator()
        self.next_grain_id = 1  # For new recrystallized grains
        
        # Physical parameters
        self.critical_stored_energy = 1.3  # Threshold for nucleation
        self.nucleation_constant = 0.001  # Base nucleation probability
        self.mobility_0 = 1.0  # Base mobility
        self.Q_activation = 150000  # J/mol
        self.R = 8.314  # J/(mol·K)
        
    def calculate_mobility(self, misorientation: float) -> float:
        """Calculate grain boundary mobility"""
        # Arrhenius temperature dependence
        mobility = self.mobility_0 * np.exp(-self.Q_activation / (self.R * self.temperature))
        
        # Misorientation dependence
        if misorientation < 15:  # Low angle
            mobility *= (misorientation / 15)
        
        return mobility
    
    def nucleation_check(self, cell: Cell) -> bool:
        """Check if nucleation should occur"""
        if cell.is_recrystallized:
            return False
        
        if cell.stored_energy < self.critical_stored_energy:
            return False
        
        # Probability increases with stored energy
        energy_factor = (cell.stored_energy - self.critical_stored_energy) / self.critical_stored_energy
        prob = self.nucleation_constant * energy_factor * self.dt
        
        return np.random.random() < prob
    
    def nucleate(self, i: int, j: int, k: int):
        """Create a new recrystallized grain"""
        cell = self.grid.cells[i, j, k]
        
        # New grain with new orientation
        phi1 = np.random.uniform(0, 360)
        PHI = np.random.uniform(0, 180)
        phi2 = np.random.uniform(0, 360)
        
        cell.orientation = EulerAngles(phi1, PHI, phi2)
        cell.grain_id = self.next_grain_id
        self.next_grain_id += 1
        cell.is_recrystallized = True
        cell.stored_energy = 0.01  # Nearly zero
        cell.dislocation_density = 1e10  # Very low
    
    def growth_check(self, cell: Cell, neighbor: Cell, misorientation: float) -> bool:
        """Check if grain boundary should migrate"""
        # No growth if same grain
        if cell.grain_id == neighbor.grain_id:
            return False
        
        # Calculate driving pressure
        driving_pressure = cell.stored_energy - neighbor.stored_energy
        
        # Recrystallization front has additional driving force
        if neighbor.is_recrystallized and not cell.is_recrystallized:
            driving_pressure += 0.5
        
        # No growth against driving pressure
        if driving_pressure <= 0:
            return False
        
        # Calculate velocity
        mobility = self.calculate_mobility(misorientation)
        velocity = mobility * driving_pressure
        
        # Probability of switching
        cell_size = 1.0  # μm
        prob = velocity * self.dt / cell_size
        
        return np.random.random() < min(prob, 1.0)
    
    def simulate_step(self):
        """Perform one CA time step"""
        # Lists to track changes
        cells_to_nucleate = []
        cells_to_switch = []
        
        # Phase 1: Check for nucleation
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    if self.nucleation_check(self.grid.cells[i, j, k]):
                        cells_to_nucleate.append((i, j, k))
        
        # Phase 2: Check for growth (random order)
        indices = [(i, j, k) for i in range(self.grid.nx) 
                   for j in range(self.grid.ny) 
                   for k in range(self.grid.nz)]
        random.shuffle(indices)
        
        for i, j, k in indices:
            cell = self.grid.cells[i, j, k]
            neighbors = self.grid.get_neighbors(i, j, k)
            
            # Check each neighbor
            growth_candidates = []
            for ni, nj, nk in neighbors:
                neighbor = self.grid.cells[ni, nj, nk]
                
                if cell.grain_id == neighbor.grain_id:
                    continue
                
                misorientation = self.energy_calc.calculate_misorientation(
                    cell.orientation, neighbor.orientation
                )
                
                if self.growth_check(cell, neighbor, misorientation):
                    growth_candidates.append((ni, nj, nk))
            
            # Select one neighbor randomly if multiple candidates
            if growth_candidates:
                ni, nj, nk = random.choice(growth_candidates)
                neighbor = self.grid.cells[ni, nj, nk]
                cells_to_switch.append((i, j, k, neighbor))
        
        # Phase 3: Apply changes
        # Nucleation
        for i, j, k in cells_to_nucleate:
            self.nucleate(i, j, k)
        
        # Growth
        for i, j, k, neighbor in cells_to_switch:
            cell = self.grid.cells[i, j, k]
            cell.grain_id = neighbor.grain_id
            cell.orientation = neighbor.orientation.copy()
            cell.is_recrystallized = neighbor.is_recrystallized
            if neighbor.is_recrystallized:
                cell.stored_energy = neighbor.stored_energy
                cell.dislocation_density = neighbor.dislocation_density
        
        # Phase 4: Recovery (stored energy reduction in non-RX regions)
        if self.temperature > 0:
            recovery_rate = 0.001 * np.exp(-self.Q_activation / (2 * self.R * self.temperature))
            for i in range(self.grid.nx):
                for j in range(self.grid.ny):
                    for k in range(self.grid.nz):
                        cell = self.grid.cells[i, j, k]
                        if not cell.is_recrystallized:
                            cell.stored_energy *= (1 - recovery_rate * self.dt)
        
        self.time += self.dt
        
        # Return number of nucleation and growth events
        return len(cells_to_nucleate), len(cells_to_switch)
    
    def get_grain_ids(self):
        """Extract grain ID array for visualization"""
        grain_ids = np.zeros((self.grid.nx, self.grid.ny, self.grid.nz), dtype=int)
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    grain_ids[i, j, k] = self.grid.cells[i, j, k].grain_id
        return grain_ids
    
    def get_rx_fraction(self):
        """Calculate recrystallized volume fraction"""
        rx_count = 0
        total = self.grid.nx * self.grid.ny * self.grid.nz
        
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    if self.grid.cells[i, j, k].is_recrystallized:
                        rx_count += 1
        
        return rx_count / total

def visualize_evolution(grid_size=(20, 20, 20), temperature=823, max_steps=100):
    """Run and visualize the complete simulation"""
    # Create grid
    print(f"Creating {grid_size[0]}x{grid_size[1]}x{grid_size[2]} grid...")
    grid = CAGrid(*grid_size)
    
    # Initialize simulation
    ca_sim = RecrystallizationCA(grid, temperature=temperature)
    
    # Setup figure
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2)
    ax3 = plt.subplot(2, 3, 3)
    ax4 = plt.subplot(2, 3, 4)
    ax5 = plt.subplot(2, 3, 5)
    ax6 = plt.subplot(2, 1, 2)
    
    # Storage for kinetics
    times = []
    rx_fractions = []
    
    # Visualization steps
    viz_steps = [0, 20, 40, 60, 80, 100]
    viz_index = 0
    axes = [ax1, ax2, ax3, ax4, ax5]
    
    # Color map for grains
    cmap = plt.cm.get_cmap('tab20')
    
    print("\nRunning simulation...")
    print("Step | Nucleations | Growth Events | RX Fraction")
    print("-" * 50)
    
    for step in range(max_steps + 1):
        if step > 0:
            n_nuc, n_growth = ca_sim.simulate_step()
        else:
            n_nuc, n_growth = 0, 0
        
        # Calculate statistics
        rx_frac = ca_sim.get_rx_fraction()
        times.append(ca_sim.time)
        rx_fractions.append(rx_frac)
        
        # Print progress
        if step % 10 == 0:
            print(f"{step:4d} | {n_nuc:11d} | {n_growth:13d} | {rx_frac:11.2%}")
        
        # Visualize at specific steps
        if step in viz_steps and viz_index < len(axes):
            grain_ids = ca_sim.get_grain_ids()
            slice_z = grid_size[2] // 2
            
            # Create custom colormap
            unique_ids = np.unique(grain_ids)
            n_colors = len(unique_ids)
            colors_list = [cmap(i % 20) for i in range(n_colors)]
            grain_cmap = colors.ListedColormap(colors_list)
            
            # Plot slice
            im = axes[viz_index].imshow(grain_ids[:, :, slice_z], 
                                       cmap=grain_cmap, 
                                       interpolation='nearest')
            axes[viz_index].set_title(f'Step {step}, RX: {rx_frac:.1%}')
            axes[viz_index].set_xlabel('X')
            axes[viz_index].set_ylabel('Y')
            viz_index += 1
    
    # Plot kinetics curve
    ax6.plot(times, rx_fractions, 'b-', linewidth=2)
    ax6.set_xlabel('Time (arbitrary units)')
    ax6.set_ylabel('Recrystallized Fraction')
    ax6.set_title('Recrystallization Kinetics')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.05])
    
    # Add JMAK fit if enough RX
    if max(rx_fractions) > 0.1:
        try:
            # Fit JMAK equation: X = 1 - exp(-kt^n)
            from scipy.optimize import curve_fit
            
            def jmak(t, k, n):
                return 1 - np.exp(-k * t**n)
            
            # Fit only where RX > 0.01
            mask = np.array(rx_fractions) > 0.01
            if np.sum(mask) > 5:
                popt, _ = curve_fit(jmak, np.array(times)[mask], 
                                   np.array(rx_fractions)[mask], 
                                   p0=[0.001, 3.0])
                
                t_fit = np.linspace(0, max(times), 100)
                rx_fit = jmak(t_fit, *popt)
                ax6.plot(t_fit, rx_fit, 'r--', 
                        label=f'JMAK fit: n={popt[1]:.2f}')
                ax6.legend()
        except:
            pass
    
    plt.tight_layout()
    
    # Print final statistics
    final_grains = ca_sim.get_grain_ids()
    unique_grains = len(np.unique(final_grains))
    print(f"\nFinal Statistics:")
    print(f"  Number of grains: {unique_grains}")
    print(f"  RX fraction: {rx_fractions[-1]:.2%}")
    print(f"  Average grain size: {grid_size[0]*grid_size[1]*grid_size[2]/unique_grains:.1f} cells")
    
    return fig, ca_sim

def plot_final_microstructure(ca_sim, slice_z=None):
    """Plot the final microstructure with proper grain visualization"""
    grain_ids = ca_sim.get_grain_ids()
    
    if slice_z is None:
        slice_z = ca_sim.grid.nz // 2
    
    # Get 2D slice
    slice_data = grain_ids[:, :, slice_z]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Grain structure
    unique_grains = np.unique(slice_data)
    n_grains = len(unique_grains)
    
    # Create custom colormap
    if n_grains <= 20:
        cmap = plt.cm.get_cmap('tab20')
    else:
        cmap = plt.cm.get_cmap('hsv')
    
    # Normalize grain IDs to colormap range
    norm = plt.Normalize(vmin=slice_data.min(), vmax=slice_data.max())
    
    im1 = ax1.imshow(slice_data, cmap=cmap, norm=norm, interpolation='nearest')
    ax1.set_title(f'Final Microstructure (Z={slice_z})\n{n_grains} grains in this slice')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Grain ID')
    
    # Plot 2: Recrystallization state
    rx_map = np.zeros_like(slice_data, dtype=float)
    for i in range(ca_sim.grid.nx):
        for j in range(ca_sim.grid.ny):
            if ca_sim.grid.cells[i, j, slice_z].is_recrystallized:
                rx_map[i, j] = 1.0
            else:
                rx_map[i, j] = ca_sim.grid.cells[i, j, slice_z].stored_energy
    
    im2 = ax2.imshow(rx_map, cmap='RdBu_r', vmin=0, vmax=2, interpolation='nearest')
    ax2.set_title('Recrystallization State\n(Blue=RX, Red=Deformed)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Stored Energy / RX State')
    
    plt.tight_layout()
    
    # Print statistics
    total_grains_3d = len(np.unique(grain_ids))
    rx_fraction = ca_sim.get_rx_fraction()
    print(f"\nFinal Microstructure Statistics:")
    print(f"  Total grains (3D): {total_grains_3d}")
    print(f"  Grains in slice: {n_grains}")
    print(f"  RX fraction: {rx_fraction:.1%}")
    print(f"  Average grain volume: {ca_sim.grid.nx*ca_sim.grid.ny*ca_sim.grid.nz/total_grains_3d:.1f} cells")
    
    return fig


    
    