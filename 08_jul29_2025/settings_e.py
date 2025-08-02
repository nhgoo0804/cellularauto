#####################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
from typing import List, Tuple

class Cell:
    """Cell state for CA simulation"""
    def __init__(self):
        self.phase = 0  # 0=deformed, 1=recrystallized
        self.orientation = 0  # grain orientation ID
        self.dislocation_density = 1e15  # m^-2
        self.stored_energy = 1.0  # J/m^3

class CAGrid:
    """3D Cellular Automaton grid following OMicroN structure"""
    def __init__(self, nx: int, ny: int, nz: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        # Create 3D grid
        self.cells = [[[Cell() for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]
        self.time = 0.0
        self.temperature = 773  # K
        
        # Initialize with deformed microstructure
        self._initialize_deformed_structure()
    
    def _initialize_deformed_structure(self):
        """Create initial deformed structure with varying stored energy"""
        # All cells start deformed with same orientation
        base_orientation = 0
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    cell = self.cells[i][j][k]
                    cell.phase = 0  # deformed
                    cell.orientation = base_orientation
                    # Stored energy varies spatially (deformation heterogeneity)
                    cell.stored_energy = 1.0 + 0.5 * np.sin(i/5) * np.cos(j/5)
                    cell.dislocation_density = 1e15 * cell.stored_energy
        
        # Add high-energy deformation bands
        self._add_deformation_bands()
    
    def _add_deformation_bands(self):
        """Add deformation bands with higher stored energy"""
        # Diagonal band 1
        for i in range(self.nx):
            j = int(self.ny * (0.3 + 0.1 * np.sin(i / 3)))
            if 0 <= j < self.ny:
                for k in range(self.nz):
                    for dj in range(-1, 2):
                        if 0 <= j + dj < self.ny:
                            self.cells[i][j + dj][k].stored_energy = 2.5
                            self.cells[i][j + dj][k].dislocation_density = 5e15
        
        # Diagonal band 2
        for j in range(self.ny):
            i = int(self.nx * (0.6 + 0.1 * np.cos(j / 3)))
            if 0 <= i < self.nx:
                for k in range(self.nz):
                    for di in range(-1, 2):
                        if 0 <= i + di < self.nx:
                            self.cells[i + di][j][k].stored_energy = 2.5
                            self.cells[i + di][j][k].dislocation_density = 5e15

class RecrystallizationCA:
    """CA model for recrystallization following standard metallurgical approach"""
    
    def __init__(self, grid: CAGrid):
        self.grid = grid
        self.nucleation_sites = []
        self.current_max_orientation = 0
        
        # Physical parameters (typical values for steel)
        self.Qnuc = 250000  # J/mol - nucleation activation energy
        self.Qgb = 140000   # J/mol - grain boundary migration activation energy
        self.R = 8.314      # J/(mol·K)
        self.M0 = 1e-6      # m^4/(J·s) - pre-exponential mobility
        self.Nnuc = 1e10    # nuclei/m^3 - nucleation site density
        self.critical_driving_force = 1e6  # J/m^3
        
        # CA parameters
        self.cell_size = 1e-6  # m (1 μm)
        self.time_step = 0.01  # s
        
    def nucleation_probability(self, cell: Cell, T: float) -> float:
        """Site-saturated nucleation probability"""
        if cell.phase == 1:  # already recrystallized
            return 0.0
        
        # Driving force from stored energy
        P = cell.stored_energy * 1e6  # Convert to J/m^3
        
        if P < self.critical_driving_force:
            return 0.0
        
        # Classical nucleation theory
        rate = self.Nnuc * np.exp(-self.Qnuc / (self.R * T))
        
        # Convert to probability
        volume = self.cell_size ** 3
        prob = 1 - np.exp(-rate * volume * self.time_step)
        
        return prob
    
    def grain_boundary_mobility(self, T: float) -> float:
        """Temperature-dependent grain boundary mobility"""
        return self.M0 * np.exp(-self.Qgb / (self.R * T))
    
    def can_grow(self, cell_from: Cell, cell_to: Cell, T: float) -> bool:
        """Check if grain boundary can migrate from cell_from to cell_to"""
        # Same grain - no migration
        if cell_from.orientation == cell_to.orientation:
            return False
        
        # Only recrystallized grains can grow
        if cell_from.phase == 0:
            return False
        
        # Driving pressure (J/m^3)
        P = (cell_to.stored_energy - cell_from.stored_energy) * 1e6
        
        # Grain boundary velocity
        M = self.grain_boundary_mobility(T)
        v = M * P
        
        # Migration distance in one time step
        distance = v * self.time_step
        
        # Probability of migration
        prob = distance / self.cell_size
        
        return random.random() < prob
    
    def get_neighbors(self, i: int, j: int, k: int) -> List[Tuple[int, int, int]]:
        """Get valid neighbors (26-connectivity)"""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni, nj, nk = i + di, j + dj, k + dk
                    # Check bounds
                    if 0 <= ni < self.grid.nx and 0 <= nj < self.grid.ny and 0 <= nk < self.grid.nz:
                        neighbors.append((ni, nj, nk))
        return neighbors
    
    def nucleate(self, i: int, j: int, k: int):
        """Create new recrystallized grain"""
        cell = self.grid.cells[i][j][k]
        self.current_max_orientation += 1
        cell.orientation = self.current_max_orientation
        cell.phase = 1  # recrystallized
        cell.stored_energy = 0.0  # strain-free
        cell.dislocation_density = 1e10  # very low
    
    def step(self):
        """One CA time step"""
        T = self.grid.temperature
        nucleated = 0
        grown = 0
        
        # Phase 1: Nucleation check
        nucleation_list = []
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    cell = self.grid.cells[i][j][k]
                    if random.random() < self.nucleation_probability(cell, T):
                        nucleation_list.append((i, j, k))
        
        # Apply nucleation
        for i, j, k in nucleation_list:
            self.nucleate(i, j, k)
            nucleated += 1
        
        # Phase 2: Growth - create list of potential switches
        switch_list = []
        
        # Random order iteration
        cells = [(i, j, k) for i in range(self.grid.nx) 
                 for j in range(self.grid.ny) 
                 for k in range(self.grid.nz)]
        random.shuffle(cells)
        
        for i, j, k in cells:
            cell_to = self.grid.cells[i][j][k]
            
            # Only non-recrystallized cells can be consumed
            if cell_to.phase == 1:
                continue
            
            # Check all neighbors
            neighbors = self.get_neighbors(i, j, k)
            rx_neighbors = []
            
            for ni, nj, nk in neighbors:
                cell_from = self.grid.cells[ni][nj][nk]
                if cell_from.phase == 1 and self.can_grow(cell_from, cell_to, T):
                    rx_neighbors.append((ni, nj, nk))
            
            # If multiple RX neighbors, choose one randomly
            if rx_neighbors:
                ni, nj, nk = random.choice(rx_neighbors)
                switch_list.append(((i, j, k), (ni, nj, nk)))
        
        # Apply growth
        for (i, j, k), (ni, nj, nk) in switch_list:
            cell_to = self.grid.cells[i][j][k]
            cell_from = self.grid.cells[ni][nj][nk]
            
            # Copy state from growing grain
            cell_to.phase = 1
            cell_to.orientation = cell_from.orientation
            cell_to.stored_energy = 0.0
            cell_to.dislocation_density = 1e10
            grown += 1
        
        # Update time
        self.grid.time += self.time_step
        
        return nucleated, grown
    
    def get_phase_field(self, z_slice=None):
        """Get phase field for visualization"""
        if z_slice is None:
            z_slice = self.grid.nz // 2
        
        phase = np.zeros((self.grid.nx, self.grid.ny))
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                phase[i, j] = self.grid.cells[i][j][z_slice].phase
        return phase
    
    def get_orientation_field(self, z_slice=None):
        """Get orientation field for visualization"""
        if z_slice is None:
            z_slice = self.grid.nz // 2
        
        orientation = np.zeros((self.grid.nx, self.grid.ny))
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                orientation[i, j] = self.grid.cells[i][j][z_slice].orientation
        return orientation
    
    def get_rx_fraction(self):
        """Calculate recrystallized volume fraction"""
        rx_count = 0
        total = self.grid.nx * self.grid.ny * self.grid.nz
        
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                for k in range(self.grid.nz):
                    if self.grid.cells[i][j][k].phase == 1:
                        rx_count += 1
        
        return rx_count / total

def run_simulation():
    """Run a complete recrystallization simulation"""
    # Create grid
    nx, ny, nz = 100, 100, 1  # 2D simulation for faster visualization
    grid = CAGrid(nx, ny, nz)
    grid.temperature = 823  # 550°C
    
    # Create CA model
    ca = RecrystallizationCA(grid)
    
    # Setup figure for animation
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Time points to capture
    capture_times = [0, 10, 20, 40, 60, 80, 100, 150]
    capture_index = 0
    
    # Storage for kinetics
    times = []
    rx_fractions = []
    
    print("Time (s) | Nucleated | Grown | RX Fraction")
    print("-" * 45)
    
    # Run simulation
    max_steps = 2000
    for step in range(max_steps):
        # Perform CA step
        n_nuc, n_grown = ca.step()
        
        # Calculate statistics
        rx_frac = ca.get_rx_fraction()
        times.append(grid.time)
        rx_fractions.append(rx_frac)
        
        # Print progress
        if step % 100 == 0:
            print(f"{grid.time:8.2f} | {n_nuc:9d} | {n_grown:6d} | {rx_frac:11.3f}")
        
        # Capture microstructure at specific times
        if capture_index < len(capture_times) and grid.time >= capture_times[capture_index]:
            orientation = ca.get_orientation_field()
            
            # Create discrete colormap
            unique_grains = np.unique(orientation)
            n_colors = len(unique_grains)
            cmap = plt.cm.get_cmap('tab20')
            
            # Plot
            ax = axes[capture_index]
            im = ax.imshow(orientation.T, origin='lower', interpolation='nearest', cmap=cmap)
            ax.set_title(f't = {grid.time:.1f}s, RX = {rx_frac:.1%}')
            ax.set_xlabel('X (μm)')
            ax.set_ylabel('Y (μm)')
            ax.axis('equal')
            
            capture_index += 1
        
        # Stop if fully recrystallized
        if rx_frac > 0.99:
            print(f"\nRecrystallization complete at t = {grid.time:.2f}s")
            break
    
    plt.tight_layout()
    
    # Plot kinetics
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times, rx_fractions, 'b-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Recrystallized Fraction')
    ax.set_title('Recrystallization Kinetics')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(times)])
    ax.set_ylim([0, 1.05])
    
    # Final statistics
    orientation_final = ca.get_orientation_field()
    n_grains = len(np.unique(orientation_final)) - 1  # Subtract deformed phase
    print(f"\nFinal number of grains: {n_grains}")
    print(f"Average grain area: {(nx*ny)/n_grains:.1f} cells")
    
    plt.show()
    
    return ca, grid

if __name__ == "__main__":
    print("=== Cellular Automaton Recrystallization Simulation ===")
    print("Based on standard metallurgical CA approach\n")
    
    ca_model, grid = run_simulation()

