import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import yaml
import vtk
from vtk.util import numpy_support
import os
import time

class CellularAutomatonRexGG3D:
    """
    3D Cellular Automaton for Recrystallization and Grain Growth
    Using quaternions for orientation and reading from VTI files
    """
    
    def __init__(self, config_file: str):
        """Initialize from YAML configuration file"""
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Material parameters from config
        self.grain_boundary_energy = self.config['material']['grain_boundary_energy']
        self.mobility_pre_exponential = self.config['material']['mobility_pre_exponential']
        self.activation_energy = self.config['material']['activation_energy']
        self.temperature = self.config['material']['temperature']
        self.R = 8.314  # J/(mol·K)
        
        # Misorientation thresholds
        self.min_misorientation = self.config['thresholds']['min_misorientation']
        self.lagb_threshold = self.config['thresholds']['lagb_threshold']
        self.hagb_threshold = self.config['thresholds']['hagb_threshold']
        
        # Stored energy parameters
        self.stored_energy_factor = self.config['energy']['stored_energy_factor']
        self.max_kam_degrees = self.config['energy']['max_kam_degrees']
        
        # Kinetic factor
        self.beta_factor = self.config['kinetics']['beta_factor']
        
        # Simulation parameters
        self.max_time = self.config['simulation']['max_time']
        self.max_steps = self.config['simulation']['max_steps']
        self.save_interval = self.config['simulation']['save_interval']
        self.output_dir = self.config['simulation']['output_directory']
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load initial microstructure
        self.initial_microstructure_path = self.config['initial_microstructure']['path']
        self._load_initial_microstructure()
        
        # Initialize cellular automaton state
        self._initialize_ca_state()
        
        # Statistics tracking
        self.total_time = 0.0
        self.step_count = 0
        
        print(f"Initialized 3D CA with grid size: {self.nx} x {self.ny} x {self.nz}")
        print(f"Total cells: {self.n_cells}")
    
    def _load_initial_microstructure(self):
        """Load initial microstructure from VTI file"""
        print(f"Loading initial microstructure from: {self.initial_microstructure_path}")
        
        # Read VTI file
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(self.initial_microstructure_path)
        reader.Update()
        
        # Get the image data
        image_data = reader.GetOutput()
        
        # Get dimensions
        self.nx, self.ny, self.nz = image_data.GetDimensions()
        self.n_cells = self.nx * self.ny * self.nz
        
        # Get spacing (cell size)
        spacing = image_data.GetSpacing()
        self.dx = spacing[0] * 1e-6  # Convert to meters if needed
        
        # Extract arrays
        point_data = image_data.GetPointData()
        
        # Get orientation data (assuming EulerAngles or OrientationID array)
        if point_data.GetArray('OrientationID'):
            ori_array = point_data.GetArray('OrientationID')
            self.orientation_id = numpy_support.vtk_to_numpy(ori_array).flatten()
        else:
            raise ValueError("OrientationID array not found in VTI file")
        
        # Get Euler angles if available
        self.quaternions = {}
        if point_data.GetArray('EulerAngles'):
            euler_array = point_data.GetArray('EulerAngles')
            euler_data = numpy_support.vtk_to_numpy(euler_array)
            
            # Convert Euler angles to quaternions for each unique orientation
            unique_oris = np.unique(self.orientation_id)
            for ori in unique_oris:
                # Find first cell with this orientation
                idx = np.where(self.orientation_id == ori)[0][0]
                phi1, Phi, phi2 = euler_data[idx]
                self.quaternions[ori] = self.euler_to_quaternion(phi1, Phi, phi2)
        else:
            # Generate random orientations if Euler angles not provided
            print("EulerAngles not found, generating random orientations")
            unique_oris = np.unique(self.orientation_id)
            for ori in unique_oris:
                phi1 = np.random.uniform(0, 360)
                Phi = np.random.uniform(0, 180)
                phi2 = np.random.uniform(0, 360)
                self.quaternions[ori] = self.euler_to_quaternion(phi1, Phi, phi2)
        
        # Get KAM if available
        if point_data.GetArray('KAM'):
            kam_array = point_data.GetArray('KAM')
            self.kam = numpy_support.vtk_to_numpy(kam_array).flatten()
        else:
            print("KAM not found, will calculate")
            self.kam = np.zeros(self.n_cells)
            self._calculate_kam()
        
        # Get stored energy if available
        if point_data.GetArray('StoredEnergy'):
            energy_array = point_data.GetArray('StoredEnergy')
            self.stored_energy = numpy_support.vtk_to_numpy(energy_array).flatten()
        else:
            print("StoredEnergy not found, will calculate from KAM")
            self.stored_energy = np.zeros(self.n_cells)
            self._set_deformation_energy()
        
        print(f"Loaded microstructure with {len(self.quaternions)} unique orientations")
    
    def _initialize_ca_state(self):
        """Initialize cellular automaton state arrays"""
        # CA state
        self.consumption_rate = np.zeros(self.n_cells)
        self.consumed_fraction = np.zeros(self.n_cells)
        self.growing_neighbor = -np.ones(self.n_cells, dtype=int)
        
        # Tracking
        self.boundary_cells = set()
        self.is_recrystallized = np.zeros(self.n_cells, dtype=bool)
        
        # Cache for misorientation calculations
        self._misorientation_cache = {}
        
        # Identify initial boundary cells
        self._identify_boundary_cells()
        
        print(f"Initial boundary cells: {len(self.boundary_cells)}")
    
    def euler_to_quaternion(self, phi1, Phi, phi2):
        """Convert Euler angles (degrees) to quaternion"""
        # Convert to radians
        phi1 = np.radians(phi1)
        Phi = np.radians(Phi)
        phi2 = np.radians(phi2)
        
        # Calculate quaternion components
        c1 = np.cos(phi1/2)
        c2 = np.cos(Phi/2)
        c3 = np.cos(phi2/2)
        s1 = np.sin(phi1/2)
        s2 = np.sin(Phi/2)
        s3 = np.sin(phi2/2)
        
        w = c1*c2*c3 + s1*s2*s3
        x = c1*c2*s3 - s1*s2*c3
        y = c1*s2*c3 + s1*c2*s3
        z = s1*c2*c3 - c1*s2*s3
        
        # Normalize
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        return (w/norm, x/norm, y/norm, z/norm)
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return (w, x, y, z)
    
    def quaternion_conjugate(self, q):
        """Get conjugate of quaternion"""
        w, x, y, z = q
        return (w, -x, -y, -z)
    
    def _cell_to_indices(self, cell_idx: int) -> Tuple[int, int, int]:
        """Convert flat cell index to 3D indices"""
        i = cell_idx // (self.ny * self.nz)
        j = (cell_idx % (self.ny * self.nz)) // self.nz
        k = cell_idx % self.nz
        return i, j, k
    
    def _indices_to_cell(self, i: int, j: int, k: int) -> int:
        """Convert 3D indices to flat cell index"""
        return i * self.ny * self.nz + j * self.nz + k
    
    def _get_neighbors(self, cell_idx: int) -> List[int]:
        """Get Von Neumann neighbors (6-connectivity in 3D)"""
        i, j, k = self._cell_to_indices(cell_idx)
        neighbors = []
        
        for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            ni = i + di
            nj = j + dj
            nk = k + dk
            if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                neighbors.append(self._indices_to_cell(ni, nj, nk))
        
        return neighbors
    
    def _get_all_neighbors(self, cell_idx: int) -> List[int]:
        """Get Moore neighbors (26-connectivity in 3D)"""
        i, j, k = self._cell_to_indices(cell_idx)
        neighbors = []
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    nk = k + dk
                    if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                        neighbors.append(self._indices_to_cell(ni, nj, nk))
        
        return neighbors
    
    def _calculate_misorientation_quaternion(self, ori_id1: int, ori_id2: int) -> float:
        """Calculate misorientation between two orientations using quaternions with caching"""
        if ori_id1 == ori_id2:
            return 0.0
        
        # Check cache
        key = (min(ori_id1, ori_id2), max(ori_id1, ori_id2))
        if key in self._misorientation_cache:
            return self._misorientation_cache[key]
        
        q1 = self.quaternions.get(ori_id1, (1, 0, 0, 0))
        q2 = self.quaternions.get(ori_id2, (1, 0, 0, 0))
        
        # Calculate misorientation quaternion: q_mis = q1^(-1) * q2
        q1_conj = self.quaternion_conjugate(q1)
        q_mis = self.quaternion_multiply(q1_conj, q2)
        
        # Extract misorientation angle
        w = min(max(q_mis[0], -1.0), 1.0)  # Clamp to avoid numerical errors
        angle_rad = 2.0 * np.arccos(abs(w))
        angle_deg = np.degrees(angle_rad)
        
        # Apply crystal symmetry (simplified - for cubic symmetry)
        result = min(angle_deg, 62.8)
        
        # Cache result
        self._misorientation_cache[key] = result
        return result
    
    def _calculate_kam(self):
        """Calculate Kernel Average Misorientation for each cell"""
        print("Calculating KAM...")
        for cell_idx in range(self.n_cells):
            if cell_idx % 100000 == 0:
                print(f"  Progress: {cell_idx/self.n_cells*100:.1f}%")
            
            neighbors = self._get_all_neighbors(cell_idx)
            
            if len(neighbors) == 0:
                self.kam[cell_idx] = 0.0
                continue
            
            total_misorientation = 0.0
            n_boundaries = 0
            
            for nbr in neighbors:
                mis = self._calculate_misorientation_quaternion(
                    self.orientation_id[cell_idx],
                    self.orientation_id[nbr]
                )
                
                if mis > self.min_misorientation:
                    total_misorientation += mis
                    n_boundaries += 1
            
            if n_boundaries > 0:
                self.kam[cell_idx] = total_misorientation / n_boundaries
            else:
                self.kam[cell_idx] = 0.0
    
    def _set_deformation_energy(self):
        """Set stored energy based on KAM"""
        # Base stored energy from KAM
        for cell_idx in range(self.n_cells):
            # Exponential relationship with KAM
            normalized_kam = self.kam[cell_idx] / self.max_kam_degrees
            self.stored_energy[cell_idx] = self.stored_energy_factor * (normalized_kam ** 1.5)
    
    def _identify_boundary_cells(self):
        """Identify cells at boundaries"""
        self.boundary_cells.clear()
        
        for cell_idx in range(self.n_cells):
            neighbors = self._get_neighbors(cell_idx)
            
            for nbr in neighbors:
                if self.orientation_id[cell_idx] != self.orientation_id[nbr]:
                    mis = self._calculate_misorientation_quaternion(
                        self.orientation_id[cell_idx],
                        self.orientation_id[nbr]
                    )
                    if mis > self.min_misorientation:
                        self.boundary_cells.add(cell_idx)
                        break
    
    def _get_boundary_energy(self, misorientation: float) -> float:
        """Read-Shockley model for boundary energy"""
        if misorientation < self.min_misorientation:
            return 0.0
        
        if misorientation >= self.hagb_threshold:
            return self.grain_boundary_energy
        
        # Read-Shockley equation
        theta_m = self.hagb_threshold * np.pi / 180
        theta = misorientation * np.pi / 180
        
        return self.grain_boundary_energy * (theta / theta_m) * (1 - np.log(theta / theta_m))
    
    def _get_mobility(self, misorientation: float) -> float:
        """Get boundary mobility with misorientation dependence"""
        if misorientation < self.min_misorientation:
            return 0.0
        
        # Base mobility with Arrhenius temperature dependence
        M0 = self.mobility_pre_exponential * np.exp(-self.activation_energy / (self.R * self.temperature))
        
        if misorientation >= self.hagb_threshold:
            return M0
        else:
            # Reduced mobility for low angle boundaries
            return M0 * (misorientation / self.hagb_threshold) ** 2
    
    def calculate_reorientation_rates(self) -> float:
        """Calculate growth rates based on energy minimization"""
        max_rate = 0.0
        mean_stored_energy = np.mean(self.stored_energy)
        
        # Reset
        self.consumption_rate.fill(0.0)
        self.growing_neighbor.fill(-1)
        
        for cell_idx in self.boundary_cells:
            current_ori = self.orientation_id[cell_idx]
            
            # Calculate current cell's boundary energy per unit area
            current_boundary_energy = 0.0
            for nbr in self._get_all_neighbors(cell_idx):
                mis = self._calculate_misorientation_quaternion(current_ori, self.orientation_id[nbr])
                current_boundary_energy += self._get_boundary_energy(mis)
            
            # Normalize boundary energy by number of neighbors
            current_boundary_energy /= len(self._get_all_neighbors(cell_idx))
            
            # Total energy density (J/m³)
            current_total_energy = current_boundary_energy / self.dx + self.stored_energy[cell_idx]
            
            best_rate = 0.0
            best_neighbor = -1
            
            # Try each neighbor's orientation
            neighbors = self._get_neighbors(cell_idx)
            
            for nbr_idx in neighbors:
                nbr_ori = self.orientation_id[nbr_idx]
                if nbr_ori == current_ori:
                    continue
                
                # Calculate energy if this cell adopts neighbor's orientation
                new_boundary_energy = 0.0
                for other_nbr in self._get_all_neighbors(cell_idx):
                    mis = self._calculate_misorientation_quaternion(nbr_ori, self.orientation_id[other_nbr])
                    new_boundary_energy += self._get_boundary_energy(mis)
                
                # Normalize
                new_boundary_energy /= len(self._get_all_neighbors(cell_idx))
                
                # New total energy density
                new_total_energy = new_boundary_energy / self.dx + self.stored_energy[nbr_idx]
                
                # Energy decrease is the driving force (J/m³)
                energy_decrease = current_total_energy - new_total_energy
                
                # Growth advantage for low stored energy regions
                if self.stored_energy[nbr_idx] < 0.3 * mean_stored_energy:
                    energy_decrease = max(energy_decrease, self.stored_energy[cell_idx] * 0.5)
                
                # Only proceed if energy decreases
                if energy_decrease <= 0:
                    continue
                
                # Get mobility of the boundary
                boundary_mis = self._calculate_misorientation_quaternion(current_ori, nbr_ori)
                mobility = self._get_mobility(boundary_mis)
                
                # Rate proportional to mobility and driving pressure
                velocity = mobility * energy_decrease
                rate = velocity / self.dx * self.beta_factor
                
                if rate > best_rate:
                    best_rate = rate
                    best_neighbor = nbr_idx
            
            self.consumption_rate[cell_idx] = best_rate
            self.growing_neighbor[cell_idx] = best_neighbor
            max_rate = max(max_rate, best_rate)
        
        return max_rate
    
    def update_microstructure(self, dt: float) -> int:
        """Update cells based on consumption"""
        cells_to_switch = []
        
        # Update consumed fractions
        for cell_idx in self.boundary_cells:
            if self.consumption_rate[cell_idx] > 0:
                self.consumed_fraction[cell_idx] += self.consumption_rate[cell_idx] * dt
                
                if self.consumed_fraction[cell_idx] >= 1.0:
                    cells_to_switch.append(cell_idx)
        
        # Switch cells
        n_switched = 0
        for cell_idx in cells_to_switch:
            nbr_idx = int(self.growing_neighbor[cell_idx])
            if nbr_idx >= 0:
                # Copy orientation
                self.orientation_id[cell_idx] = self.orientation_id[nbr_idx]
                
                # Update stored energy
                self.stored_energy[cell_idx] = self.stored_energy[nbr_idx]
                
                # Track recrystallization
                energy_threshold = 0.2 * np.mean(self.stored_energy)
                if self.stored_energy[cell_idx] < energy_threshold:
                    self.is_recrystallized[cell_idx] = True
                
                # Reset
                self.consumed_fraction[cell_idx] = 0.0
                n_switched += 1
        
        # Update boundaries
        if n_switched > 0:
            self._identify_boundary_cells()
            # Update KAM very rarely (expensive operation)
            if self.step_count % 5000 == 0:
                self._calculate_kam()
        
        return n_switched
    
    def simulate_step(self) -> Tuple[float, int]:
        """Perform one simulation step"""
        max_rate = self.calculate_reorientation_rates()
        
        if max_rate < 1e-15:
            return 0.0, 0
        
        # Adaptive time step
        dt = min(0.5 / max_rate, 1.0)
        
        n_switched = self.update_microstructure(dt)
        
        self.total_time += dt
        self.step_count += 1
        
        return dt, n_switched
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        unique_oris = len(np.unique(self.orientation_id))
        rex_fraction = np.mean(self.is_recrystallized)
        mean_stored_energy = np.mean(self.stored_energy)
        mean_kam = np.mean(self.kam)
        
        return {
            'n_orientations': unique_oris,
            'rex_fraction': rex_fraction,
            'mean_stored_energy': mean_stored_energy,
            'mean_kam': mean_kam,
            'n_boundaries': len(self.boundary_cells),
            'time': self.total_time,
            'step': self.step_count
        }
    
    def save_state_to_vti(self, filename: str):
        """Save current state to VTI file"""
        # Create image data
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(self.nx, self.ny, self.nz)
        image_data.SetSpacing(self.dx*1e6, self.dx*1e6, self.dx*1e6)  # Convert back to micrometers
        
        # Add arrays
        # Orientation ID
        ori_array = numpy_support.numpy_to_vtk(self.orientation_id)
        ori_array.SetName("OrientationID")
        image_data.GetPointData().AddArray(ori_array)
        
        # KAM
        kam_array = numpy_support.numpy_to_vtk(self.kam)
        kam_array.SetName("KAM")
        image_data.GetPointData().AddArray(kam_array)
        
        # Stored Energy
        energy_array = numpy_support.numpy_to_vtk(self.stored_energy)
        energy_array.SetName("StoredEnergy")
        image_data.GetPointData().AddArray(energy_array)
        
        # Recrystallized state
        rex_array = numpy_support.numpy_to_vtk(self.is_recrystallized.astype(int))
        rex_array.SetName("Recrystallized")
        image_data.GetPointData().AddArray(rex_array)
        
        # Write to file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(image_data)
        writer.Write()
    
    def run_simulation(self):
        """Run the complete simulation"""
        print(f"\nStarting simulation...")
        print(f"Max time: {self.max_time}s")
        print(f"Max steps: {self.max_steps}")
        print(f"Save interval: {self.save_interval} steps")
        
        # Save initial state
        initial_file = os.path.join(self.output_dir, f"step_000000.vti")
        self.save_state_to_vti(initial_file)
        print(f"Saved initial state to {initial_file}")
        
        # Statistics history
        history = {
            'time': [0.0],
            'step': [0],
            'rex_fraction': [0.0],
            'n_orientations': [len(self.quaternions)],
            'mean_stored_energy': [np.mean(self.stored_energy)]
        }
        
        # Run simulation
        last_save_step = 0
        start_time = time.time()
        
        while self.total_time < self.max_time and self.step_count < self.max_steps:
            dt, n_switched = self.simulate_step()
            
            if dt == 0:
                print("\nSteady state reached")
                break
            
            # Save state at intervals
            if self.step_count - last_save_step >= self.save_interval:
                # Save VTI file
                step_file = os.path.join(self.output_dir, f"step_{self.step_count:06d}.vti")
                self.save_state_to_vti(step_file)
                
                # Get statistics
                stats = self.get_statistics()
                history['time'].append(stats['time'])
                history['step'].append(stats['step'])
                history['rex_fraction'].append(stats['rex_fraction'])
                history['n_orientations'].append(stats['n_orientations'])
                history['mean_stored_energy'].append(stats['mean_stored_energy'])
                
                # Print progress
                elapsed = time.time() - start_time
                print(f"Step {self.step_count}: t={self.total_time:.3f}s, "
                      f"Rex={stats['rex_fraction']:.3f}, "
                      f"Grains={stats['n_orientations']}, "
                      f"Elapsed={elapsed:.1f}s")
                
                last_save_step = self.step_count
        
        # Save final state
        final_file = os.path.join(self.output_dir, f"step_{self.step_count:06d}_final.vti")
        self.save_state_to_vti(final_file)
        
        # Save history to file
        history_file = os.path.join(self.output_dir, "simulation_history.npz")
        np.savez(history_file, **history)
        
        print(f"\nSimulation completed:")
        print(f"  Total steps: {self.step_count}")
        print(f"  Final time: {self.total_time:.3f}s")
        print(f"  Final rex fraction: {history['rex_fraction'][-1]:.3f}")
        print(f"  Output saved to: {self.output_dir}")
        
        return history


def create_example_yaml():
    """Create an example YAML configuration file"""
    config = {
        'initial_microstructure': {
            'path': 'in_microstructure.vti',
            'description': 'Path to initial microstructure VTI file'
        },
        'material': {
            'grain_boundary_energy': 0.5,  # J/m^2
            'mobility_pre_exponential': 1e-3,  # m^4/(J·s)
            'activation_energy': 140e3,  # J/mol
            'temperature': 873  # K
        },
        'thresholds': {
            'min_misorientation': 1.0,  # degrees
            'lagb_threshold': 2.0,  # degrees
            'hagb_threshold': 15.0  # degrees
        },
        'energy': {
            'stored_energy_factor': 1e7,  # J/m^3
            'max_kam_degrees': 15.0  # degrees
        },
        'kinetics': {
            'beta_factor': 0.5
        },
        'simulation': {
            'max_time': 20.0,  # seconds
            'max_steps': 50000,
            'save_interval': 100,  # steps
            'output_directory': 'output_ca_simulation'
        }
    }
    
    with open('condition.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("Created example configuration file: condition.yaml")


def create_example_vti():
    """Create an example VTI file for testing"""
    # Create a small 3D grid
    nx, ny, nz = 50, 50, 50
    
    # Create image data
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)
    image_data.SetSpacing(1.0, 1.0, 1.0)  # micrometers
    
    # Create orientation data (simple grain structure)
    n_cells = nx * ny * nz
    orientation_id = np.zeros(n_cells, dtype=int)
    
    # Create a few grains
    n_grains = 10
    grain_centers = np.random.rand(n_grains, 3)
    grain_centers[:, 0] *= nx
    grain_centers[:, 1] *= ny
    grain_centers[:, 2] *= nz
    
    for idx in range(n_cells):
        i = idx // (ny * nz)
        j = (idx % (ny * nz)) // nz
        k = idx % nz
        
        # Find nearest grain center
        min_dist = float('inf')
        nearest_grain = 0
        
        for g in range(n_grains):
            dx = i - grain_centers[g, 0]
            dy = j - grain_centers[g, 1]
            dz = k - grain_centers[g, 2]
            dist = dx*dx + dy*dy + dz*dz
            
            if dist < min_dist:
                min_dist = dist
                nearest_grain = g
        
        orientation_id[idx] = nearest_grain
    
    # Add orientation array
    ori_array = numpy_support.numpy_to_vtk(orientation_id)
    ori_array.SetName("OrientationID")
    image_data.GetPointData().AddArray(ori_array)
    
    # Add Euler angles
    euler_angles = np.zeros((n_cells, 3))
    for g in range(n_grains):
        grain_cells = np.where(orientation_id == g)[0]
        phi1 = np.random.uniform(0, 360)
        Phi = np.random.uniform(0, 180)
        phi2 = np.random.uniform(0, 360)
        euler_angles[grain_cells] = [phi1, Phi, phi2]
    
    euler_array = numpy_support.numpy_to_vtk(euler_angles)
    euler_array.SetName("EulerAngles")
    euler_array.SetNumberOfComponents(3)
    image_data.GetPointData().AddArray(euler_array)
    
    # Write to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName("in_microstructure.vti")
    writer.SetInputData(image_data)
    writer.Write()
    
    print("Created example VTI file: in_microstructure.vti")


if __name__ == "__main__":
    # Create example files if they don't exist
    if not os.path.exists('condition.yaml'):
        create_example_yaml()
    
    if not os.path.exists('in_microstructure.vti'):
        create_example_vti()
    
    # Run simulation
    ca = CellularAutomatonRexGG3D('condition.yaml')
    history = ca.run_simulation()
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['time'], history['rex_fraction'])
    plt.xlabel('Time (s)')
    plt.ylabel('Recrystallized Fraction')
    plt.title('Recrystallization Kinetics')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(history['time'], history['n_orientations'])
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Grains')
    plt.title('Grain Count Evolution')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.semilogy(history['time'], history['mean_stored_energy'])
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Stored Energy (J/m³)')
    plt.title('Stored Energy Evolution')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(history['step'], history['time'])
    plt.xlabel('Simulation Step')
    plt.ylabel('Time (s)')
    plt.title('Time vs Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ca.output_dir, 'simulation_summary.png'))
    plt.show()