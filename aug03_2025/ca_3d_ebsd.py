import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
import yaml
import vtk
from vtk.util import numpy_support
import os
import time
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import pandas as pd

class CellularAutomatonRexGG3D_EBSD:
    """
    3D Cellular Automaton for Recrystallization and Grain Growth
    Based ONLY on Euler angles - no deformation data considered
    Works with EBSD data format (x, y, z, phi1, PHI, phi2, phase)
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
        
        # Stored energy parameters (uniform or KAM-based only)
        self.stored_energy_uniform = self.config['energy']['stored_energy_uniform']
        self.use_kam_energy = self.config['energy']['use_kam_energy']
        self.kam_energy_factor = self.config['energy']['kam_energy_factor']
        self.max_kam_degrees = self.config['energy']['max_kam_degrees']
        
        # Kinetic factor
        self.beta_factor = self.config['kinetics']['beta_factor']
        
        # Simulation parameters
        self.max_time = self.config['simulation']['max_time']
        self.max_steps = self.config['simulation']['max_steps']
        self.save_interval = self.config['simulation']['save_interval']
        self.output_dir = self.config['simulation']['output_directory']
        
        # Grain detection parameters
        self.grain_detection_threshold = self.config.get('grain_detection', {}).get('misorientation_threshold', 5.0)
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize cache before loading microstructure
        self._misorientation_cache = {}
        
        # Load initial microstructure based on data type
        self.data_source = self.config['initial_microstructure']['type']
        if self.data_source == 'ebsd':
            self.ebsd_data_path = self.config['initial_microstructure']['path']
            self._load_initial_microstructure_ebsd()
        elif self.data_source == 'damask':
            self.initial_microstructure_path = self.config['initial_microstructure']['path']
            self._load_initial_microstructure_damask()
        else:
            raise ValueError(f"Unknown data source type: {self.data_source}")
        
        # Initialize cellular automaton state
        self._initialize_ca_state()
        
        # Statistics tracking
        self.total_time = 0.0
        self.step_count = 0
        
        print(f"Initialized 3D CA with grid size: {self.nx} x {self.ny} x {self.nz}")
        print(f"Total cells: {self.n_cells}")
        print("Note: Using ONLY Euler angles for CA simulation (no deformation data)")
    
    def _load_initial_microstructure_ebsd(self):
        """Load initial microstructure from EBSD data file"""
        print(f"Loading EBSD data from: {self.ebsd_data_path}")
        
        # Read EBSD data
        # Assume CSV format with columns: x, y, z, phi1, PHI, phi2, phase
        ebsd_df = pd.read_csv(self.ebsd_data_path)
        
        # Validate columns
        required_cols = ['x', 'y', 'z', 'phi1', 'PHI', 'phi2', 'phase']
        if not all(col in ebsd_df.columns for col in required_cols):
            raise ValueError(f"EBSD data must contain columns: {required_cols}")
        
        # Extract grid dimensions
        x_unique = np.sort(ebsd_df['x'].unique())
        y_unique = np.sort(ebsd_df['y'].unique())
        z_unique = np.sort(ebsd_df['z'].unique())
        
        self.nx = len(x_unique)
        self.ny = len(y_unique)
        self.nz = len(z_unique)
        self.n_cells = self.nx * self.ny * self.nz
        
        # Calculate cell size (assuming uniform spacing)
        if self.nx > 1:
            self.dx = x_unique[1] - x_unique[0]
        else:
            self.dx = 1.0  # Default if only one layer
        
        print(f"Grid: {self.nx} x {self.ny} x {self.nz} cells")
        print(f"Cell size: {self.dx:.6f} (units from EBSD data)")
        
        # Create mapping from (x,y,z) to indices
        x_to_i = {x: i for i, x in enumerate(x_unique)}
        y_to_j = {y: j for j, y in enumerate(y_unique)}
        z_to_k = {z: k for k, z in enumerate(z_unique)}
        
        # Initialize arrays
        self.euler_angles = np.zeros((self.n_cells, 3))  # phi1, PHI, phi2
        self.phase_id = np.ones(self.n_cells, dtype=int)  # Default phase = 1
        
        # Fill arrays from EBSD data
        for _, row in ebsd_df.iterrows():
            i = x_to_i[row['x']]
            j = y_to_j[row['y']]
            k = z_to_k[row['z']]
            cell_idx = self._indices_to_cell(i, j, k)
            
            self.euler_angles[cell_idx] = [row['phi1'], row['PHI'], row['phi2']]
            self.phase_id[cell_idx] = int(row['phase'])
        
        # Convert Euler angles to quaternions and detect grains
        print("Converting Euler angles to quaternions...")
        quaternions_data = np.zeros((self.n_cells, 4))
        for i in range(self.n_cells):
            quaternions_data[i] = self.euler_to_quaternion(
                self.euler_angles[i, 0],
                self.euler_angles[i, 1],
                self.euler_angles[i, 2]
            )
        
        # Detect grains based on orientation similarity
        print("Detecting grains from orientation data...")
        self.orientation_id, self.quaternions = self._detect_grains_from_quaternions(quaternions_data)
        
        # Calculate KAM
        print("Calculating KAM...")
        self.kam = np.zeros(self.n_cells)
        self._calculate_kam()
        
        # Set initial stored energy (uniform or KAM-based only)
        self._set_initial_stored_energy()
        
        print(f"Detected {len(self.quaternions)} grains")
        print(f"Mean stored energy: {np.mean(self.stored_energy):.2e} J/m³")
    
    def _load_initial_microstructure_damask(self):
        """Load initial microstructure from DAMASK VTI file - ONLY orientations"""
        print(f"Loading DAMASK output from: {self.initial_microstructure_path}")
        print("Note: Only extracting orientation data, ignoring deformation data")
        
        # Read VTI file
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(self.initial_microstructure_path)
        reader.Update()
        
        # Get the image data
        image_data = reader.GetOutput()
        
        # Get dimensions (note: VTI has points, we need cells)
        dims = image_data.GetDimensions()
        self.nx = dims[0] - 1  # cells = points - 1
        self.ny = dims[1] - 1
        self.nz = dims[2] - 1
        self.n_cells = self.nx * self.ny * self.nz
        
        # Get spacing (cell size)
        spacing = image_data.GetSpacing()
        self.dx = spacing[0]  # Already in meters for DAMASK output
        
        print(f"Grid: {self.nx} x {self.ny} x {self.nz} cells")
        print(f"Cell size: {self.dx*1e6:.1f} μm")
        
        # Get cell data (DAMASK stores data on cells, not points)
        cell_data = image_data.GetCellData()
        
        # Extract ONLY quaternions from 'phase/mechanical/O / 1'
        quat_array = cell_data.GetArray('phase/mechanical/O / 1')
        if quat_array is None:
            raise ValueError("Quaternion array 'phase/mechanical/O / 1' not found in VTI file")
        
        quaternions_data = numpy_support.vtk_to_numpy(quat_array)
        print(f"Loaded quaternions with shape: {quaternions_data.shape}")
        
        # Detect grains based on orientation similarity
        print("Detecting grains from orientation data...")
        self.orientation_id, self.quaternions = self._detect_grains_from_quaternions(quaternions_data)
        
        # Calculate KAM
        print("Calculating KAM...")
        self.kam = np.zeros(self.n_cells)
        self._calculate_kam()
        
        # Set stored energy (uniform or KAM-based only)
        self._set_initial_stored_energy()
        
        print(f"Detected {len(self.quaternions)} grains")
        print(f"Mean stored energy: {np.mean(self.stored_energy):.2e} J/m³")
    
    def _set_initial_stored_energy(self):
        """Set initial stored energy - NO deformation data used"""
        self.stored_energy = np.zeros(self.n_cells)
        
        if self.use_kam_energy:
            # Use KAM-based stored energy
            print("Setting stored energy based on KAM values")
            normalized_kam = self.kam / self.max_kam_degrees
            self.stored_energy = self.kam_energy_factor * (normalized_kam ** 1.5)
        else:
            # Use uniform stored energy
            print("Setting uniform stored energy")
            self.stored_energy = np.full(self.n_cells, self.stored_energy_uniform)
        
        # Add small random noise to avoid perfect symmetry
        noise = np.random.uniform(0.95, 1.05, self.n_cells)
        self.stored_energy *= noise
        
        print(f"Stored energy range: {np.min(self.stored_energy):.2e} - {np.max(self.stored_energy):.2e} J/m³")
    
    def save_as_ebsd_format(self, filename: str):
        """Save current state in EBSD format"""
        data = []
        
        for cell_idx in range(self.n_cells):
            i, j, k = self._cell_to_indices(cell_idx)
            
            # Get position
            x = i * self.dx
            y = j * self.dx
            z = k * self.dx
            
            # Get current orientation
            grain_id = self.orientation_id[cell_idx]
            quat = self.quaternions[grain_id]
            
            # Convert quaternion back to Euler angles
            phi1, PHI, phi2 = self.quaternion_to_euler(quat)
            
            # Get phase (default to 1 if not set)
            phase = self.phase_id[cell_idx] if hasattr(self, 'phase_id') else 1
            
            data.append({
                'x': x,
                'y': y,
                'z': z,
                'phi1': phi1,
                'PHI': PHI,
                'phi2': phi2,
                'phase': phase,
                'grain_id': grain_id,
                'kam': self.kam[cell_idx],
                'stored_energy': self.stored_energy[cell_idx],
                'is_recrystallized': int(self.is_recrystallized[cell_idx])
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Saved EBSD format data to: {filename}")
    
    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (Bunge convention)"""
        w, x, y, z = q
        
        # Ensure quaternion is normalized
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix elements
        r11 = w*w + x*x - y*y - z*z
        r12 = 2*(x*y - w*z)
        r13 = 2*(x*z + w*y)
        r21 = 2*(x*y + w*z)
        r22 = w*w - x*x + y*y - z*z
        r23 = 2*(y*z - w*x)
        r31 = 2*(x*z - w*y)
        r32 = 2*(y*z + w*x)
        r33 = w*w - x*x - y*y + z*z
        
        # Extract Euler angles (Bunge convention)
        if abs(r33) < 0.999:
            PHI = np.arccos(r33)
            phi1 = np.arctan2(r31, -r32)
            phi2 = np.arctan2(r13, r23)
        else:
            # Special case when PHI ~ 0 or 180
            PHI = np.arccos(np.clip(r33, -1, 1))
            phi1 = np.arctan2(r12, r11)
            phi2 = 0.0
        
        # Convert to degrees
        phi1 = np.degrees(phi1) % 360
        PHI = np.degrees(PHI)
        phi2 = np.degrees(phi2) % 360
        
        return phi1, PHI, phi2
    
    @staticmethod
    def convert_damask_vti_to_ebsd(vti_file: str, output_csv: str, cell_size: float = None):
        """
        Convert DAMASK VTI file to EBSD format CSV - ONLY orientation data
        
        Parameters:
        -----------
        vti_file : str
            Path to DAMASK VTI file
        output_csv : str
            Path to output CSV file in EBSD format
        cell_size : float, optional
            Cell size in desired units. If None, will use VTI spacing
        """
        print(f"Converting DAMASK VTI to EBSD format (orientations only)...")
        print(f"Input: {vti_file}")
        print(f"Output: {output_csv}")
        
        # Read VTI file
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(vti_file)
        reader.Update()
        
        # Get the image data
        image_data = reader.GetOutput()
        
        # Get dimensions
        dims = image_data.GetDimensions()
        nx = dims[0] - 1  # cells = points - 1
        ny = dims[1] - 1
        nz = dims[2] - 1
        n_cells = nx * ny * nz
        
        # Get spacing
        spacing = image_data.GetSpacing()
        dx = cell_size if cell_size is not None else spacing[0]
        
        print(f"Grid: {nx} x {ny} x {nz} cells")
        print(f"Cell size: {dx}")
        
        # Get cell data
        cell_data = image_data.GetCellData()
        
        # Extract quaternions
        quat_array = cell_data.GetArray('phase/mechanical/O / 1')
        if quat_array is None:
            raise ValueError("Quaternion array 'phase/mechanical/O / 1' not found in VTI file")
        
        quaternions_data = numpy_support.vtk_to_numpy(quat_array)
        
        # Prepare EBSD format data
        data = []
        
        for cell_idx in range(n_cells):
            # Calculate indices
            i = cell_idx // (ny * nz)
            j = (cell_idx % (ny * nz)) // nz
            k = cell_idx % nz
            
            # Calculate position
            x = i * dx
            y = j * dx
            z = k * dx
            
            # Get quaternion
            quat = quaternions_data[cell_idx]
            
            # Convert to Euler angles
            ca = CellularAutomatonRexGG3D_EBSD.__new__(CellularAutomatonRexGG3D_EBSD)
            phi1, PHI, phi2 = ca.quaternion_to_euler(quat)
            
            # Default phase = 1
            phase = 1
            
            data.append({
                'x': x,
                'y': y,
                'z': z,
                'phi1': phi1,
                'PHI': PHI,
                'phi2': phi2,
                'phase': phase
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        
        print(f"Conversion complete. Saved {n_cells} cells to {output_csv}")
        print("Note: Only orientation data was extracted (no deformation data)")
        
        return df
    
    def _detect_grains_from_quaternions(self, quaternions_data):
        """Detect grains by clustering cells with similar orientations"""
        n_cells = quaternions_data.shape[0]
        
        # Method 1: Simple misorientation-based clustering
        orientation_id = np.zeros(n_cells, dtype=int)
        unique_quaternions = {}
        current_grain_id = 0
        
        # Process cells and group by orientation
        for cell_idx in range(n_cells):
            q_cell = quaternions_data[cell_idx]
            
            # Check if this orientation is similar to existing grains
            assigned = False
            for grain_id, q_grain in unique_quaternions.items():
                misorientation = self._calculate_misorientation_from_quaternions(q_cell, q_grain)
                if misorientation < self.grain_detection_threshold:
                    orientation_id[cell_idx] = grain_id
                    assigned = True
                    break
            
            # If not similar to any existing grain, create new grain
            if not assigned:
                orientation_id[cell_idx] = current_grain_id
                unique_quaternions[current_grain_id] = q_cell.copy()
                current_grain_id += 1
        
        # Method 2: Alternative - spatial clustering with orientation
        if len(unique_quaternions) == 1:
            print("Only one orientation detected. Attempting spatial clustering...")
            orientation_id, unique_quaternions = self._spatial_grain_detection(quaternions_data)
        
        return orientation_id, unique_quaternions
    
    def _spatial_grain_detection(self, quaternions_data):
        """Detect grains using spatial clustering combined with orientation"""
        # Create a feature vector combining position and orientation
        features = []
        
        for cell_idx in range(self.n_cells):
            i, j, k = self._cell_to_indices(cell_idx)
            # Normalize spatial coordinates
            x = i / self.nx
            y = j / self.ny
            z = k / self.nz
            
            # Get quaternion
            q = quaternions_data[cell_idx]
            
            # Combine spatial and orientation info
            # Weight orientation more heavily
            spatial_weight = 0.1
            orientation_weight = 1.0
            
            feature = np.concatenate([
                [x * spatial_weight, y * spatial_weight, z * spatial_weight],
                q * orientation_weight
            ])
            features.append(feature)
        
        features = np.array(features)
        
        # Use DBSCAN clustering
        clustering = DBSCAN(eps=0.15, min_samples=10).fit(features)
        labels = clustering.labels_
        
        # Handle noise points (-1 label)
        if -1 in labels:
            # Assign noise points to nearest cluster
            noise_mask = labels == -1
            non_noise_mask = ~noise_mask
            
            if np.any(non_noise_mask):
                tree = cKDTree(features[non_noise_mask])
                for idx in np.where(noise_mask)[0]:
                    _, nearest_idx = tree.query(features[idx])
                    labels[idx] = labels[np.where(non_noise_mask)[0][nearest_idx]]
        
        # Create unique quaternions dictionary
        unique_quaternions = {}
        for grain_id in np.unique(labels):
            if grain_id >= 0:
                grain_cells = np.where(labels == grain_id)[0]
                # Use average quaternion for the grain
                avg_q = np.mean(quaternions_data[grain_cells], axis=0)
                # Normalize
                avg_q = avg_q / np.linalg.norm(avg_q)
                unique_quaternions[grain_id] = avg_q
        
        return labels, unique_quaternions
    
    def _calculate_misorientation_from_quaternions(self, q1, q2):
        """Calculate misorientation angle between two quaternions"""
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate dot product
        dot = np.abs(np.dot(q1, q2))
        dot = min(dot, 1.0)  # Clamp to avoid numerical errors
        
        # Calculate misorientation angle
        angle_rad = 2.0 * np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
        
        # Apply cubic symmetry
        return min(angle_deg, 62.8)
    
    def _initialize_ca_state(self):
        """Initialize cellular automaton state arrays"""
        # CA state
        self.consumption_rate = np.zeros(self.n_cells)
        self.consumed_fraction = np.zeros(self.n_cells)
        self.growing_neighbor = -np.ones(self.n_cells, dtype=int)
        
        # Tracking
        self.boundary_cells = set()
        self.is_recrystallized = np.zeros(self.n_cells, dtype=bool)
        
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
        for cell_idx in range(self.n_cells):
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
                
                # For recrystallization: growth advantage for low stored energy regions
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
        """Save current state to VTI file compatible with DAMASK format"""
        # Create image data
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(self.nx+1, self.ny+1, self.nz+1)  # Points = cells + 1
        image_data.SetSpacing(self.dx, self.dx, self.dx)
        
        # Add arrays as cell data (matching DAMASK format)
        # Grain ID
        ori_array = numpy_support.numpy_to_vtk(self.orientation_id)
        ori_array.SetName("GrainID")
        image_data.GetCellData().AddArray(ori_array)
        
        # Quaternions
        quat_data = np.zeros((self.n_cells, 4))
        for i in range(self.n_cells):
            grain_id = self.orientation_id[i]
            quat_data[i] = self.quaternions[grain_id]
        
        quat_array = numpy_support.numpy_to_vtk(quat_data)
        quat_array.SetName("phase/mechanical/O / 1")
        image_data.GetCellData().AddArray(quat_array)
        
        # KAM
        kam_array = numpy_support.numpy_to_vtk(self.kam)
        kam_array.SetName("KAM")
        image_data.GetCellData().AddArray(kam_array)
        
        # Stored Energy
        energy_array = numpy_support.numpy_to_vtk(self.stored_energy)
        energy_array.SetName("StoredEnergy")
        image_data.GetCellData().AddArray(energy_array)
        
        # Recrystallized state
        rex_array = numpy_support.numpy_to_vtk(self.is_recrystallized.astype(int))
        rex_array.SetName("Recrystallized")
        image_data.GetCellData().AddArray(rex_array)
        
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
        initial_vti = os.path.join(self.output_dir, f"step_000000.vti")
        self.save_state_to_vti(initial_vti)
        
        initial_ebsd = os.path.join(self.output_dir, f"step_000000_ebsd.csv")
        self.save_as_ebsd_format(initial_ebsd)
        
        print(f"Saved initial state to {initial_vti} and {initial_ebsd}")
        
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
                step_vti = os.path.join(self.output_dir, f"step_{self.step_count:06d}.vti")
                self.save_state_to_vti(step_vti)
                
                # Save EBSD format
                step_ebsd = os.path.join(self.output_dir, f"step_{self.step_count:06d}_ebsd.csv")
                self.save_as_ebsd_format(step_ebsd)
                
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
        final_vti = os.path.join(self.output_dir, f"step_{self.step_count:06d}_final.vti")
        self.save_state_to_vti(final_vti)
        
        final_ebsd = os.path.join(self.output_dir, f"step_{self.step_count:06d}_final_ebsd.csv")
        self.save_as_ebsd_format(final_ebsd)
        
        # Save history to file
        history_file = os.path.join(self.output_dir, "simulation_history.npz")
        np.savez(history_file, **history)
        
        print(f"\nSimulation completed:")
        print(f"  Total steps: {self.step_count}")
        print(f"  Final time: {self.total_time:.3f}s")
        print(f"  Final rex fraction: {history['rex_fraction'][-1]:.3f}")
        print(f"  Output saved to: {self.output_dir}")
        
        return history


def create_ebsd_yaml_config():
    """Create YAML configuration for EBSD data files"""
    config = {
        'initial_microstructure': {
            'type': 'ebsd',  # 'ebsd' or 'damask'
            'path': 'ebsd_data.csv',
            'description': 'Path to EBSD data CSV file'
        },
        'material': {
            'grain_boundary_energy': 0.5,  # J/m^2
            'mobility_pre_exponential': 1e-4,  # m^4/(J·s)
            'activation_energy': 140e3,  # J/mol
            'temperature': 873  # K
        },
        'thresholds': {
            'min_misorientation': 1.0,  # degrees
            'lagb_threshold': 2.0,  # degrees
            'hagb_threshold': 15.0  # degrees
        },
        'energy': {
            'stored_energy_uniform': 1e7,  # J/m^3 - uniform stored energy
            'use_kam_energy': True,  # If True, use KAM-based energy instead of uniform
            'kam_energy_factor': 5e7,  # J/m^3 - factor for KAM-based energy
            'max_kam_degrees': 15.0  # degrees - normalization for KAM
        },
        'kinetics': {
            'beta_factor': 0.5
        },
        'grain_detection': {
            'misorientation_threshold': 5.0  # degrees - for detecting grains from orientations
        },
        'simulation': {
            'max_time': 100.0,  # seconds
            'max_steps': 10000,
            'save_interval': 50,
            'output_directory': 'output_ebsd_ca_simulation'
        }
    }
    
    with open('config_ebsd.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("Created configuration file: config_ebsd.yaml")


def create_example_ebsd_data():
    """Create example EBSD data file for testing"""
    # Create a simple 10x10x5 grid with 3 grains
    data = []
    
    # Grid parameters
    nx, ny, nz = 10, 10, 5
    dx = 1.0  # 1 micron spacing
    
    # Define three grains with different orientations
    grain_orientations = [
        (0.0, 0.0, 0.0),      # Grain 1
        (45.0, 30.0, 15.0),   # Grain 2
        (90.0, 45.0, 30.0)    # Grain 3
    ]
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = i * dx
                y = j * dx
                z = k * dx
                
                # Assign grain based on position
                if i < nx/3:
                    grain = 0
                elif i < 2*nx/3:
                    grain = 1
                else:
                    grain = 2
                
                phi1, PHI, phi2 = grain_orientations[grain]
                
                # Add some random noise to orientations
                phi1 += np.random.normal(0, 2.0)
                PHI += np.random.normal(0, 2.0)
                phi2 += np.random.normal(0, 2.0)
                
                data.append({
                    'x': x,
                    'y': y,
                    'z': z,
                    'phi1': phi1,
                    'PHI': PHI,
                    'phi2': phi2,
                    'phase': 1
                })
    
    df = pd.DataFrame(data)
    df.to_csv('example_ebsd_data.csv', index=False)
    print("Created example EBSD data: example_ebsd_data.csv")


###########

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, KDTree
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_realistic_ebsd_data(nx=20, ny=20, nz=10, n_grains=10, dx=1.0, 
                               orientation_spread=2.0, save_path='realistic_ebsd_data.csv',
                               visualize=True):
    """
    Create realistic EBSD data with proper 3D grain shapes using Voronoi tessellation
    
    Parameters:
    -----------
    nx, ny, nz : int
        Grid dimensions
    n_grains : int
        Number of grains to create
    dx : float
        Grid spacing (microns)
    orientation_spread : float
        Standard deviation of orientation noise within grains (degrees)
    save_path : str
        Path to save the CSV file
    visualize : bool
        Whether to create visualization plots
    """
    
    print(f"Creating realistic EBSD data with {n_grains} grains in {nx}x{ny}x{nz} grid...")
    
    # Method 1: Voronoi tessellation for grain structure
    # Generate random seed points for grains
    np.random.seed(42)  # For reproducibility
    
    # Create seed points with some minimum distance between them
    seed_points = []
    min_distance = min(nx, ny, nz) / (n_grains**(1/3)) * 0.8
    
    while len(seed_points) < n_grains:
        # Generate random point
        new_point = np.array([
            np.random.uniform(0, nx),
            np.random.uniform(0, ny),
            np.random.uniform(0, nz)
        ])
        
        # Check minimum distance to existing points
        if len(seed_points) == 0:
            seed_points.append(new_point)
        else:
            distances = [np.linalg.norm(new_point - p) for p in seed_points]
            if min(distances) > min_distance:
                seed_points.append(new_point)
    
    seed_points = np.array(seed_points)
    
    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(seed_points)
    
    # Generate random orientations for each grain
    grain_orientations = []
    for i in range(n_grains):
        # Generate random Euler angles with good spread
        phi1 = np.random.uniform(0, 360)
        PHI = np.random.uniform(0, 180)
        phi2 = np.random.uniform(0, 360)
        grain_orientations.append((phi1, PHI, phi2))
    
    print(f"Generated grain orientations:")
    for i, (phi1, PHI, phi2) in enumerate(grain_orientations):
        print(f"  Grain {i}: ({phi1:.1f}°, {PHI:.1f}°, {phi2:.1f}°)")
    
    # Create grid and assign grains
    data = []
    grain_map = np.zeros((nx, ny, nz), dtype=int)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Find nearest seed point (grain center)
                point = np.array([i, j, k])
                _, grain_id = tree.query(point)
                grain_map[i, j, k] = grain_id
                
                # Get base orientation for this grain
                phi1, PHI, phi2 = grain_orientations[grain_id]
                
                # Add realistic orientation spread within grain
                # Orientation varies smoothly from grain center
                distance_to_center = np.linalg.norm(point - seed_points[grain_id])
                max_distance = np.sqrt(nx**2 + ny**2 + nz**2)
                
                # Add spatially correlated noise
                noise_factor = (distance_to_center / max_distance) * 2.0 + 1.0
                phi1 += np.random.normal(0, orientation_spread * noise_factor)
                PHI += np.random.normal(0, orientation_spread * noise_factor)
                phi2 += np.random.normal(0, orientation_spread * noise_factor)
                
                # Ensure angles are in valid ranges
                phi1 = phi1 % 360
                PHI = np.clip(PHI, 0, 180)
                phi2 = phi2 % 360
                
                # Store data
                data.append({
                    'x': i * dx,
                    'y': j * dx,
                    'z': k * dx,
                    'phi1': phi1,
                    'PHI': PHI,
                    'phi2': phi2,
                    'phase': 1,
                    'grain_id_true': grain_id  # Store true grain ID for verification
                })
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"\nSaved realistic EBSD data to: {save_path}")
    print(f"Total cells: {len(df)}")
    
    # Calculate grain statistics
    grain_sizes = np.bincount(grain_map.flatten())
    print(f"\nGrain size statistics:")
    print(f"  Mean size: {np.mean(grain_sizes):.1f} cells")
    print(f"  Min size: {np.min(grain_sizes)} cells")
    print(f"  Max size: {np.max(grain_sizes)} cells")
    
    if visualize:
        create_visualizations(grain_map, df, nx, ny, nz)
    
    return df, grain_map

def create_visualizations(grain_map, df, nx, ny, nz):
    """Create visualization plots for the grain structure"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D scatter plot colored by grain
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors = plt.cm.tab20(df['grain_id_true'] / df['grain_id_true'].max())
    ax1.scatter(df['x'], df['y'], df['z'], c=colors, s=20, alpha=0.6)
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Z (μm)')
    ax1.set_title('3D Grain Structure')
    
    # 2. XY slice at middle Z
    ax2 = fig.add_subplot(2, 3, 2)
    z_middle = nz // 2
    im2 = ax2.imshow(grain_map[:, :, z_middle].T, origin='lower', cmap='tab20')
    ax2.set_xlabel('X (cells)')
    ax2.set_ylabel('Y (cells)')
    ax2.set_title(f'XY Slice at Z={z_middle}')
    plt.colorbar(im2, ax=ax2, label='Grain ID')
    
    # 3. XZ slice at middle Y
    ax3 = fig.add_subplot(2, 3, 3)
    y_middle = ny // 2
    im3 = ax3.imshow(grain_map[:, y_middle, :].T, origin='lower', cmap='tab20', aspect='auto')
    ax3.set_xlabel('X (cells)')
    ax3.set_ylabel('Z (cells)')
    ax3.set_title(f'XZ Slice at Y={y_middle}')
    plt.colorbar(im3, ax=ax3, label='Grain ID')
    
    # 4. Grain size distribution
    ax4 = fig.add_subplot(2, 3, 4)
    grain_sizes = np.bincount(grain_map.flatten())
    ax4.hist(grain_sizes, bins=20, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Grain Size (cells)')
    ax4.set_ylabel('Count')
    ax4.set_title('Grain Size Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Euler angle distributions
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(df['phi1'], bins=30, alpha=0.5, label='φ₁', edgecolor='black')
    ax5.hist(df['PHI'], bins=30, alpha=0.5, label='Φ', edgecolor='black')
    ax5.hist(df['phi2'], bins=30, alpha=0.5, label='φ₂', edgecolor='black')
    ax5.set_xlabel('Angle (degrees)')
    ax5.set_ylabel('Count')
    ax5.set_title('Euler Angle Distributions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Orientation spread within grains
    ax6 = fig.add_subplot(2, 3, 6)
    grain_stds = []
    for grain_id in range(len(grain_sizes)):
        grain_data = df[df['grain_id_true'] == grain_id]
        if len(grain_data) > 1:
            std_phi1 = grain_data['phi1'].std()
            std_PHI = grain_data['PHI'].std()
            std_phi2 = grain_data['phi2'].std()
            grain_stds.append(np.mean([std_phi1, std_PHI, std_phi2]))
    
    ax6.hist(grain_stds, bins=20, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Mean Orientation Spread (degrees)')
    ax6.set_ylabel('Count')
    ax6.set_title('Orientation Spread Within Grains')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_grain_structure.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_columnar_grain_structure(nx=20, ny=20, nz=10, n_grains_xy=8, dx=1.0,
                                    orientation_spread=1.0, save_path='columnar_ebsd_data.csv'):
    """
    Create columnar grain structure (grains extend through entire Z direction)
    More realistic for certain materials like directionally solidified alloys
    """
    print(f"Creating columnar grain structure...")
    
    # Create 2D Voronoi in XY plane
    np.random.seed(42)
    seed_points_2d = []
    min_distance = min(nx, ny) / np.sqrt(n_grains_xy) * 0.7
    
    while len(seed_points_2d) < n_grains_xy:
        new_point = np.array([
            np.random.uniform(0, nx),
            np.random.uniform(0, ny)
        ])
        
        if len(seed_points_2d) == 0:
            seed_points_2d.append(new_point)
        else:
            distances = [np.linalg.norm(new_point - p) for p in seed_points_2d]
            if min(distances) > min_distance:
                seed_points_2d.append(new_point)
    
    seed_points_2d = np.array(seed_points_2d)
    tree_2d = KDTree(seed_points_2d)
    
    # Generate orientations
    grain_orientations = []
    for i in range(n_grains_xy):
        phi1 = np.random.uniform(0, 360)
        PHI = np.random.uniform(0, 180)
        phi2 = np.random.uniform(0, 360)
        grain_orientations.append((phi1, PHI, phi2))
    
    # Create data
    data = []
    for i in range(nx):
        for j in range(ny):
            # Find grain ID based on XY position
            point_2d = np.array([i, j])
            _, grain_id = tree_2d.query(point_2d)
            
            for k in range(nz):
                phi1, PHI, phi2 = grain_orientations[grain_id]
                
                # Add small orientation gradient along Z
                z_gradient = k / nz * orientation_spread
                phi1 += np.random.normal(0, orientation_spread) + z_gradient
                PHI += np.random.normal(0, orientation_spread)
                phi2 += np.random.normal(0, orientation_spread) + z_gradient
                
                # Ensure valid ranges
                phi1 = phi1 % 360
                PHI = np.clip(PHI, 0, 180)
                phi2 = phi2 % 360
                
                data.append({
                    'x': i * dx,
                    'y': j * dx,
                    'z': k * dx,
                    'phi1': phi1,
                    'PHI': PHI,
                    'phi2': phi2,
                    'phase': 1
                })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Saved columnar grain structure to: {save_path}")
    return df

def create_deformed_grain_structure(nx=20, ny=20, nz=10, n_grains=8, 
                                    deformation_axis='z', compression_ratio=0.5,
                                    dx=1.0, save_path='deformed_ebsd_data.csv'):
    """
    Create a deformed grain structure with elongated grains
    Simulates rolled or compressed microstructure
    """
    print(f"Creating deformed grain structure...")
    
    # Start with equiaxed grains and deform them
    if deformation_axis == 'z':
        # Compress in Z, elongate in XY
        seed_points = []
        for i in range(n_grains):
            x = np.random.uniform(0, nx)
            y = np.random.uniform(0, ny)
            z = np.random.uniform(0, nz * compression_ratio)
            seed_points.append([x, y, z])
    
    seed_points = np.array(seed_points)
    
    # Generate orientations with texture
    grain_orientations = []
    for i in range(n_grains):
        # Add preferred orientation (texture) due to deformation
        if i < n_grains // 2:
            # Texture component 1
            phi1 = np.random.normal(0, 15)
            PHI = np.random.normal(45, 10)
            phi2 = np.random.normal(0, 15)
        else:
            # Texture component 2
            phi1 = np.random.normal(90, 15)
            PHI = np.random.normal(45, 10)
            phi2 = np.random.normal(45, 15)
        
        grain_orientations.append((phi1, PHI, phi2))
    
    # Build KDTree with anisotropic metric
    data = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Apply inverse deformation to find nearest grain
                if deformation_axis == 'z':
                    point = np.array([i, j, k * compression_ratio])
                
                # Find nearest grain
                distances = [np.linalg.norm(point - sp) for sp in seed_points]
                grain_id = np.argmin(distances)
                
                phi1, PHI, phi2 = grain_orientations[grain_id]
                
                # Add deformation-induced orientation gradient
                strain_factor = 1.0 + (k / nz) * 0.5
                phi1 += np.random.normal(0, 5 * strain_factor)
                PHI += np.random.normal(0, 5 * strain_factor)
                phi2 += np.random.normal(0, 5 * strain_factor)
                
                # Ensure valid ranges
                phi1 = phi1 % 360
                PHI = np.clip(PHI, 0, 180)
                phi2 = phi2 % 360
                
                data.append({
                    'x': i * dx,
                    'y': j * dx,
                    'z': k * dx,
                    'phi1': phi1,
                    'PHI': PHI,
                    'phi2': phi2,
                    'phase': 1
                })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Saved deformed grain structure to: {save_path}")
    return df

##########



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        # Convert DAMASK VTI to EBSD format
        if len(sys.argv) < 4:
            print("Usage: python script.py convert <input.vti> <output.csv> [cell_size]")
            sys.exit(1)
        
        input_vti = sys.argv[2]
        output_csv = sys.argv[3]
        cell_size = float(sys.argv[4]) if len(sys.argv) > 4 else None
        
        CellularAutomatonRexGG3D_EBSD.convert_damask_vti_to_ebsd(input_vti, output_csv, cell_size)
        
    else:
        # Create example files if they don't exist
        if not os.path.exists('config_ebsd.yaml'):
            create_ebsd_yaml_config()
        
        if not os.path.exists('example_ebsd_data.csv'):
            create_example_ebsd_data()
        
        # Run simulation
        ca = CellularAutomatonRexGG3D_EBSD('config_ebsd.yaml')
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