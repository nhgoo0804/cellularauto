04_test.ipynb works well.

The key fixes that made it work were:

Allowing recrystallized grains to grow - The special condition for rex grains to consume deformed material
Higher mobility - Increasing from 1e-12 to 1e-4 m⁴/(J·s)
Proper nucleation - Placing rex nuclei at grain boundaries with energy gradients

The simulation is now correctly showing:

Primary recrystallization: New strain-free grains consuming the deformed matrix
Grain growth: Larger grains consuming smaller ones after recrystallization

You should see in the full visualization:

Decreasing stored energy over time
Increasing recrystallized fraction
Evolution of grain size distribution
JMAK kinetics plot showing the recrystallization behavior


05_test.ipynb 

You're absolutely right! I apologize for the misunderstanding. The model should NOT have pre-placed nucleation sites. Instead, recrystallization nuclei should emerge naturally through competitive subgrain growth. Let me fix this:

nothing happens, thus I suggest that you make the initial state deformed much more. i.e. some location has severe stored energy, high misorientation. One more thing I mention is how about deal with quaternions rather than Euler angles when you calculate the misorientation and stored energy.

You're absolutely right! The current initial state isn't deformed enough to drive recrystallization. Let me create a much more heavily deformed microstructure with severe orientation gradients and stored energy concentrations. I'll also implement quaternion-based misorientation calculations for more accurate results.

06_test.ipynb

Excellent question! Let me analyze why the subgrains/cell blocks aren't growing. Looking at the calculate_reorientation_rates() method, here are the likely issues:
1. Energy Calculation Problem
pythoncurrent_total_energy = current_boundary_energy + self.stored_energy[cell_idx] * self.dx
new_total_energy = new_boundary_energy + self.stored_energy[nbr_idx] * self.dx
The stored energy is multiplied by dx (1e-6 m), making it negligible compared to boundary energy! This means:

Boundary energy: ~0.5 J/m²
Stored energy contribution: ~1e7 × 1e-6 = 10 J/m² (much smaller than it should be)

2. Cell Blocks Have Small Misorientation
The cell blocks only have 1-3° misorientation from parent grains, which means:

Very low boundary energy (Read-Shockley model)
Very low mobility (mobility ∝ (θ/15°)⁴ for LAGB)
Weak driving force for growth

3. Energy Minimization Logic
The current logic requires the total energy to decrease for growth. But cell blocks might not provide enough energy reduction because:

Their boundaries are low-angle (low energy)
The stored energy difference might not overcome the boundary energy change

Quick Fixes:
1. Fix the energy scaling:
python# Remove the dx multiplication for stored energy
current_total_energy = current_boundary_energy + self.stored_energy[cell_idx] / 1e6  # Convert to comparable units
2. Increase cell block misorientation:
python# Change from 1-3° to 10-20° for cell blocks
small_angle = np.radians(np.random.uniform(10, 20))
3. Add a growth advantage for low-energy regions:
python# Boost driving force for low stored energy neighbors
if self.stored_energy[nbr_idx] < 0.3 * np.mean(self.stored_energy):
    energy_decrease *= 2.0  # Growth advantage for recovered regions
