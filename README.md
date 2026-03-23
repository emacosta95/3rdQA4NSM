# 3rdQA4NSM — Brillouin Wigner method for the Quasiparticle Nuclear Shell Model

> **Brillouin-Wigner perturbation theory for quasiparticle effective Hamiltonians in the sd-shell**

This repository contains the code associated with the paper:

> *Brillouin-Wigner Effective Hamiltonians for Quasiparticle Encoding in the Nuclear Shell Model*
> E. Costa

The goal of this project is to extend the quasiparticle (QP) approximation for nuclear shell model (NSM) Hamiltonians beyond leading order, using Brillouin-Wigner (BW) perturbation theory. This produces a corrected effective Hamiltonian that lives entirely in the small quasiparticle subspace — making it suitable for encoding on near-term quantum devices (NISQ) without fermionic overhead.

---

## Physics Background

The nuclear shell model (NSM) describes nuclei in the sd-shell by diagonalizing a many-body Hamiltonian built from single-particle energies and two-body matrix elements (e.g., the USDB interaction). The full many-body Hilbert space grows exponentially with nucleon number, making exact diagonalization (Active Configuration Interaction, ACI) expensive.

The **quasiparticle (QP) approximation** reduces this to a small subspace spanned by paired neutron-proton states. While compact, this approximation breaks down for nuclei with many valence nucleons. This code implements three increasingly refined corrections:

1. **Full BW method** — iterative BW series applied to the full NSM Hamiltonian.
2. **Truncated BW method with Pauli blocking** — BW corrections derived from the 2-quasiparticle sector, with off-diagonal transitions suppressed by a Pauli blocking factor.
3. **HF-BW method** — BW corrections where the resolvent denominator is replaced by the Hartree-Fock energy, allowing a controlled truncation of the perturbative series.

The resulting effective Hamiltonians are compared to the exact ACI ground state energy and wavefunction across 12 nucleon configurations: `(2,2), (4,2), (6,2), (8,2), (10,2), (4,4), (4,8), (4,10), (6,10), (8,8), (8,10), (10,10)` neutron-proton pairs.

---

## Repository Structure

```
3rdQA4NSM/
│
├── src/
│   ├── brillouin_wigner_utils.py      # Core BW perturbation theory functions
│   ├── hartree_fock_library.py        # HF energy functional and orbital optimization
│   ├── cg_utils.py                    # Clebsch-Gordan coefficient utilities
│   └── qiskit_utils.py                # Qiskit/IonQ circuit utilities for hardware execution
│
├── data/
│   ├── usdb.nat                       # USDB interaction file (single-particle energies + 2BMEs)
│   └── bw_calculations.pkl            # Output: energies, fidelities, and accuracies
│
├── compute_fidelities_for_brillouin_wigner.py   # Main script: runs all BW methods
└── README.md
```

---

## Source Code Reference

### `src/brillouin_wigner_utils.py`

The core of the perturbative machinery. All functions take the block-decomposed Hamiltonian matrices `H_QQ`, `H_QR`, `H_RQ`, `H_RR` as input, where `Q` is the quasiparticle subspace and `R` is the rest of the Hilbert space.

---

#### `full_brillouin_wigner_method(hamiltonians, threshold, eigvals_aci, nsteps_iteration)`

Applies the standard Brillouin-Wigner perturbation theory to the **full NSM Hamiltonian** at the target particle number. The effective correction to `H_QQ` is built iteratively as:

$$\Delta H^{(n)}_{QQ} = H_{QR} \frac{1}{E^n} H_{RR}^{n-1} H_{RQ}$$

The series is summed until the energy converges below `threshold`. Returns the corrected ground state wavefunction, energy, effective Hamiltonian, and convergence history.

**Returns:** `(psi_qq, energy, H_eff, infidelities, energy_errors, delta_H)`

---

#### `full_brillouin_wigner_method_pauliblockade(hamiltonians, threshold, eigvals_aci, nsteps_iteration)`

Applies a **truncated BW method** where the coupling matrices are computed from the **2-quasiparticle sector** only, with Pauli blocking corrections applied to suppress unphysical transitions. The Pauli blocking factor for a transition adding single-particle state `a` is `(1 - n_a)`, estimated from the mean-field occupancy.

The series structure is:
- Order 0: `H_QR_pb @ H_RQ_pb`
- Order n ≥ 1: `H_QR_pb @ H_RR_pb @ ... @ H_RR @ H_RQ_pb`

This produces a two-body effective interaction `ΔH_QQ` that is then expressed in the full QP basis using `HardcoreBosonsBasis`.

**Returns:** `(psi_qq, energy, H_eff, infidelities, energy_errors, delta_H)`

---

#### `brillouin_wigner_method_hf_ansatz(hamiltonians_hf, hamiltonian_qq, density_matrix_rr, QPC2body, QPC, threshold, eigvals_aci, nsteps_iteration)`

Applies the **HF-BW method**, where the resolvent denominator `(E - H_RR)^{-1}` is replaced by `1/E_HF`, with the HF density matrix `ρ_RR` inserted to project corrections onto the physically relevant part of the R space:

$$\Delta H^{(n)}_{QQ} \sim H_{QR} \rho_{RR} \frac{1}{E_{HF}^n} H_{RR}^{n-1} H_{RQ}$$

The resulting corrections are lifted to the full QP many-body basis via `HardcoreBosonsBasis.adag_adag_a_a_matrix_optimized`. The method converges faster and the effective Hamiltonian is more compact than the full BW version.

**Returns:** `(psi_qq, energy, H_eff, infidelities, energy_errors, delta_H)`

---

#### `computation_pauli_blockade(basis, QPC, psi_qq)`

Computes the observables needed for Pauli blocking: the single-particle density `n_i = ⟨ψ|c†_i c_i|ψ⟩` and the two-body vacuum density matrix `ρ^(2)_00 = ⟨ψ|(1-n_a)(1-n_b)|ψ⟩`, where `ψ` is the QP ground state extended to the full Hilbert space.

**Returns:** `(rho_vacuum, single_particle_density)`

---

### `src/hartree_fock_library.py`

Implements a differentiable Hartree-Fock energy functional in PyTorch for nuclear systems with neutrons and protons, used to obtain the HF reference state and orbitals for the HF-BW method.

---

#### `class HFEnergyFunctionalNuclear(nn.Module)`

A PyTorch `nn.Module` that parametrizes the HF Slater determinant via orthogonal orbital matrices `A_n` (neutrons) and `A_p` (protons), obtained through QR decomposition. The energy functional is:

$$E_{HF} = \text{Tr}[h\rho] + \frac{1}{2}\sum_{abcd} V_{abcd} \rho_{ca}\rho_{db}$$

summed over neutron-neutron, proton-proton, and neutron-proton contributions. Optimized with Adam in the main script.

**Key methods:**
- `forward()` — computes and returns the HF energy.
- `hf_component_from_indices(occ_idx, species)` — returns `⟨S|HF⟩` for a Slater determinant specified by occupied orbital indices (used to build `ψ_HF` in the 2-body sector).
- `build_fock_matrices_factorized()` — returns separate neutron and proton Fock matrices including cross-species contributions.
- `transform_integrals_using_fock_rotation(U, h, V)` — transforms one- and two-body integrals under a unitary rotation `U`.

---

#### `slater_determinants_combined(C_n, C_p, fock_basis)`

Computes the Slater determinant amplitudes `⟨S|HF⟩` for a combined neutron-proton system over the full Fock basis, by evaluating the product of neutron and proton minor determinants.

#### `gram_schmidt(V)`

Standard Gram-Schmidt orthogonalization of a matrix `V` (columns are vectors). Used for orbital orthonormalization.

#### `build_fock_matrix(h_mat, V_tensor, rho)`

Builds the Fock matrix `F_ab = h_ab + Σ_cd V_{acbd} ρ_{dc}` from single-particle energies, two-body tensor, and density matrix. Returns a Hermitian matrix.

---

### `src/cg_utils.py`

Computes **Clebsch-Gordan coefficients** `⟨j1 m1 j2 m2 | J M⟩` using the standard recursion relations (adapted from Márton Juhász's implementation). Used for angular momentum coupling in the nuclear basis construction.

**Key classes/functions:**
- `class CG` — stores quantum numbers `(j1, m1, j2, m2, J, M)` and the coefficient value.
- `ClebschGordan(j1, j2, J)` — returns all non-zero CG coefficients for a given `(j1, j2, J)` triplet.

---

### `src/qiskit_utils.py`

Utilities for running the quasiparticle effective Hamiltonian on quantum hardware (IonQ) via Qiskit.

**Key functions:**
- `get_hamiltonian(t_onebody, n_sites)` — converts a one-body hopping dictionary to a `SparsePauliOp` in the XX+YY+Z form (Jordan-Wigner encoding of the QP Hamiltonian).
- `ionq_energy_expectation(circuits, hamiltonian, backend, shots, nparticle_sector)` — computes `⟨H⟩` from X/Y/Z basis measurement circuits on an IonQ backend. Supports optional particle-number sector filtering to approximately restore symmetry.
- `expectation_from_counts(counts, pauli)` — computes `⟨P⟩` from Qiskit measurement counts for a given Pauli string using parity counting.
- `get_shots_qiskit(circuit_x, circuit_y, circuit_z, shots, backend)` — runs three measurement circuits (X, Y, Z bases) on a backend and returns the counts.

---

## Main Script: `compute_fidelities_for_brillouin_wigner.py`

The main script runs all three BW methods for each of the 12 nucleon configurations and saves the results. It proceeds as follows:

1. **Load the USDB interaction** from `data/usdb.nat` using `SingleParticleState` and `get_twobody_nuclearshell_model`.
2. **Build the ACI Hamiltonian** for each `(nparticles_a, nparticles_b)` pair using `FermiHubbardHamiltonian`, and compute the exact ground state energy.
3. **Construct the QP subspace** using `QuasiParticlesConverterOnlynnpp` and project the Hamiltonian into `H_QQ`, `H_QR`, `H_RQ`, `H_RR` blocks.
4. **Full BW method** — call `full_brillouin_wigner_method` to get the corrected effective Hamiltonian and its ground state.
5. **Truncated BW with Pauli blocking** — build the 2-body sector Hamiltonian and Pauli blocking factors, call `full_brillouin_wigner_method_pauliblockade`, extract two-body QP effective interaction coefficients, and reconstruct `ΔH_QQ` in the full QP basis using `HardcoreBosonsBasis`.
6. **HF-BW method** — optimize the HF wavefunction in the full and 2-body particle sectors with `HFEnergyFunctionalNuclear`, build the HF density matrix in the R space, and call `brillouin_wigner_method_hf_ansatz`.
7. **Compare methods** — energies, wavefunction fidelities `|⟨ψ_FBW|ψ_method⟩|²`, and operator fidelities `‖H_FBW - H_method‖_F / ‖H_FBW‖_F` are recorded for all methods.
8. **Save results** to `data/bw_calculations.pkl`.

---

## Installation

```bash
git clone https://github.com/emacosta95/3rdQA4NSM.git
cd 3rdQA4NSM
pip install -r requirements.txt
```

**Dependencies:**
- `numpy`, `scipy` — numerical linear algebra and sparse matrix operations
- `torch` — HF orbital optimization
- `qiskit` — quantum circuit construction and hardware execution
- `tqdm`, `matplotlib` — progress bars and plotting
- `NSMFermions` — internal package for NSM Hamiltonians, QP basis construction, and QML utilities

---

## Running

```bash
python compute_fidelities_for_brillouin_wigner.py
```

Results are saved to `data/bw_calculations.pkl` as a dictionary with keys:

| Key | Description |
|-----|-------------|
| `nparticles` | List of `(n_neutrons, n_protons)` configurations |
| `energies_exact` | ACI ground state energies |
| `energies_qq` | Zeroth-order QP energies |
| `energies_fbw` | Full BW energies |
| `energies_truncated_bw` | Truncated BW energies |
| `energies_hfbw` | HF-BW energies |
| `accuracies_fbw` | Energy error per iteration (Full BW) |
| `accuracies_hfbw` | Energy error per iteration (HF-BW) |
| `fidelities_truncated_bw` | `\|⟨ψ_FBW\|ψ_truncBW⟩\|²` |
| `fidelities_hfbw` | `\|⟨ψ_FBW\|ψ_HFBW⟩\|²` |
| `operator_fidelities_bw` | Frobenius distance: truncBW vs FBW |
| `operator_fidelities_hfbw` | Frobenius distance: HF-BW vs FBW |

---

## Citation

If you use this code, please cite:

```bibtex
@article{costa2026bwnsm,
  title   = {Brillouin-Wigner Effective Hamiltonians for Quasiparticle Encoding in the Nuclear Shell Model},
  author  = {Costa, Emanuele},
  year    = {2026},
  note    = {arXiv preprint}
}
```

---

## License

MIT License. See `LICENSE` for details.
