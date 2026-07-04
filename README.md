# Inverse Design of VO₂‑Based Multilayers via Genetic Algorithm

This repository contains the implementation developed for a research project carried out as part of the Thematic Core – Alternative Energy Sources course in the Computer Engineering undergraduate program at Universidade Federal do Vale do São Francisco (UNIVASF). The objective is to investigate whether a multilayer optical structure containing VO₂ can reproduce, through inverse design, a target spectral response associated with the insulating phase of the material at temperatures below its intrinsic transition point.

## Motivation

VO₂ is a widely studied material for smart window applications due to its reversible metal–insulator transition (MIT), which produces strong infrared-blocking behavior near 68 °C. However, lowering the effective transition temperature of this material remains a major challenge in practical applications. This motivated the study of multilayer structures that could alter the optical response in a way that mimics or enhances the desired switching behavior at lower temperatures.

## Hypothesis

The central hypothesis is that a multilayer stack containing VO₂ can be designed so that its optical response resembles the behavior expected for VO₂ in its insulating state at a lower effective temperature, such as 26 °C. Rather than relying on direct modification of the material itself, the approach explores the possibility of achieving the same functional effect through structural design and material arrangement.

## Methodology

The workflow followed in this project can be summarized in four main steps:

1. Define a target spectral function.
2. Simulate the optical response of candidate multilayer structures using the Transfer Matrix Method (TMM).
3. Optimize the structure using a Genetic Algorithm (GA).
4. Select the best candidates and compare their responses against reference results obtained with FEM-based simulations.

## Target function

The optimization problem is formulated as the search for a multilayer structure whose optical response approximates a desired spectral profile. In this repository, the target is represented by a super-Gaussian function of the form

$$
f(\lambda) = B + A \exp\left[-\left(\frac{(\lambda - \lambda_0)^2}{2\sigma^2}\right)^p\right]
$$

where:

- $A$ is the peak amplitude;
- $B$ is the baseline;
- $\lambda_0$ is the center wavelength;
- $\sigma$ is the width parameter;
- $p$ is the super-Gaussian order.

In the experiments carried out here, the target was configured as a narrow transmittance peak centered around 575 nm, with a high order parameter to make the profile sharper and more selective.

## Optical model and thermal preprocessing

The optical response is evaluated using TMM for multilayer stacks composed of different materials. Before optimization, the material data are prepared for several temperatures.

The repository includes a thermal preprocessing pipeline in which:

- the optical constants $n$ and $k$ of each material are loaded from reference data;
- the extinction coefficient $k$ is updated according to a thermal expansion-based model;
- the corresponding change in $n$ is inferred through a discrete Kramers–Kronig formulation.

This makes it possible to evaluate how the optical response changes with temperature without requiring a full experimental dataset for every temperature of interest.

## Genetic algorithm formulation

The optimization is performed with a GA adapted to the structure of the problem.

### Representation

Each individual represents a multilayer stack. The structure is encoded in a way that includes:

- a set of layers;
- the material assigned to each layer;
- the thickness assigned to each layer;
- an activity mask that allows some layers to be effectively disabled.

### Initial population

The initial population is generated from a template that defines the allowed sequence of optical elements. This template includes:

- an incidence medium;
- one or more intermediate layers;
- a substrate.

Some layers are marked as free, meaning that their material and/or thickness can be optimized by the GA.

### Fitness function

The fitness of each structure is based on the discrepancy between the computed optical spectrum and the target spectrum. The cost function is computed over multiple spectral bands and multiple temperatures, with weighted contributions that emphasize the most relevant portions of the spectrum.

### Selection, crossover and mutation

The algorithm uses tournament selection to choose parents. The crossover and mutation operators were adapted to respect the structure of the representation and the template constraints.

#### Crossover

The crossover operator is a custom multi-point crossover that does not exchange information blindly across the full chromosome. Instead, it only swaps variables in positions that are actually free to vary. This preserves the integrity of fixed layers and avoids invalid offspring.

#### Mutation

The mutation operator includes several mechanisms:

- scramble of free layers;
- activation or deactivation of fully free layers;
- mutation of materials in free-material positions;
- mutation of thicknesses in free-thickness positions.

These operations allow the algorithm to explore different arrangements while still respecting the template and the admissible search space.

## Template system

The template system is one of the key components of this implementation. A template defines the high-level architecture of the stack, while the GA explores the free variables inside that architecture.

The main entities are:

- IncidenceMedium: fixed entry medium;
- Layer: fixed or partially free layer;
- FreeBlock: region that may expand into one or more free layers;
- Substrate: fixed exit medium.

This representation allows the optimization process to be constrained to physically meaningful structures and makes it possible to preserve a desired overall topology while varying the internal configuration.

## Simulation parameters

The main parameters used in the current configuration are:

- maximum number of layers: 10;
- temperatures: 288, 293, 298, 303 and 308 K;
- wavelengths: 300 nm to 1100 nm;
- population size: 200;
- maximum generations: 500;
- mutation rate: 0.1;
- crossover rate: 0.8;
- crossover points: 3;
- tournament size: 3;
- elitism fraction: 0.1;
- target class: transmittance;
- target center: 575 nm;
- allowed thicknesses: 10 nm to 100 nm in steps of 5 nm.

## Validation and statistical analysis

To assess the robustness of the selected solutions, the project also includes validation procedures in which the thicknesses of the best-performing structures are slightly perturbed and the resulting spectra are compared. This provides a controlled way to evaluate sensitivity and to produce a set of samples suitable for paired statistical analysis, such as the paired t-test mentioned in the original research workflow.

## Repository structure

The repository is organized as follows:

- main.ipynb: main notebook containing the experimental setup and execution flow;
- tests.ipynb: auxiliary notebook for validation and exploratory analysis;
- genetic_algorithm/: implementation of the GA operators and evaluation logic;
- template/: template-based structure generation and layer representation;
- specifications/: material, environment and optical type definitions;
- experiment/: configuration, checkpointing and report generation;
- tmm/: transfer matrix optical calculations;
- data/, database/ and original_material_data/: input data and processed material datasets.

## How to run

The main execution flow is implemented in the notebook main.ipynb. In practice, the steps are:

1. define the materials and temperature grid;
2. configure the target function and spectral bands;
3. define the template and experiment parameters;
4. run the genetic algorithm;
5. inspect the generated checkpoints and reports in the experiments directory.

## Summary

This project combines optical modeling, thermal preprocessing of material properties, and evolutionary optimization to search for multilayer structures that approximate a desired spectral behavior. The resulting framework is intended as a flexible tool for inverse design of VO₂-based optical stacks and for the exploration of temperature-dependent optical responses.
