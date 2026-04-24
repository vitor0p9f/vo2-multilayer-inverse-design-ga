# VO₂ Multilayer Inverse Design Using Genetic Algorithm

<div align="justify">

This repository contains the code developed for a research project conducted as part of the Thematic Core – Alternative Energy Sources course in the Computer Engineering bachelor's program at Universidade Federal do Vale do São Francisco (UNIVASF).

## Motivation

<div align="justify">

VO₂ is a widely studied material for smart window applications due to its reversible `metal–insulator transition (MIT)`, which leads to infrared-blocking behavior at approximately 68 °C (154.4 °F).

## Hypothesis

<div align="justify">

Given that existing methods to reduce the critical temperature of the metal–insulator transition (MIT) are complex and not yet suitable for large-scale industrial production, this raised the following question for `Joaquim Junior Isidio De Lima`, a professor at UNIVASF:

<div align="justify">

> Is it possible to design a multilayer structure containing VO₂ that exhibits the desired behavior of VO₂ in its insulating phase at a lower temperature, for example, 26 °C (78.8 °F)?

<div align="justify">

To investigate this question, the professor began testing several structures using the `Finite Element Method (FEM)` in a professional simulation software.

<div align="justify">

However, the search space of possible multilayer configurations is extremely large, making an exhaustive search computationally infeasible. For this reason, we proposed simplifying the problem as an optimization task involving the inverse design of the structure through parameter optimization of a `Transfer Matrix Method (TMM)`.

## Methodology

<div align="justify">

The methodology adopted in this research was based on the following steps:

<div align="justify">

1. Define a target function mathematically
2. Optimize the parameters of the TMM model using a `Genetic Algorithm (GA)`
3. Select the five best-performing structures.
4. Use a paired t-test to compare the structures using `FEM` and `TMM` results

### Genetic Algorithm

<div align="justify">

The implementation of the `GA` followed the standard approach, with the necessary operators adapted to operate correctly in this context.

#### Phenotype Structure

<div align="justify">

In the context of a `GA`, a phenotype represents a solution to the problem. In this work, the phenotype is defined by the following structure:

<div align="justify">

- A gene representing the number of layers in the structure
- A chromosome representing the arrangement of the layers in the structure

<div align="justify">

Inside the chromosome, a gene pair `(r, t)` is defined, where `r` represents the refractive index of the layer material at 26 °C (78.8 °F), and `t` represents the thickness of the material. Each pair `(r, t)` therefore defines a single layer in the structure.

#### Initial Population Generation

#### Fitness

<div align="justify">

The fitness function consists of the following mathematical function:

<div align="center">

![Fitness function](images/fitness_function.png)

<div align="justify">

Where `i` denotes the individual and `e(i)` is the weighted Root Mean Square Error (RMSE) between the target function and the spectrum of the current structure.

<div align="justify">

Since the light spectrum can be divided into three main ranges, weights were assigned to each range to represent their relative importance in the optimization process. The weights used were:

<div align="justify">

- 0.45 for the infrared spectrum
- 0.35 for the visible spectrum
- 0.20 for the ultraviolet spectrum

#### Parents Selection

<div align="justify">

Parent selection was performed using tournament selection, in which a subset of the population is chosen through `k` tournaments with `n` participants each. The winner of each tournament is determined based on its fitness value.

#### Crossover Operator

<div align="justify">

Due to the vector representation of the chromosome, the two-point crossover operator was applied. The same strategy was also used for the gene representing the number of layers.

<div align="justify">

Since the number of layers may differ between individuals, the crossover points were determined based on the parent with the smallest chromosome length.

#### Mutation Operator

<div align="justify">

To efficiently manage the mutation of an individual, two mutation operators were designed. Each operator targets a different component of the individual representation: the `gene` that encodes the `number of layers`, and the `chromosome` that encodes the `arrangement of layers`.

<div align="justify">

The first operator is a traditional `bit-flip mutation` applied to a binary representation of the number of layers. In this operator, each bit of the binary gene is independently evaluated according to the mutation probability. Whenever a mutation occurs, the corresponding bit is flipped (0 → 1 or 1 → 0). This operator introduces structural variability by allowing the genetic algorithm to explore individuals with different numbers of layers.

<div align="justify">

The second operator mutates the chromosome that encodes the sequence of layers. Since each layer is represented as a pair `(material, thickness)`, the mutation operator is composed of multiple internal operations designed to explore different types of variation. These operations are applied sequentially in a cascade, meaning that the chromosome may undergo `none`, `one`, or `multiple` modifications during a mutation event. Three mutation mechanisms are used:

<div align="justify">

1. Layer swap operator: this operator randomly selects a range of positions in the chromosome and scrambles them. This allows the algorithm to explore different layer orderings without modifying their physical properties.
2. Pair replacement operator: in this operator, each (material, thickness) pair in the chromosome is independently evaluated according to the mutation probability. When a mutation occurs, the corresponding pair is replaced with another randomly generated valid pair. This mutation introduces new layer compositions into the chromosome.
3. Internal pair modification operator: this operator modifies the internal attributes of an existing pair. Two types of modifications can occur: material mutation, in which the material is replaced by another material randomly selected from the available material set, and thickness mutation, in which the thickness is perturbed by a small variation of up to ±5% of its value. Each element of the pair has an independent probability of undergoing mutation, meaning that the material, the thickness, or both may be modified. This type of mutation enables fine local adjustments in the solution space.

<div align="justify">

All mutation mechanism follows the same predefined mutation rate throughout the evolutionary process.

### New Population Selection

<div align="justify">

The selection of the new population was based on the `(μ + λ) evolutionary strategy`, in which parents and offspring are ranked according to their fitness and the best individuals are selected to form the next population.

<div align="justify">

This strategy ensures that the best solutions found so far are preserved while allowing newly generated candidate solutions to compete for survival in the next generation.

### Statistical Analysis

<div align="justify">

To obtain a reasonable amount of data for the paired t-test, the layer thickness was perturbed between one percent and ten percent, in steps of zero point five percent, generating twenty-one samples for each structure.

<div align="justify">

After that, all samples were evaluated using `TMM` and `FEM`, and the paired t-test was applied.

## Results