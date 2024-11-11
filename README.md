# ISCAS and Research Overview on One-Bit Quantization

## Introduction

This repository provides an in-depth exploration of research focused on one-bit quantization and signal processing. The contents include papers, projects, and documentation detailing methods for optimizing and implementing one-bit quantization for various applications in signal processing and communications.

## Overview of Projects

### 1. **Optimization-Based One-Bit Quantization (ISCAS 2024)**
   - **Authors**: Florian Mayer, Christian Vogel
   - **Abstract**: One-bit quantization simplifies signal representation by mapping real-valued signals into a binary format. This paper introduces a mapping function \( F(\cdot) \) that minimizes the squared \( l_2 \)-norm error between the original and quantized signals, leading to efficient one-bit representations.
   - **Key Equations**:
     \[
     E(x, b) = \|e\|_2^2 = \sum_{n=0}^{N-1} |e_n|^2 = \|W \cdot (x - b)\|_2^2 = \|W \cdot d\|_2^2
     \]
   - **Special Cases**: The optimization framework generalizes methods like Sigma-Delta quantizers, highlighting new possibilities for adaptive one-bit signal processing.

### 2. **Block-Based Optimization for One-Bit Quantization (ISCAS 2025)**
   - **Objective**: To address the high noise introduced in complex signals through block-based partitioning of signals and independent optimization within each block.
   - **Mathematical Formulation**:
     \[
     b = F(x) \quad \text{where} \quad x \in \mathbb{R}^N, \quad b \in \{-1, 1\}^N
     \]
   - **Error Analysis**:
     \[
     e = R \cdot (x - b), \quad E(x, b) = \|e\|_2^2
     \]

## Theoretical Insights and Research Contributions

### A. **One-Bit Compressive Sensing**
   - **Concept**: Extends classical compressive sensing by using 1-bit measurements as constraints and reconstructing signals on the unit sphere to preserve consistency.
   - **Mathematical Insight**:
     \[
     y = \text{sign}(\Phi x), \quad \text{where } \|x\|_2 = 1
     \]

### B. **Sigma-Delta Modulation Analysis**
   - **Technique**: Achieves high-resolution ADC conversion through oversampling and noise shaping, allowing a one-bit ADC to exceed traditional multi-bit ADC performance.

### C. **Sequential and Parallel Processing Approaches**
   - **Sequential Optimization**: Simplifies one-bit signal reconstruction by updating one sample at a time.
   - **Potential Challenges**:
     - High computational demands for large-scale signals due to the NP-hard nature of global optimization.

## Applications and Implications

- **Communications**: Enhances low-complexity, high-efficiency receivers with oversampling capabilities.
- **Audio Processing**: Facilitates high sampling rates and reduces processing complexity.
- **Machine Learning**: Optimizes neural network structures for edge devices by minimizing data size through one-bit representations.

## Open Research Questions

- How can the one-bit quantization function \( F(\cdot) \) be enhanced to ensure exact signal reconstruction while maintaining low computational costs?
- What adaptations in quantization can help handle noise and environmental uncertainties, leading to more robust systems?

## File Structure

```
├── .git/
├── 01_Library/
├── 41_ISCAS2024/
│   ├── optimization_approach.pdf
├── 42_ISCAS2025/
│   ├── block_based_quantization.pdf
├── 90_Sandbox/
│   ├── experimental_codes/
├── 99_WasteBin/
│   └── archived_notes.md
├── README.md
```

## References and Citations

1. Mayer, F., & Vogel, C. (2023). An Optimization-Based Approach to One-Bit Quantization. IEEE.
2. Boufounos, P. T., & Baraniuk, R. G. (2008). 1-Bit Compressive Sensing.
3. Aziz, P. M., Sorensen, H. V., & Van der Spiegel, J. (1996). An Overview of Sigma-Delta Converters.

## Contact Information

**Author**: Florian Mayer  
**Affiliation**: FH JOANNEUM University of Applied Sciences, Graz, Austria  
**Email**: [florian.mayer@fh-joanneum.at](mailto:florian.mayer@fh-joanneum.at).
