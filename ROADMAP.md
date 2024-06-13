# Roadmap

## Introduction

This document outlines the development roadmap for Specials, detailing our vision, current focus, and future goals. This roadmap is intended to provide guidance for contributors and users about the future direction of the project.

## Vision

The Specials package aims to provide highly-optimized, hardware-accelerated special functions implemented in Mojo, with a strong emphasis on numerical accuracy and stability. Our goal is to become a go-to resource for special functions in AI computing.

## Current Focus (Next 12 Months)

- **Keep Up with Mojo Updates**: Monitor updates to the Mojo compiler and standard library, ensuring the proper use of language features that best meet the needs of Specials.
- **Adopt Language Standards and Best Practices**: Follow Mojo's standards and best practices for code formatting and documentation.
- **Explore Numerical Methods**: Investigate numerical methods to generate a single polynomial approximation that produces correctly rounded results for all inputs of a given elementary function, regardless of the floating-point format used. This will facilitate the addition of support for new formats and reduce the number of lookup tables.
- **Review Critical Algorithms and Data Structures**: Optimize critical algorithms and data structures used in various mathematical function implementations to reduce execution time, memory usage, and package size.
- **Facilitate Benchmarking**: Make it easier to add and maintain micro benchmarks for Specials, promoting reliability and reproducibility of comparative accuracy and computational efficiency results. Ensure benchmarks are fair, meaning they evaluate alternatives under the same conditions.
- **Expand and Rigorize Testing**: Enhance test coverage for numerical methods and data structures, floating-point formats, input domains, and edge cases. Make tests more rigorous by reducing tolerance values appropriately for the precision of each floating-point format.
- **Ensure Hardware Accelerator Support**: Ensure Specials executes correctly and efficiently on hardware accelerators supported by Mojo.

For the next few months, adding implementations for new mathematical functions is not a priority. The rapid evolution of Mojo makes adapting a large codebase costly. Possible exceptions include the `gamma`, `lgamma`, `beta`, `lbeta` functions, and the elementary functions they depend on.

## Future Goals (Beyond 12 Months)

- **Add Implementations for Special Functions**:
  - `igamma`: Lower regularized incomplete Gamma function
  - `igammac`: Upper regularized incomplete Gamma function
  - `igammainv`: Inverse of the `igamma` function
  - `igammacinv`: Inverse of the `igammac` function
  - `ibeta`: Lower regularized incomplete Beta function
  - `ibetac`: Upper regularized incomplete Beta function
  - `ibetainv`: Inverse of the `ibeta` function
  - `ibetacinv`: Inverse of the `ibetainv` function

- **Explore Partial Derivatives**: Investigate ways to implement partial derivatives of the special functions listed above.

- **Start Accepting Community Contributions**: Begin accepting pull requests with contributions from the community to foster collaboration and enhance the development of Specials.
