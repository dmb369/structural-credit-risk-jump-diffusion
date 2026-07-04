# References

**Merton, R. C. (1974).** "On the Pricing of Corporate Debt: The Risk
Structure of Interest Rates." *Journal of Finance*, 29(2), 449-470.
— Origin of the structural (option-theoretic) default model used in Phase 1.

**Merton, R. C. (1976).** "Option Pricing When Underlying Stock Returns Are
Discontinuous." *Journal of Financial Economics*, 3(1-2), 125-144.
— Origin of the jump-diffusion process used in Phase 2.

**Aguilar, J.-P., Pesci, C., & James, N. (2021).** "A structural approach to
default modelling with pure jump processes." arXiv:2102.06299.
— The original research target for this project. Motivated the jump
extension conceptually (GBM underestimates short-horizon default risk) but
was not implemented directly; see METHODOLOGY.md Section 3.1 for why.

**Vassalou, M., & Xing, Y. (2004).** "Default Risk in Equity Returns."
*Journal of Finance*, 59(2), 831-868.
— Standard reference for the iterative Merton calibration procedure and the
KMV-style default point (Current Liabilities + 0.5 × Long-Term Debt) used in
Phase 1.

**Jones, E. P., Mason, S. P., & Rosenfeld, E. (1984).** "Contingent Claims
Analysis of Corporate Capital Structures: An Empirical Investigation."
*Journal of Finance*, 39(3), 611-625.

**Eom, Y. H., Helwege, J., & Huang, J. Z. (2004).** "Structural Models of
Corporate Bond Pricing: An Empirical Analysis." *Review of Financial
Studies*, 17(2), 499-544.
— Both empirically document the "credit spread puzzle": Merton-style GBM
models systematically underprice short-horizon default risk. This project's
Phase 1 vs. Phase 2 comparison reproduces that effect on new data.

**Botha, A., & Muller, T. (2025).** "Approaches for modelling the
term-structure of default risk under IFRS 9: A tutorial using discrete-time
survival analysis." arXiv:2507.15441.
— Considered as an alternative project direction (industry-standard PD
methodology); not used here, kept as a reference for possible future work.
