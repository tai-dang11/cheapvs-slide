+++
title = "Reliable and Efficient Amortized Model-based Evaluation"
date = "2024-09-23"
outputs = ["Reveal"]
[logo]
src = "images/sail-logo.jpg"

+++

{{< slide auto-animate="" >}}
### Accelerate Virtual Screening 
### with Amortized Neural Search 
### and Multi-Objective Bayesian Optimization


---
{{< slide auto-animate="" >}}
## 1.Introduction: Problem Set-up of Virtual Screening

- For **a given protein** linked to a certain disease,
<span class="fragment">we want to select a few small molecules (i.e., ligand)</span>
<span class="fragment">from a library of millions candidates</span>
<span class="fragment">such that the selected candidate will have the highest utility in disease treating.</span>

- <span class="fragment">For modern libraries, it is feasible scaling up to billions or even trillions of compounds enhances the reach and impact of virtual screening.</span>


---
{{< slide auto-animate="" >}}
## 1.Introduction: Problem Set-up of Virtual Screening
