+++
title = "Preferential Multi-Objective Bayesian Optimization for Drug Discovery"
date = "2025-01-01"
outputs = ["Reveal"]
math=true
codeFences = true

+++
{{< slide auto-animate="" >}}
<div style="text-align: left;">
  <h3 style="font-size: 1.5em;">Preferential Multi-Objective Bayesian Optimization for Drug Discovery</h3>
</div>

<p style="font-size: 30px;">
Tai Dang<sup>1,2</sup>,Hung Pham<sup>3</sup>,Sang Truong<sup>2</sup>, Ari Glenn<sup>2</sup>, Wendy Nguyen<sup>1</sup>, Edward A Pham<sup>2</sup>, Jeffrey S. Glenn<sup>2</sup>, Sanmi Koyejo<sup>2</sup>, Thang Luong<sup>1</sup> 
</p>
<p style="font-size: 30px;">
<sup>1</sup>RHF.AI, <sup>2</sup>Stanford University, <sup>3</sup>Imperial College London
</p>

<div style="display: flex; justify-content: space-between; width: 30%;">
  <img src="figures/rhf.png" alt="Third Logo" style="height: 85px; margin-left: 10px; margin-top: 250px">
  <img src="figures/SOM_vert_Web_Color_LG.png" alt="Main Logo" style="height: 85px; margin-left: 60px; margin-top: 250px"> 
  <img src="figures/sail-logo.jpg" alt="Second Logo" style="height: 85px; margin-left: 75px; margin-top: 250px">
</div>

---
{{< slide auto-animate="" >}}
### Outline
- Problem Setup
- Preference Elicitation from Pairwise Comparisons
- Benchmarking Docking Models
- Chemist-guided Active Virtual Screening


---
{{< slide auto-animate="" >}}
### Overview: Challenges in Virtual Screening

<!-- <p><b>Problem:</b> Large-scale virtual screening is computationally expensive, inefficient.</p>

<li class="fragment"><b>Computational Bottleneck:</b> Exhaustive docking of millions of compounds wastes substantial resources on low-quality hits.</li>

<li class="fragment"><b>Manual Hit Selection:</b> Chemists must manually evaluate molecules, balancing multiple properties, making the process slow and labor-intensive.</li>

<li class="fragment"><b>Lack of Multi-Objective Optimization:</b> Traditional VS prioritizes affinity, ignores other key factors, leading to wasted effort on unsuitable candidates.</li>

<p class="fragment"><b>→ A more efficient, expert-informed, and multi-objective approach is needed.</b></p> -->
<p><b>Problem:</b> Traditional virtual screening is inefficient and overly focused on binding affinity.</p> <ul> <li class="fragment"><b>Computational Waste:</b> Exhaustive docking wastes resources on low-quality hits.</li> <li class="fragment"><b>Manual Bottlenecks:</b> Chemists manually evaluate compounds, slowing progress.</li> <li class="fragment"><b>Lack of Multi-Objective Optimization:</b> Other crucial properties-synthesizability, solubility, safety-are often ignored.</li> </ul> 
<p class="fragment"><b>→ A smarter, multi-objective, expert-informed strategy is essential.</b></p>

---
{{< slide auto-animate="" >}}
### Overview: Streamlining Virtual Screening with Advanced Techniques

<div style="margin-top: 20px; display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 50%;">
    <h3 style="font-size: 36px;">Challenges:</h3>
    <ol style="font-size: 32px;">
      <li class="fragment" data-fragment-index="1">Leverage expert intuition to optimize drug candidate selection</li>
      <li class="fragment" data-fragment-index="2">Comparing diffusion vs. physics-based models for affinity prediction</li>
      <li class="fragment" data-fragment-index="3">Single-objective virtual screening insufficient</li>
    </ol>
  </div>

  <div style="width: 50%;">
    <h3 style="font-size: 36px; padding-left: 30px;">Solutions:</h3>
    <ol style="font-size: 32px; padding-left: 30px;">
      <li class="fragment" data-fragment-index="1">Expert Elicitation through preference learning</li>
      <li class="fragment" data-fragment-index="2">Benchmarking docking models in VS.</li>
      <li class="fragment" data-fragment-index="3">Leverage Multi-objective Bayesian Optmization</li>
    </ol>
  </div>
</div>

---
{{< slide auto-animate="" >}}
### Overview: End-to-End Pipeline
A Unified Workflow for Efficient Virtual Screening

<div style="display: flex; justify-content: center; align-items: center; margin-top: 0.5em; gap: 0.5em; flex-wrap: nowrap;">
  <div style="text-align: center; width: 18%;">
    <p style="font-size: 0.9em; margin-bottom: 0.2em;"><strong>1. Input</strong></p>
    <p style="font-size: 0.8em; margin-bottom: 0.3em;">Ligand library</p>
  </div>
  <div style="font-size: 1.2em;">&rarr;</div>

  <div style="text-align: center; width: 22%;">
    <p style="font-size: 0.8em; margin-bottom: 0.2em;"><strong>2. Docking Module</strong></p>
    <p style="font-size: 0.8em; margin-bottom: 0.3em;">Docking Model</p>
  </div>
  <div style="font-size: 1.2em;">&rarr;</div>
  
  <div style="text-align: center; width: 22%;">
    <p style="font-size: 0.9em; margin-bottom: 0.2em;"><strong>3. Active Preference Model</strong></p>
    <p style="font-size: 0.8em; margin-bottom: 0.3em;">BO + Chemist Feedback</p>
  </div>
  <div style="font-size: 1.2em;">&rarr;</div>

  <div style="text-align: center; width: 18%;">
    <p style="font-size: 0.9em; margin-bottom: 0.2em;"><strong>4. Decision</strong></p>
    <p style="font-size: 0.8em; margin-bottom: 0.3em;">Top Candidates</p>
  </div>
</div>


---
{{< slide auto-animate="" >}}
### Overview
<section>
  <figure style="text-align: center; margin-top: 100px; position: relative;">
    <img src="figures/overview_nopointer.png" alt="Overview Image" style="width: 95%; max-width: 1000px;">
    <div style="position: absolute; top: 0; left: 29.5%; width: 70%; height: 100%; background: white;" class="fragment fade-out"></div>
    <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 56%; background: white;" class="fragment fade-out"></div>
    <div style="position: absolute; bottom: 0; left: 0; width: 45.9%; height: 55%; background: white;"></div>
  </figure>

  <figure style="text-align: center; margin-top: -579px; position: relative;" class="fragment fade-in">
    <img src="figures/overview.png" alt="Overview Image" style="width: 95%; max-width: 1000px;">
    <div style="position: absolute; bottom: 0; left: 0; width: 38%; height: 55%; background: white;" class="fragment fade-out"></div>
  </figure>
</section>


---
{{< slide auto-animate="" >}}
### Outline
- Problem Setup
- <span style="opacity: 0.5;">Virtual Screening on Synthetic Functions</span>
- <span style="opacity: 0.5;">Benchmarking Docking Models</span>
- <span style="opacity: 0.5;">Chemist-guided Active Preferential Virtual Screening</span>


---
{{< slide auto-animate="" >}}
### 1. Problem Setup
For **a given protein** linked to a certain disease,
{{% fragment %}}the goal of virtual screening is to select a **few** small molecules (i.e., ligand){{% /fragment %}}
{{% fragment %}}from a library of **millions** candidates{{% /fragment %}}
{{% fragment %}}such that the selected candidate will have the **highest utility** in disease treating.{{% /fragment %}}

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 90%; margin-top: 0px; margin-left: 60px">
<img src="figures/vs.png">
<figcaption style="text-align: center; font-size: 24px; margin-top: 0px;">
    Traditional Virtual Screening Process
    <span style="font-size: 22px;"><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8188596/">(Graff et al., 2021)</a></span>
</figcaption>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 1. Problem Setup
{{% fragment %}}Traditional docking wastes vast computational resources on low-scoring compounds, though only top-ranked molecules move forward for validation.{{% /fragment %}}

{{% fragment %}}To improve efficiency, virtual screening incorporates <b>hit identification</b>, where chemists select promising compounds based on ligand properties.
{{% /fragment %}}

{{% fragment %}}However, exhaustively screening millions of candidates remains infeasible, even with expert-defined trade-offs.{{% /fragment %}}
{{% fragment %}}To solve this, we prioritize high-potential ligands while eliminating poor candidates early.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 1. Problem Setup: Active Virtual Screening

<p class="fragment"><b>Bayesian Optimization</b> iteratively refines ligand selection by balancing exploration and exploitation, often optimizing for binding affinity.</p>

<li class="fragment"><b>Step 1:</b> Start with a ligand library $\mathcal{L} = \{ l_1, \dots, l_N \}$ and an empty dataset $\mathcal{D}$. The goal is to iteratively identify ligands with high binding affinity.</li>

<li class="fragment"><b>Step 2:</b> Use an acquisition function $\alpha$ to select a subset $\mathcal{D}_i$ with high predicted binding affinity. Update $\mathcal{D}$ and remove selected ligands from $\mathcal{L}$.</li>

<li class="fragment"><b>Step 3:</b> Train a model on $\mathcal{D}$ to improve binding affinity predictions and refine ligand selection in the next iteration.</li>

{{% fragment %}}
<img src="figures/avs.png" alt="Active Virtual Screening Diagram" style="display: block; margin: 0 auto; width: 60%;" class="fragment">
<figcaption style="text-align: center; font-size: 24px; margin-top: 20px;">
    Active Virtual Screening Process
    <span style="font-size: 22px;"><a href="https://openreview.net/pdf?id=7d7Gpiyc2TU">(Graff et al., 2021)</a></span>
</figcaption>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### Outline
- <span style="opacity: 0.5;">Problem Setup</span>
- Preference Elicitation from Pairwise Comparisons
- <span style="opacity: 0.5;">Benchmarking Docking Models</span>
- <span style="opacity: 0.5;">Chemist-guided Active Virtual Screening</span>


---
{{< slide auto-animate="" >}}
### 2. Preference Elicitation from Pairwise Comparisons
{{% fragment %}}
**Problem:**  Traditional virtual screening prioritizes binding affinity but ignores other key drug properties (e.g., toxicity, solubility), making hit selection inefficient.  
{{% /fragment %}}

{{% fragment %}}
**Solution:**  Leverage preference learning to model expert intuition, capturing trade-offs between multiple ligand properties through pairwise comparisons.  
{{% /fragment %}}



---
{{< slide auto-animate="" >}}
### 2. Preference Elicitation from Pairwise Comparisons
Learning a preference model from binary data is equivalent to training a classifier.

{{% fragment %}}Given two ligands $\ell_1$ and $\ell_2$ with properties $x_{\ell_1}$ and $x_{\ell_2}$ (e.g., affinity, toxicity, solubility), we model their preference as:{{% /fragment %}}

{{% fragment %}}
```math
p(\ell_1 \succ \ell_2 \mid x_{\ell_1}, x_{\ell_2}) = \frac{e^{f(x_{\ell_1})}}{e^{f(x_{\ell_1})} + e^{f(x_{\ell_2})}}
```
{{% /fragment %}}

{{% fragment %}}
```math
= \frac{1}{1 + e^{-[f(x_{\ell_1}) - f(x_{\ell_2})]}}
```
{{% /fragment %}}

{{% fragment %}}
```math
= \sigma(f(x_{\ell_1}) - f(x_{\ell_2}))
```
{{% /fragment %}}

{{% fragment %}}where $\sigma(\cdot)$ is the sigmoid function, mapping the difference in ligand scores to a preference probability.{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 2. Preference Elicitation from Pairwise Comparisons
{{% fragment %}}
**Approach:**  We train a preference model using ligand properties (binding affinity, lipophilicity, molecular weight, half-life) as input. The utility function $f$ is modeled using a pairwise **Gaussian Process**.  
{{% /fragment %}}

{{% fragment %}}
**Synthetic:**  Generate 1,200 pairwise comparisons using synthetic functions
{{% /fragment %}}

{{% fragment %}}
**Human:** Experts rank ligands given a protein, generating pairwise comparisons.
{{% /fragment %}}

{{% fragment %}}
**Evaluation:**  
<li> 80/20 train-test split with 20-fold cross-validation. </li>
<li> Metrics: Accuracy and ROC AUC. </li>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 2. Preference Elicitation from Pairwise Comparisons
<img src="figures/preference_results.png" alt="preference_results" style="display: block; margin: 20px auto 0 auto; width: 40%;">
</figcaption> 

{{% fragment %}}
<div style="border: 2px solid #333; background-color: rgb(255, 203, 208); padding: 12px; margin-top: 15px; width: calc(100% - 400px); margin-left: 190px; text-align: center; font-size: 0.8em; font-weight: bold; border-radius: 15px;">
Preferential learning robustly recovers the latent utility function with high accuracy and AUC on both synthetic and human data.
</div>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### Outline
- <span style="opacity: 0.5;">Problem Setup</span>
- <span style="opacity: 0.5;">Preference Elicitation from Pairwise Comparisons</span>
- Benchmarking Docking Models
- <span style="opacity: 0.5;">Chemist-guided Active Virtual Screening</span>


---
{{< slide auto-animate="" >}}
### 3. Benchmarking Docking Models
{{% fragment %}}
**Problem:** While traditional docking tools like Vina (especially GPU-accelerated versions) are already fast, it's unclear how newer diffusion-based docking models compare in speed and accuracy.
{{% /fragment %}}

{{% fragment %}}
**Solution:** Benchmark our diffusion-based docking models against Vina to evaluate performance, runtime, and flexibility in modeling ligand conformations.
{{% /fragment %}}

{{% fragment %}}
**How This Fits In:** This docking module slots into our iterative loop (active screening + chemist feedback), speeding up each evaluate-and-choose cycle.
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Diffusion Model: Noise to pattern
{{% fragment %}}**Diffusion models** are a type of machine learning model used to generate data by starting with noise and gradually creating a meaningful pattern.{{% /fragment %}}

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 80%; margin-left: 80px;"> 
  <img src="figures/cat.gif" alt="Diffusion Process" style="width: 30%;"> 
  <figcaption style="text-align: center; font-size: 24px; margin-top: 10px;">Transforming noise into meaningful structures.</figcaption> 
</figure> 
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Diffusion Model: Why used for docking?
Why Use Diffusion Models for Molecules?

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 100%; margin-top: 0px;">
  <img src="figures/molecular_diffusion.png" alt="Molecular Diffusion Process" style="width: 100%;">
  <figcaption style="text-align: center; font-size: 24px; margin-top: 0px;">From random points to a structured 3D molecule.</figcaption>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Diffusion Model: Training Data
{{% fragment %}}
The PDB database is limited:
- Contains only ~17,000 protein-ligand pairs.
- Features around 5,000 unique proteins.
{{% /fragment %}}

{{% fragment %}}
For robust diffusion model training, millions of diverse data points are needed. Data augmentation enhances:
- **Ligand Diversity**: Broader chemical structure and property range.
- **Protein Diversity**: Wider variety of binding sites for better model generalization.
{{% /fragment %}}

{{% fragment %}}
Data augmentation techniques create a richer dataset, boosting model performance.
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Diffusion Model: Training Data
{{% fragment %}}
**Data Augmentation Techniques**:
<ul style="font-size: 26px"> 
    <li class="fragment"><b>Molecular Dynamics:</b> Employed 59,330 dynamic frames of 14,387 protein-ligand complexes to model ligand flexibility, amounting to 75K training data.</li> 
    <li class="fragment"><b>Data Crawling:</b> Curated 322K protein-ligand complexes, yielding 80K unique proteins.</li>
    <li class="fragment"><b>Pharmacophore Alignment:</b> Generated up to 11M pharmacophore-consistent ligand pairs, significantly expanding the ligand training data.</li> 
</ul> 
{{% /fragment %}}

{{% fragment %}}
<div style="display: flex; justify-content: space-between; margin-top: 20px;">
    <div style="width: 49%;">
        <img src="figures/md1.gif" alt="MD Simulation Example" style="width: 55%; height: auto; margin-left: 100px; margin-top: -20px;">
        <p style="text-align: center; font-size: 20px; margin-left: -130px; margin-top: -40px;">
          Figure 1: MD Simulation Trajectories  
          (<a href="https://www.deshawresearch.com/" target="_blank" style="color: #007bff; text-decoration: none;">Source</a>)
        </p>
    </div>
    <div style="width: 23%; margin-top: -45px; margin-right: 180px; white-space: nowrap;">
        <img src="figures/pharmacophore.png" alt="Pharmacophore Model Example" style="height: auto;">
        <p style="text-align: center; font-size: 20px; margin-top: -30px: ellipsis; margin-top: -30px; margin-left: -60px">
            Figure 2: Pharmacophore Modeling  
            (<a href="https://www.researchgate.net/figure/Example-of-a-shared-feature-pharmacophore-model-that-was-generated-by-LigandScout-21_fig1_251702792" target="_blank" style="color: #007bff; text-decoration: none;">Source</a>)
        </p>
    </div>
</div>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Diffusion Model: Results
{{% fragment %}}
<figure style="text-align: center; margin-top: 0px; position: relative;">
  <img src="figures/violin_affinity.png" style="width: 50%; max-width: 700px;">
  <p style="text-align: left; font-size: 20px; margin-top: 10px;">Violin plot of binding affinities (kcal/mol) for different docking models. Vina achieves the lowest median binding affinity, followed by EDM-S, while Chai exhibits the weakest binding.
</p>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Diffusion Model: Results
**Benchmark on Posebusters Dataset**:
{{% fragment %}}Posebusters: Version 1 (428 structures) and Version 2 (308 structures), released post-2021 in PDB.{{% /fragment %}}
{{% fragment %}}Performance: % of ligand pairs with $RMSD < 2 Å$ in pocket alignment.{{% /fragment %}}

{{% fragment %}}
<figure style="text-align: center; margin-top: -20px; position: relative;">
  <img src="figures/docking_results.png" alt="Docking Results" style="width: 100%; max-width: 1000px;">
</figure>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 3. Diffusion Model: Affinity vs RMSD
{{% fragment %}}Most diffusion models optimize for RMSD, but RMSD only measures geometric similarity, not binding strength.
{{% /fragment %}}

{{% fragment %}}**Why RMSD Falls Short:** Low RMSD (<2Å) can still cause steric clashes and fails to capture a ligand's regulatory potential.{{% /fragment %}}

{{% fragment %}}**Affinity Matters:** Binding affinity is a stronger indicator of drug effectiveness.
{{% /fragment %}}

{{% fragment %}}
<figure style="text-align: center; margin-top: -20px; position: relative;">
  <img src="figures/affinity_rmsd.png" alt="Docking Results" style="width: 75%; max-width: 1000px;">
</figure>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 3. Diffusion Model: Neural Search for Docking
<ul style="font-size: 0.9em;"> 
  <li class="fragment"><b>Traditional Tools</b> (e.g., Vina, Smina): ~1.5s per pose</li> 
  <li class="fragment"><b>Chai</b> (AlphaFold3-like): ~1.5 min for 5 pose</li> 
  <li class="fragment"><b>Our Diffusion Model</b>: ~10s for 128 poses</li> 
</ul>
{{% fragment %}}
<figure style="text-align: center; margin-top: -10px; position: relative;">
  <img src="figures/acc_flops.png" style="width: 53%; max-width: 800px;">
</figure>
{{% /fragment %}}

{{% fragment %}}
<div style="border: 2px solid #333; background-color: rgb(255, 203, 208); padding: 12px; margin-top: -15px; width: calc(100% - 300px); margin-left: 150px; text-align: center; font-size: 0.8em; font-weight: bold; border-radius: 15px;">
Key: Diffusion models show promise in binding affinity prediction, though physics-based methods demonstrate greater efficiency and accuracy.
</div>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### Outline
- <span style="opacity: 0.5;">Problem Setup</span>
- <span style="opacity: 0.5;">Preference Elicitation from Pairwise Comparisons</span>
- <span style="opacity: 0.5;">Benchmarking Docking Models</span>
- Chemist-guided Active Virtual Screening


---
{{< slide auto-animate="" >}}
### 4. CheapVS: Chemist-guided Active Preferential Virtual Screening Framework
{{% fragment %}}**Problem:** Traditional virtual screening is computationally expensive and inefficient, requiring exhaustive docking and manual hit selection.
{{% /fragment %}}

{{% fragment %}}**Solution:** CheapVS leverages **active learning** and **chemist-driven preferences** to efficiently prioritize high-potential ligands, reducing computational costs.
{{% /fragment %}}

{{% fragment %}}**How This Fits In:** By integrating **expert knowledge** with **machine learning**, CheapVS refines ligand selection, optimizing both efficiency and drug quality.
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4. CheapVS: Chemist-guided Active Preferential Virtual Screening Framework
<p>
  Depending on the disease and target protein, experts have <b>intuition</b> about ligand characteristics,  
  {{% fragment %}}balancing synthesizability, affinity, solubility, and side effects.{{% /fragment %}}  
  {{% fragment %}}For instance, bulky functional groups can enhance binding but reduce solubility or increase toxicity, complicating optimization.{{% /fragment %}}  
</p>


{{% fragment %}}
<div style="display: flex; justify-content: center; width: 100%; gap: -30px; margin-top: -100px;">
  <figure style="display: flex; flex-direction: column; align-items: center; width: 45%;">
    <img src="figures/lig1.png" style="width: 100%;" alt="Aff: -10.11, PSA: 67.66">
    <figcaption style="text-align: left; font-size: 20px; margin-top: -50px;">Affinity: -10.11, Solubility: 67.66</figcaption>
  </figure>
  <figure style="display: flex; flex-direction: column; align-items: center; width: 45%;">
    <img src="figures/lig2.png" style="width: 100%;" alt="Aff: -6.3, Solubility: 128.37">
    <figcaption style="text-align: left; font-size: 20px; margin-top: -50px;">Affinity: -6.3, Solubility: 128.37</figcaption>
  </figure>
</div>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 4.CheapVS
{{% fragment %}}These implicit expert knowledge, encoded as preferences over ligands, are valuable to elicit for effective virtual screening.{{% /fragment %}}
{{% fragment %}}We can leverage toolkits from the field of machine learning from human preferences to tackle this challenge.{{% /fragment %}}
<span class="fragment">
<table style="width: 90%; margin-top: 20px; border-collapse: collapse; text-align: center; font-size: 24px;">
  <tr>
    <th style="border: 1px solid #ddd; padding: 8px;">First ligand</th>
    <th style="border: 1px solid #ddd; padding: 8px;">Second ligand</th>
    <th style="border: 1px solid #ddd; padding: 8px;">Preference $(x_1 \succ x_2)$</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">[-7.81, 113.38, 0.51]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">[-8.12, 116.28, 0.47]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">0</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">[-10.45, 186.17, 0.29]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">[-8.12, 116.28, 0.47]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">1</td>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">[-6.18, 35.32, 0.83]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">[-8.12, 116.28, 0.47]</td>
    <td style="border: 1px solid #ddd; padding: 8px;">0</td>
  </tr>
</table>
<p style="font-size: 24px; text-align: center; margin-top: 15px;">
  <em>Each ligand is represented by a set of features, such as affinity, polar surface area, QED drug-likeness score</em>
</p>
</span>

---
{{< slide auto-animate="" >}}
### 4.CheapVS
<img src="figures/app.png" alt="app" style="display: block; margin: 0px auto 0 auto; width: 70%;">
<figcaption style="text-align: center; font-size: 28px; margin-top: 10px;">App for interacting with Chemists.


---
{{< slide auto-animate="" >}}
### 4.CheapVS: Overview
<figure style="text-align: center; margin-top: -50px; position: relative;" class="fragment fade-in">
  <img src="figures/presentation_overview.png" alt="Overview Image" style="width: 80%; max-width: 1000px;">
</figure>

---
{{< slide auto-animate="" >}}
### 4.CheapVS: algorithm
<p><b>Goal:</b> Identify the top <i>k</i> drug ligands for a given protein $\rho$.</p>
<li class="fragment"><b>Step 1:</b> Start with a ligand library $\mathcal{L} = \{\ell_1, \dots, \ell_N\}$ and an empty dataset $\mathcal{D}$.</li>
<li class="fragment"><b>Step 2:</b> Select a subset $\mathcal{D}_i \subset \mathcal{L}$ using the acquisition function $\alpha$ on $f$; update $\mathcal{D} \gets \mathcal{D} \cup \mathcal{D}_i$ and remove these ligands from $\mathcal{L} \gets \mathcal{L} \setminus \mathcal{D}_i$.</li>
<li class="fragment"><b>Step 3:</b> Dock ligands in $\mathcal{D}_i$ using the $p_\theta$ to estimate binding affinity $x_{\ell,\rho}^{\text{aff}}$.</li>
<li class="fragment"><b>Step 4:</b> Train a GP $g_P \sim \mathcal{GP}(\mu, k)$ on ligand fingerprints to predict affinity.</li>
<li class="fragment"><b>Step 5:</b> Compute additional ligand properties (e.g., solubility and toxicity).</li>
<li class="fragment"><b>Step 6:</b> Train a GP $f$ with pairwise preferences from ligand properties:  
  $$ f(\mathbf{x}) \sim \mathcal{GP}(\mu, k) \quad \text{and} \quad p(\ell_1 \succ \ell_2 \mid x_{\ell_1}, x_{\ell_2}) = \sigma(f(x_{\ell_1}) - f(x_{\ell_2}))$$
</li>
<li class="fragment"><b>Step 7:</b> Return to Step 2 or terminate when the computational budget is reached.</li>

---
{{< slide auto-animate="" >}}
### 4.CheapVS: Experiment Setup
BO Optimization for EGFR and DRD2
<p><b>EGFR (Epidermal Growth Factor Receptor)</b> is a protein that regulates cell growth. Mutations in EGFR are linked to cancers. <b>DRD2 (Dopamine Receptor D2)</b> is the protein target of many psychotic disorders (such as depression, schizophrenia).</p>
  <ul>
    <li class="fragment">Screening library: 100K molecules.</li>
    <li class="fragment">37 and 58 FDA-approved or late-stage drugs as goal-optimal molecules.</li>
    <li class="fragment">Expert-labeled preferences for multi-objective optimization.</li>
    <li class="fragment">Multi Objectives: 4 for EGFR, 5 for DRD2.</li>
    <li class="fragment">BO samples 1%, adds 0.5% per iteration (10 iterations, 6% total).</li>
  </ul>
</section>

---
{{< slide auto-animate="" >}}
### 4.CheapVS: Results
<img src="figures/cheapvs_main.png" alt="cheapvs_main" style="display: block; margin: 20px auto 0 auto; width: 50%;">
</figcaption> 

{{% fragment %}}
<div style="border: 2px solid #333; background-color: rgb(255, 203, 208); padding: 12px; margin-top: 15px; width: calc(100% - 280px); margin-left: 150px; text-align: center; font-size: 0.8em; font-weight: bold; border-radius: 15px;">
Key: Incorporating expert preferences outperforms affinity-only methods, emphasizing the critical role of chemical intuition in drug discovery.
</div>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4.CheapVS: GP Elicitation
<img src="figures/elicitation.png" alt="cheapvs_main" style="display: block; margin: 50px auto 0 auto; width: 55%;">
<figcaption style="text-align: left; font-size: 23.5px; margin-top: 10px;">Predictive utility scores after BO on expert preference elicitation. The box plot contrasts drugs vs. non-drugs, while heatmaps show utility across two objectives. Results align with medicinal chemistry ranges.


---
{{< slide auto-animate="" >}}
### 4.CheapVS: Multi-Objective Trade-Off
{{% fragment %}}Single-objective fails to capture trade-offs in drug discovery. Understanding how ligand properties interact helps us model expert preferences more accurately.{{% /fragment %}}

{{% fragment %}}**Approach:** Model interactions between continuous ligand properties using linear regression, incorporating higher-order terms to capture complex dependencies.
```math
y = x_1w_1 + x_2w_2 + x_1x_2w_3
```
where  $y$ is the utility score, and $x_1$, $x_2$ are ligand properties.{{% /fragment %}}

{{% fragment %}}**Evaluation Metrics:** Accuracy, ROC AUC
{{% /fragment %}}

{{% fragment %}}**Hypothesis:** Higher-order interactions improve prediction performance by capturing complex dependencies among ligand properties.
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 4.CheapVS: Multi-Objective Trade-Off Results
<img src="figures/interaction.png" alt="interaction" style="display: block; margin: 50px auto 0 auto; width: 50%;">
{{% fragment %}}<b>Key Finding:</b>
<li class="fragment"> Higher-order interactions enhance performance by capturing interdependencies. </li>  
<li class="fragment"> GPs model these dependencies, eliminating explicit interaction terms. </li>  
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 5. Conclusion
<ul>
    <li class="fragment"><b>Efficient Drug Discovery:</b> Our framework accelerates VS by leveraging preferential multi-objective BO, requiring only a small subset of ligands and expert pairwise preferences.</li>
    <li class="fragment"><b>Strong Performance:</b> Our algorithm identified <b>16/37 EGFR</b> and <b>36/57 DRD2</b> drugs, significantly outperforming baselines and demonstrating the power of chemist-guided active preferential optimization.</li>
  </ul>

---
{{< slide auto-animate="" >}}
### 6. Next steps
<ul> 
  <li class="fragment">Listwise preference for providing richer preference information</li>
  <li class="fragment">Build on top of state-of-the-art models such as AlphaFold3</li> 
</ul>