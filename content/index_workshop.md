+++
title = "Preferential Multi-Objective Bayesian Optimization for Drug Discovery"
date = "2025-01-01"
outputs = ["Reveal"]
math=true
codeFences = true

+++
{{< slide auto-animate="" >}}
<div style="text-align: left;">
  <h3 style="font-size: 1.6em;">Preferential Multi-Objective Bayesian Optimization for Drug Discovery</h3>
</div>

<p style="font-size: 45px;text-align: center;">
Tai Dang
</p>
<p style="font-size: 30px;text-align: center;">
RHF.AI & Stanford University
</p>

---
{{< slide auto-animate="" >}}

<div style="text-align: left;">
  <h3 style="font-size: 1.6em;">Preferential Multi-Objective Bayesian Optimization for Drug Discovery</h3>
</div>

<figure style="display: flex; flex-direction: column; align-items: center; width: 60%; margin-top: 0px; margin-left: 220px">
<img src="figures/face.png">

<img src="figures/qr.png"
      style="position: absolute; bottom: -40px; right: 10px; width: 150px; height: 150px; z-index: 10;">

---
{{< slide auto-animate="" >}}
### 1. Problem Setup
For **a given protein** linked to a certain disease,
{{% fragment %}}the goal of virtual screening is to select a **few** small molecules (i.e., ligand){{% /fragment %}}
{{% fragment %}}from a library of **millions** candidates{{% /fragment %}}
{{% fragment %}}such that the selected candidate will have the **highest utility** in disease treating.{{% /fragment %}}

<figure style="display: flex; flex-direction: column; align-items: center; width: 50%; margin-top: 0px; margin-left: 300px">
<img src="figures/vs1.jpeg">
<figcaption style="text-align: center; font-size: 24px; margin-top: 0px;">
    Virtual Screening Process
    <span style="font-size: 21px;"><a href="https://pubs.acs.org/doi/full/10.1021/acs.jmedchem.3c00128">(Anastasiia, et al., 2023)</a></span>
</figcaption>

---
{{< slide auto-animate="" >}}
### Overview: Challenges in Virtual Screening
<p><b>Problem:</b> Large-scale virtual screening is computationally expensive.</p> <ul> 
<li class="fragment"><b>Computational Waste:</b> Exhaustive docking wastes resources on low-quality hits.</li> 
<li class="fragment"><b>Manual Hit Selection:</b> Slow, labor-intensive evaluation by chemists.</li> 
<li class="fragment"><b>Single-Objective Focus:</b> Prioritizing affinity ignores other critical properties.</li> 
<li class="fragment"><b>Result:</b> Wasted effort on unsuitable candidates.</li></ul>
<p class="fragment"><b>→ Need: A more efficient, expert-informed, multi-objective approach.</b></p>


---
{{< slide auto-animate="" >}}
### 1 Our Solution: Chemist-Guided Active Screening
**Core Idea**: Leverage **Preferential Multi-Objective Bayesian Optimization**.
<p class="fragment"><b>Key Innovation:</b> Guide the optimization using chemists’ intuition</p> <ul> 
<li class="fragment">Manually weighting multiple objectives is difficult & subjective.</li> 
<li class="fragment">Instead, we learn the expert's preferred trade-offs from simple pairwise choices ('Is Ligand A generally preferable to Ligand B?').</li></ul>
{{% fragment %}}**Goal**: Prioritize **high-potential ligands** early, considering multiple objectives simultaneously, guided by expert knowledge.{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 2.CheapVS: Algorithm
<div style="display: flex; align-items: flex-start; justify-content: center; gap: 10px;">
  <!-- Text Section -->
  <div style="flex: 1; font-size: 0.75em;">
    <p><b>CheapVS Loop:</b></p> 
    <ul> 
      <li class="fragment"><b>Select</b>: Choose informative ligands (Acquisition Function).</li>
      <li class="fragment"><b>Predict</b>: Get docking scores (affinity).</li>
      <li class="fragment"><b>Feedback</b>: Obtain Chemist's pairwise preferences.</li>
      <li class="fragment"><b>Learn</b>: Update multi-objective utility (GP) from preferences.</li>
      <li class="fragment"><b>Guide Selection</b>: Updated utility informs the next acquisition step.</li>
    </ul>
    <p class="fragment"><b>Output</b>: Top compounds based on learned utility.</p>
  </div>

  <!-- Image Section -->
  <div style="flex: 1.8; text-align: center;">
    <img src="figures/presentation_overview.png" style="width: 100%; max-height: 700px;">
  </div>


</div>


---
{{< slide auto-animate="" >}}
### 2.CheapVS: Experiment Setup
<p><b>Experiments on EGFR and DRD2.</b>
</p>
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
### 2.CheapVS: Results
<img src="figures/cheapvs_main_haic.png" alt="cheapvs_main" style="display: block; margin: 20px auto 0 auto; width: 90%;">
</figcaption> 

{{% fragment %}}
<div style="border: 2px solid #333; background-color: rgb(255, 203, 208); padding: 12px; margin-top: 15px; width: calc(100% - 280px); margin-left: 150px; text-align: center; font-size: 0.8em; font-weight: bold; border-radius: 15px;">
Key: Incorporating expert preferences outperforms affinity-only methods, emphasizing the critical role of chemical intuition in drug discovery.
</div>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 2. CheapVS: GP Elicitation
<img src="figures/elicitation.png" alt="cheapvs_main" style="display: block; margin: 50px auto 0 auto; width: 70%;">
<figcaption style="text-align: left; font-size: 23.5px; margin-top: 10px;">Predictive utility scores after BO on expert preference elicitation. The box plot contrasts drugs vs. non-drugs, while heatmaps show utility across two objectives. Results align with medicinal chemistry ranges.


---
{{< slide auto-animate="" >}}
### 3. Docking Efficiency Benchmark on EGFR
<ul style="font-size: 0.9em;"> 
  <li class="fragment"><b>Traditional Tools</b> (e.g., Vina, Smina): ~1.5s per pose</li> 
  <li class="fragment"><b>Chai</b> (AlphaFold3-like): ~1.5 min for 5 pose</li> 
  <li class="fragment"><b>Our Diffusion Model</b>: ~10s for 128 poses</li> 
</ul>

{{% fragment %}}
<div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-top: -10px;">
  <img src="figures/violin_affinity.png" style="width: 47%; max-width: 600px;">
  <img src="figures/acc_flops.png" style="width: 47%; max-width: 600px; margin-top: 50px;">
</div>
{{% /fragment %}}

{{% fragment %}}
<div style="border: 2px solid #333; background-color: rgb(255, 203, 208); padding: 12px; margin-top: -15px; width: calc(100% - 300px); margin-left: 150px; text-align: center; font-size: 0.8em; font-weight: bold; border-radius: 15px;">
Key: Diffusion models show promise in binding affinity prediction, though physics-based methods demonstrate greater efficiency and accuracy.
</div>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4. Thank you for listening!
- For more details, please check out this paper on<a href="https://www.arxiv.org/abs/2503.16841" target="_blank"> arXiv</a> or scan this QR code
- Come speak to me at the poster section.

<div style="text-align: center; margin-top: 20px;">
  <img src="figures/qr.png" style="width: 250px;">
</div>