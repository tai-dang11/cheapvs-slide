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
Tai Dang<sup>1,3</sup>,Hung Pham<sup>2</sup>,Sang Truong<sup>1</sup>, Ari Glenn<sup>1</sup>, Wendy Nguyen<sup>3</sup>, Edward A Pham<sup>1</sup>, Jeffrey S. Glenn<sup>1</sup>, Sanmi Koyejo<sup>1</sup>, Thang Luong<sup>1</sup> 
</p>
<p style="font-size: 30px;">
<sup>1</sup>Stanford University, <sup>2</sup>Imperial College London, <sup>3</sup>RHF.AI
</p>

<div style="display: flex; justify-content: space-between; width: 30%;">
  <img src="images/rhf.png" alt="Third Logo" style="height: 85px; margin-left: 10px; margin-top: 250px">
  <img src="images/SOM_vert_Web_Color_LG.png" alt="Main Logo" style="height: 85px; margin-left: 60px; margin-top: 250px"> 
  <img src="images/sail-logo.jpg" alt="Second Logo" style="height: 85px; margin-left: 75px; margin-top: 250px">
</div>

---
{{< slide auto-animate="" >}}
### Outline
- Problem Setup
- Virtual Screening on Synthetic Functions
- Chemist-guided Active Preferential Virtual Screening
- Protein-ligand docking with Diffusion Models


---
{{< slide auto-animate="" >}}
### Overview: Problem setup
Traditional large-scale docking consumes extensive computational resources, making the process slow and costly. 
{{% fragment %}}Chemists manually evaluate thousands of molecules, selecting hits based on intuition. {{% /fragment %}}
{{% fragment %}}This step is critical yet slow, forming a major bottleneck.{{% /fragment %}}{{% fragment %}}Experts must balance multiple drug properties (e.g., efficacy, safety, solubility),{{% /fragment %}}
{{% fragment %}}complicating decision-making and extending timelines in drug development.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### Overview
<section>
  <figure style="text-align: center; margin-top: 100px; position: relative;">
    <img src="figures/overview_nopointer.png" alt="Overview Image" style="width: 95%; max-width: 1000px;">
    <div style="position: absolute; top: 0; left: 28.5%; width: 70%; height: 100%; background: white;" class="fragment fade-out"></div>
    <div style="position: absolute; bottom: 0; left: 0; width: 100%; height: 56%; background: white;" class="fragment fade-out"></div>
    <div style="position: absolute; bottom: 0; left: 0; width: 45.9%; height: 55%; background: white;"></div>
  </figure>

  <figure style="text-align: center; margin-top: -579px; position: relative;" class="fragment fade-in">
    <img src="figures/overview.png" alt="Overview Image" style="width: 95%; max-width: 1000px;">
    <div style="position: absolute; bottom: 0; left: 0; width: 37.3%; height: 55%; background: white;" class="fragment fade-out"></div>
  </figure>
</section>

---
{{< slide auto-animate="" >}}
### Overview: Streamlining Virtual Screening with Advanced Techniques

<div style="margin-top: 20px; display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 50%;">
    <h3 style="font-size: 36px;">Challenges:</h3>
    <ol style="font-size: 32px;">
      <li class="fragment" data-fragment-index="1">Conducting virtual screening within budget constraints.</li>
      <li class="fragment" data-fragment-index="2">Incorporating multiple objectives beyond affinity.</li>
      <li class="fragment" data-fragment-index="3">Training efficient docking models with limited resources.</li>
    </ol>
  </div>

  <div style="width: 50%;">
    <h3 style="font-size: 36px; padding-left: 30px;">Solutions:</h3>
    <ol style="font-size: 32px; padding-left: 30px;">
      <li class="fragment" data-fragment-index="1">Optimize ligand selection through active screening.</li>
      <li class="fragment" data-fragment-index="2">Utilize Preferential Multi-Objective Bayesian Optimization.</li>
      <li class="fragment" data-fragment-index="3">Enhance docking model through diffusion model.</li>
    </ol>
  </div>
</div>


---
{{< slide auto-animate="" >}}
### Outline
- Problem Setup
- <span style="opacity: 0.5;">Virtual Screening on Synthetic Functions</span>
- <span style="opacity: 0.5;">Chemist-guided Active Preferential Virtual Screening</span>
- <span style="opacity: 0.5;">Protein-ligand docking with Diffusion Models</span>


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
<figcaption style="text-align: center; font-size: 20px; margin-top: 0px;">
    Traditional Virtual Screening Process
    <br>
    <span style="font-size: 16px;">Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8188596/">Graff et al., 2021</a></span>
</figcaption>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 1. Problem Setup
{{% fragment %}}Docking consumes vast computational resources on low-scoring compounds, though only top-ranked molecules advance for validation.{{% /fragment %}}
{{% fragment %}}To refine this process, virtual screening includes <b>hit identification</b>, where chemists select promising compounds based on ligand properties.{{% /fragment %}}

{{% fragment %}}Even with expert-defined trade-offs, exhaustively screening millions of candidates is infeasible.{{% /fragment %}}
{{% fragment %}}To address this, we prioritize promising ligands while avoiding those likely to be poor candidates.{{% /fragment %}}  

---
{{< slide auto-animate="" >}}
### 2. Active Virtual Screening
{{% fragment %}}To prioritize high-potential ligands, we use **Preferential Bayesian Optimization**,{{% /fragment %}}
{{% fragment %}}an approach that balances exploring new candidates and exploiting known promising ones to efficiently find optimal solutions.{{% /fragment %}}
  
{{% fragment %}}
<img src="figures/cheapvs1.png" alt="Active Virtual Screening Diagram" style="display: block; margin: 0 auto; width: 65%;" class="fragment">
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 1. Problem Setup
Learning a preference model from binary preference data can be viewed as learning a classifier. 

  {{% fragment %}}
  ```math
  p(y \mid x_1, x_2; f) = \frac{e^{f(x_1)}}{e^{f(x_1)} + e^{f(x_2)}}
  ```
  {{% /fragment %}}

  {{% fragment %}}
  ```math
  = \frac{1}{1 + e^{-[f(x_1)-f(x_2)]}}
  ```
  {{% /fragment %}}

  {{% fragment %}}
  ```math
  = \sigma(f(x_1)-f(x_2))
  ```
  {{% /fragment %}}

{{% fragment %}}where $\sigma(\cdot)$ is the sigmoid function.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 1. Problem Setup
<p><b>Goal:</b> Identify the top <i>k</i> candidate ligands for a given protein from a library.</p>
<li class="fragment"><b>Step 1:</b> Start with a ligand library $\mathcal{L} = \{ l_1, \dots, l_N \}$ and a empty dataset $\mathcal{D}$.  </li>
<li class="fragment"><b>Step 2:</b> Select a subset $\mathcal{D}_i = \{ l_i, l_k, l_f, \dots \}$  using an acquisition function $\alpha$, update the dataset $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_i$
  and remove these ligands from $\mathcal{L} \leftarrow \mathcal{L} \setminus \mathcal{D}_i$.
</li>
<li class="fragment"><b>Step 3:</b> Dock ligands in $\mathcal{D}_i$ using a docking model $\theta$ to estimate binding affinity.</li>
<li class="fragment"><b>Step 4:</b> Train a GP $g$ on ligand fingerprints $t$ to predict affinity: $ g(t) \sim \mathcal{GP}(\mu, k) $
</li>
<li class="fragment"><b>Step 5:</b> Compute additional ligand properties (e.g., solubility, toxicity).</li>
<li class="fragment"><b>Step 6:</b> Train a GP $f$ with pairwise preferences from ligand properties:  
  $$ f(\mathbf{x}) \sim \mathcal{GP}(\mu, k) \quad \text{and} \quad p(y=1 \mid \mathbf{x}_1, \mathbf{x}_2) = \sigma(f(\mathbf{x}_1) - f(\mathbf{x}_2)) $$
</li>
<li class="fragment"><b>Step 7:</b> Return to Step 2 or terminate if the computational budget is reached.</li>

---
{{< slide auto-animate="" >}}
### Overview
<figure style="text-align: center; margin-top: 0px; position: relative;" class="fragment fade-in">
  <img src="figures/overview.png" alt="Overview Image" style="width: 95%; max-width: 1000px;">
</figure>

---
{{< slide auto-animate="" >}}
### Outline
- <span style="opacity: 0.5;">Problem Setup</span>
- Virtual Screening on Synthetic Functions
- <span style="opacity: 0.5;">Chemist-guided Active Preferential Virtual Screening</span>
- <span style="opacity: 0.5;">Protein-ligand docking with Diffusion Models</span>

---
{{< slide auto-animate="" >}}
### 2. Active Virtual Screening: Experiment Setup
Synthetic Utility Landscapes
<ul>
  <li class="fragment">Testing synthetic functions helps refine the approach before human experiments.</li>
  <li class="fragment">Benchmarks: Ackley, Alpine1, Hartmann, Dropwave, Qeifail, Levy.</li>
  <li class="fragment">Each benchmark outputs a scalar utility for preference-based learning.</li>
  <li class="fragment">Simulated objectives: affinity, rotatable bonds, molecular weight, LogP.</li>
  <li class="fragment">20K-ligand subset used for computational efficiency.</li>
</ul>
{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center;">
  <div style="display: flex; justify-content: center; width: 100%; gap: 70px;">
      <img src="figures/synthetic_funcs.png" style="width: 100%; max-width: 1000px;">
  </div>
  <figcaption style="text-align: center; font-size: 20px; margin-top: 0px;">Synthetic functions landscape.</figcaption>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 2. Active Virtual Screening: Evaluation Metrics
**Metrics for evaluation** 
{{% fragment %}}
**Regret**
{{% /fragment %}}
<ul>
  <li class="fragment">$R_i = U^* - U(i)$</li>
  <li class="fragment">Where $U^*$ is the highest possible utility in the library and $U(i)$ is the highest utility found by the model at iteration $i$</li>
</ul>

{{% fragment %}}
**Percent of Best Ligand Found**
{{% /fragment %}}

<ul>
  <li class="fragment"><b>Definition</b>: Percentage of screened ligands close in affinity to the best possible ligand. ($top_k \%$)</li>
</ul>

---
{{< slide auto-animate="" >}}
### 2. Active Virtual Screening: Screening Results
{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center;">
  <div style="display: flex; justify-content: center; width: 100%; gap: 70px;">
      <img src="figures/regret.png" style="width: 70%; max-width: 1000px;">
  </div>
  <figcaption style="text-align: center; font-size: 20px; margin-top: 0px;">Preferential Multi-Objective Optimization results on synthetic functions with different docking models.</figcaption>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### Outline
- <span style="opacity: 0.5;">Problem Setup</span>
- <span style="opacity: 0.5;">Virtual Screening on Synthetic Functions</span>
- Chemist-guided Active Preferential Virtual Screening
- <span style="opacity: 0.5;">Protein-ligand docking with Diffusion Models</span>

---
{{< slide auto-animate="" >}}
### 3.Chemist-guided Active Preferential Virtual Screening: Overview
{{% fragment %}}In drug discovery, selecting candidate ligands goes beyond targeting high-affinity molecules. {{% /fragment %}}
{{% fragment %}}Experts use their deep chemical intuition to balance competing properties such as synthesizability, solubility, and potential side effects. {{% /fragment %}}
{{% fragment %}}This approach ensures ligands are not only effective but also practical and safe for therapeutic use.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3.CheapVS
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
### 3.CheapVS
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
### 3.CheapVS
<img src="figures/app.png" alt="app" style="display: block; margin: 0px auto 0 auto; width: 70%;">
<figcaption style="text-align: center; font-size: 28px; margin-top: 10px;">App for interacting with Chemists.


---
{{< slide auto-animate="" >}}
### 3.CheapVS: Experiment Setup
BO Optimization for EGFR
<p><b>EGFR (Epidermal Growth Factor Receptor)</b> is a protein that regulates cell growth. Mutations in EGFR are linked to cancers.</p>
  <ul>
    <li class="fragment">Screening library: 100K molecules.</li>
    <li class="fragment">37 FDA-approved or late-stage drugs as goal-optimal molecules.</li>
    <li class="fragment">Expert-labeled preferences for multi-objective optimization.</li>
    <li class="fragment">4 Objectives: Affinity, Molecular Weight, Lipophilicity, Half-life.</li>
    <li class="fragment">BO samples 1%, adds 0.5% per iteration (10 iterations, 6% total).</li>
  </ul>
</section>


---
{{< slide auto-animate="" >}}
### 3.CheapVS: Results
<img src="figures/cheapvs_main.png" alt="cheapvs_main" style="display: block; margin: 50px auto 0 auto; width: 60%;">
<figcaption style="text-align: left; font-size: 23.5px; margin-top: 10px;">Performance of CheapVS in identifying EGFR drugs. The plot compares docking models and objectives. Multi-objective optimization outperforms the rest, identifying up to 16 of 37 approved drugs.
</figcaption> 

---
{{< slide auto-animate="" >}}
### 3.CheapVS: GP Elicitation
<img src="figures/elicitation.png" alt="cheapvs_main" style="display: block; margin: 50px auto 0 auto; width: 60%;">
<figcaption style="text-align: left; font-size: 23.5px; margin-top: 10px;">Predictive utility scores after BO on expert preference elicitation. The box plot contrasts drugs vs. non-drugs, while heatmaps show utility across two objectives. Results align with medicinal chemistry ranges.

---
{{< slide auto-animate="" >}}
### Outline
- <span style="opacity: 0.5;">Problem Setup</span>
- <span style="opacity: 0.5;">Virtual Screening on Synthetic Functions</span>
- <span style="opacity: 0.5;">Chemist-guided Active Preferential Virtual Screening</span>
- Protein-ligand docking with Diffusion Models

---
{{< slide auto-animate="" >}}
### 4. Diffusion Model: Noise to pattern
{{% fragment %}}**Diffusion models** are a type of machine learning model used to generate data by starting with noise and gradually creating a meaningful pattern.{{% /fragment %}}

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 80%; margin-left: 80px;"> 
  <img src="figures/cat.gif" alt="Diffusion Process" style="width: 30%;"> 
  <figcaption style="text-align: center; font-size: 24px; margin-top: 10px;">Transforming noise into meaningful structures.</figcaption> 
</figure> 
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4. Diffusion Model: Why used for docking?
Why Use Diffusion Models for Molecules?

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 100%; margin-top: 0px;">
  <img src="figures/molecular_diffusion.png" alt="Molecular Diffusion Process" style="width: 100%;">
  <figcaption style="text-align: center; font-size: 24px; margin-top: 0px;">From random points to a structured 3D molecule.</figcaption>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4. Diffusion Model: Training Data
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
### 4. Diffusion Model: Training Data
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
        <p style="text-align: center; font-size: 20px; margin-left: -130px; margin-top: -40px;">Figure 1: MD Simulation Trajectories</p>
    </div>
    <div style="width: 49%;">
        <img src="figures/pharmacophore.png" alt="Pharmacophore Model Example" style="width: 80%; height: auto;">
        <p style="text-align: center; font-size: 20px; margin-top: 80px;">Figure 2: Pharmacophore Modeling</p>
    </div>
</div>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 4. Diffusion Model: Results
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
### 4. Diffusion Model: Neural Search for Docking
{{% fragment %}}Traditional docking tools are slow, limiting the efficiency of application of virtual screening.
{{% /fragment %}}

<ul> 
  <li class="fragment"><b>Traditional Tools</b> (e.g., Vina, Smina): ~5mins per pose</li> 
  <li class="fragment"><b>Chai</b> (AlphaFold3-like): ~1 mins for 1 pose (5x faster)</li> 
  <li class="fragment"><b>Our Diffusion Model</b>: ~10s for 128 poses (90x faster)</li> 
</ul>

---
{{< slide auto-animate="" >}}
### 5. Conclusion
<ul>
    <li class="fragment"><b>Efficient Drug Discovery:</b> Our framework accelerates VS by leveraging preferential multi-objective BO, requiring only a small subset of ligands and expert pairwise preferences.</li>
    <li class="fragment"><b>Strong Performance:</b> Our algorithm successfully identified 16/37 drugs, significantly outperforming baseline methods, highlighting the power of preference-based optimization.</li>
</ul>

---
{{< slide auto-animate="" >}}
### 6. Next steps
<ul> 
  <li class="fragment">Listwise preference for providing richer preference information</li>
  <li class="fragment">Build on top of state-of-the-art models such as AlphaFold3</li> 
</ul>