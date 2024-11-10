+++
title = "Accelerate Virtual Screening with Amortized Neural Search and Multi-Objective Bayesian Optimization"
date = "2024-11-09"
outputs = ["Reveal"]
math=true
codeFences = true

+++

{{< slide auto-animate="" >}}
### Accelerate Virtual Screening 
### with Amortized Neural Search 
### and Prefential Bayesian Optimization

<div style="display: flex; justify-content: space-between; width: 30%;">
  <img src="images/SOM_vert_Web_Color_LG.png" alt="Main Logo" style="height: 85px; margin-left: -150px; margin-top: 200px"> 
  <img src="images/sail-logo.jpg" alt="Second Logo" style="height: 85px; margin-left: -180px; margin-top: 200px">
</div>

---
{{< slide auto-animate="" >}}
### 1.Introduction: Virtual Screening

For **a given protein** linked to a certain disease,
{{% fragment %}}the goal of virtual screening is to select a **few** small molecules (i.e., ligand){{% /fragment %}}
{{% fragment %}}from a library of **millions** candidates{{% /fragment %}}
{{% fragment %}}such that the selected candidate will have the **highest utility** in disease treating.{{% /fragment %}}

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 80%; margin-top: 0px; margin-left: 100px">
<img src="images/vs.png">
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 1.Introduction: Virtual Screening

<span class="fragment">For modern libraries, it is feasible scaling up to billions or even trillions of compounds enhances the reach and impact of virtual screening.</span>

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 80%; margin-top: 0px; margin-left: 100px">
<img src="images/vs.png">
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 1.Introduction: Virtual Screening

<div style="margin-top: 20px; display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 50%;">
    <h3 style="font-size: 36px;">Virtual Screening Challenges:</h3>
    <ol style="font-size: 32px;">
      <li class="fragment" data-fragment-index="1"><b>Multiple, competing objectives</b> based on unknown, hard-to-quantify expert knowledge.</li>
      <li class="fragment" data-fragment-index="2"><b>Limited budget</b> to try all ligands from the library.</li>
      <li class="fragment" data-fragment-index="3">Some objectives (such as binding affinity) are expensive to evaluate even for a single ligand.</li>
    </ol>
  </div>

  <div style="width: 50%;">
    <h3 style="font-size: 36px; padding-left: 30px;">Proposed solutions:</h3>
    <ol style="font-size: 32px; padding-left: 30px;">
      <li class="fragment" data-fragment-index="1">Actively eliciting expert preferences for virtual screening with many objectives.</li>
      <li class="fragment" data-fragment-index="2">Active Virtual Screening.</li>
      <li class="fragment" data-fragment-index="3">Neural Search Engine with diffusion model.</li>
    </ol>
  </div>
</div>


---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition
Depending on the specific disease and protein, experts have **intuition** about characteristics of candidate ligands,
{{% fragment %}}trading off various objectives such as synthesizability, affinity, solubility, and side effects.{{% /fragment %}}

{{% fragment %}}
<div style="display: flex; justify-content: center; width: 100%; gap: -30px; margin-top: -100px;">
  <figure style="display: flex; flex-direction: column; align-items: center; width: 45%;">
    <img src="images/lig1.png" style="width: 100%;" alt="Aff: -10.11, PSA: 67.66">
    <figcaption style="text-align: left; font-size: 20px; margin-top: -50px;">Affinity: -10.11, Solubility: 67.66</figcaption>
  </figure>
  <figure style="display: flex; flex-direction: column; align-items: center; width: 45%;">
    <img src="images/lig2.png" style="width: 100%;" alt="Aff: -10.11, PSA: 67.66">
    <figcaption style="text-align: left; font-size: 20px; margin-top: -50px;">Affinity: -10.11, Solubility: 67.66</figcaption>
  </figure>
</div>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition
These implicit expert knowledge, encoded as preferences over ligands, are valuable to elicit for effective virtual screening.
{{% fragment %}}We can leverage toolkits from the field of machine learning from human preferences to tackle this challenge.{{% /fragment %}}

<span class="fragment">
<table style="width: 80%; margin-top: 20px; border-collapse: collapse; text-align: center; font-size: 18px;">
  <tr>
    <th style="border: 1px solid #ddd; padding: 8px;">First ligand</th>
    <th style="border: 1px solid #ddd; padding: 8px;">Second ligand</th>
    <th style="border: 1px solid #ddd; padding: 8px;">Preference $(x_1 \succ x_2)$</th>
  </tr>
  <tr>
    <td style="border: 1px solid #ddd; padding: 8px;">[-7.81, 114.38, 0.51]</td>
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

<p style="font-size: 20px; text-align: center; margin-top: 15px;">
  <em>Each ligand is represented by a set of features, such as affinity, polar surface area, QED drug-likeness score</em>
</p>
</span>

---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition

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
### 2.Eliciting Chemical Intuition
{{% fragment %}}The latent utility function $f$ can be modeled using various approaches.{{% /fragment %}}
{{% fragment %}} One popular choice is the **Gaussian Process (GP)**, a non-parametric Bayesian method that defines a distribution over possible functions.{{% /fragment %}}

<p style="color: green;">Add a visualization for functions induced by different kernel here.</p>


---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition

{{% fragment %}}By modeling utility with a Gaussian Process, we effectively capture both the uncertainty and the non-linear relationships inherent in expert preferences.{{% /fragment %}}
{{% fragment %}}Learning GP Classifier can be done with standard machine learning toolbox such as `scikit-learn`.{{% /fragment %}}
{{% fragment %}}For example, when the synthetic oracle is the Auckley function, we obtain 95% train and test accuracy.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition

Learning chemical intuition is done in a close-loop, where the computer interacts with the chemist in an active manner.
{{% fragment %}}Starting with distribution over function $f$ condition on the current data, $p(f | D)$, our procedure includes 4 iterative steps:{{% /fragment %}}

{{% fragment %}}**Step 1:** Sample two candidate utility: $f_1 \sim p(f|D), f_2 \sim p(f|D)$ {{% /fragment %}}


{{% fragment %}}**Step 2:** Find the best ligand under each utility function:

```math
x_1 = \arg\max_{x \in \mathcal{L}} f_1(x), x_2 = \arg\max_{x \in \mathcal{L}} f_2 (x)$ 
```
{{% /fragment %}}

{{% fragment %}}
**Step 3:** Present the two candidate ligands $x_1$ and $x_2$ to the expert to obtain preference $y$.
{{% /fragment %}}

{{% fragment %}}**Step 4:** Update the model in the present of new data $\mathcal{D} \leftarrow \mathcal{D} \cup \{(x_1, x_2, y)\}$.
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition

<p style="color: green;">Add a graph showing our active elicitation method can quickly find the best candidate $x$ with a minimal amount of query.
Add experiment showing that our method is robust with complex nonlinear function in high dimensional input.</p>


---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition
We have demonstrated that our method can robustly identify candidate ligands that match a complex, latent utility function in a high-dimensional space with a minimal number of queries.

{{% fragment %}}For the next step, we aim to collaborate with experts in the lab to understand their latent utility preferences via pairwise preference elicitation for virtual screening applications.{{% /fragment %}}


---
{{< slide auto-animate="" >}}
## 3. Active Virtual Screening
{{% fragment %}}Even with the right trade-off objective elicited from expert, exhaustively screening millions of candidate from the virtual screening library is practically infeasible.{{% /fragment %}}
{{% fragment %}}To address this problem, we can choose to screen ligand that looks promising, while avoid ligand that are highly certain to be a bad candidate.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
## 3. Active Virtual Screening
<ul>
  <li class="fragment">Bayesian Optimization: Efficiently explores high-potential ligands.</li>
  <li class="fragment">Surrogate model: Gaussian Process, Neural Net, Random forest</li>
  <li class="fragment">Acquisition function: UCB, Greedy, Thompson sampling</li>
</ul>
  
{{% fragment %}}
<img src="images/avs.png" alt="Active Virtual Screening Diagram" style="display: block; margin: 0 auto; width: 65%;" class="fragment">
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search
{{% fragment %}}
Traditional physics-based docking tools (e.g., Glide, Smina) are computationally expensive to evaluate ligand affinity.
{{% /fragment %}}

<ul>
  <li class="fragment"> <b>Traditional Tools</b>: 15mins/1 pose</li>
  <li class="fragment"> <b>Our model</b>: 5s/64 poses</li>
  <li class="fragment"> <b>Chai</b>: 1 min/5 poses</li>
</ul>

{{% fragment %}}
<p>We leverage the similarity in ligand structure to accelerate the binding by training a diffusion model for molecular docking</p>
{{% /fragment %}}

<ul>
  <li class="fragment"> <b>Neural Engine</b>: Uses diffusion model for rapid docking</li>
  <li class="fragment"> <b>Utility Scoring</b>: Assesses ligands on affinity, solubility, and toxicity, etc</li>
</ul>

---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search
{{% fragment %}}
<p>Accelerate pose search further:</p>
{{% /fragment %}}

<ul>
  <li class="fragment"> Local vs blind docking</li>
  <li class="fragment"> Blind docking first during initialization, local docking during active screening process</li>
  <li class="fragment"> Obtain centroid positions through ligand initializations for faster docking time</li>
</ul>

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center;">
  <div style="display: flex; justify-content: center; width: 100%; gap: 200px;">
      <img src="images/blind.gif" style="width: 30%; margin-right: 10px;" alt="Blind Docking">
      <img src="images/local.gif" style="width: 30%;" alt="Local Docking">
  </div>
  <figcaption style="text-align: center; font-size: 20px; margin-top: -30px;">Blind Docking (left) vs. Local Docking (right)</figcaption>
</figure>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search
{{% fragment %}}
**Data Augmentation Techniques**:
<ul style="font-size: 26px"> 
    <li class="fragment"><b>Molecular Dynamics (MD) Augmentation:</b> Employed 59,330 dynamic frames of 14,387 protein-ligand complexes to model ligand flexibility, amounting to 75K training data.</li> 
    <li class="fragment"><b>Data Crawling:</b> Curated 322K protein-ligand complexes from the PDBScan22 dataset, after filtering out unnatural ligands and problematic poses, to enhance structural diversity.</li> 
    <li class="fragment"><b>Pharmacophore Alignment:</b> Generated up to 11M pharmacophore-consistent ligand pairs from the Papyrus dataset, significantly expanding the model's training data.</li> 
</ul> 
{{% /fragment %}}

{{% fragment %}}
<div style="display: flex; justify-content: space-between; margin-top: 20px;">
    <div style="width: 49%;">
        <img src="images/md.gif" alt="MD Simulation Example" style="width: 50%; height: auto; margin-left: 100px;">
        <p style="text-align: center; font-size: 20px; margin-left: 00px; margin-top: -40px;">Figure 1: MD Simulation Trajectories</p>
    </div>
    <div style="width: 49%;">
        <img src="images/pharmacophore.png" alt="Pharmacophore Model Example" style="width: 80%; height: auto;">
        <p style="text-align: center; font-size: 20px; margin-top: 80px;">Figure 2: Pharmacophore Modeling</p>
    </div>
</div>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search

{{% fragment %}}
Diffusion models are probabilistic generative models that transform data from a noisy distribution into a structured, meaningful output through a reverse process.{{% /fragment %}}

{{% fragment %}}Applying SDEs for controlled generative modeling:{{% /fragment %}}
<ul> 
    <li class="fragment"> Initiate with Gaussian noise on ligand coordinates</li> 
    <li class="fragment"> Employ Euler-Maruyama to denoise step-by-step: <ul> 
      <li>$X_{t-\Delta t} = X_t + f(X_t, t) \Delta t + g(t) \sqrt{\Delta t} \cdot \epsilon$</ul> </li>
    <li class="fragment"> Optimize ligand structure discovery through iterative sampling steps via Euler</li> 
</ul> 

---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search
{{% fragment %}} **Benchmark on Posebusters Dataset**:
{{% /fragment %}}

{{% fragment %}}Posebusters Version 1 includes 428 protein-ligand structures, and Version 2 has 308, all released to the PDB after 2021. Performance is measured by the percentage of protein-ligand pairs with pocket-aligned ligand RMSD under 2 Å.
{{% /fragment %}}

{{% fragment %}}
<figure style="text-align: center; margin-top: -20px;">
  <img src="images/docking_results.png" alt="Docking Results" style="width: 100%; max-width: 1000px;">
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 5. Putting it all together: Contrained settings

{{% fragment %}}
This graph showcases the outcome of a PCA-based virtual screening process to identify and compare ligands based on their chemical similarity.
{{% /fragment %}}

<ul style="font-size: 26px;">
  <li class="fragment">PC1 and PC2: represent the principal components derived from PCA, summarizing complex ligand properties into a comprehensible two-dimensional space.</li>
  <li class="fragment">Similarity Metric: using the Tanimoto coefficient, comparing chemical fingerprints.</li>
</ul>

{{% fragment %}}
<figure style="text-align: center;">
  <img src="images/similarity.png" alt="Virtual Screening with Multiple Query Ligands and Constrained Similarity" width="45%">
  <figcaption style="font-size: 20px; margin-top: 10px;">Virtual Screening with Multiple Query Ligands and Constrained Similarity</figcaption>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 5. Putting it all together
{{% fragment %}}We perform active virtual screening on the inferred expert utility function.{{% /fragment %}}
{{% fragment %}}Our procedure respects expert preference (both hard and soft constraint) and probabilistic model to come up with a good candidate set.{{% /fragment %}}
{{% fragment %}}To search for poses required for objectives such as affinity, we accelerate the pose search by a neural search engine.{{% /fragment %}}

---
{{< slide auto-animate="" >}}

### 5. Putting it all together
{{% fragment %}} 
**Metrics for evaluation** 
{{% /fragment %}}

{{% fragment %}}
**Regret**
{{% /fragment %}}

<ul>
  <li class="fragment"><b>Definition</b>: Difference in affinity between the best possible ligand and the top ligand found by the model within the $top_k \%$.</li>
  <li class="fragment">$\text{Regret} = A_{\text{best}} - A_{\text{model}}$</li>
</ul>

{{% fragment %}}
**Percent of Best Ligand Found**
{{% /fragment %}}

<ul>
  <li class="fragment"><b>Definition</b>: Percentage of screened ligands close in affinity to the best possible ligand. ($top_k \%$)</li>
</ul>

---
{{< slide auto-animate="" >}}
### 5. Putting it all together: Screening Results
{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center;">
  <div style="display: flex; justify-content: center; width: 100%; gap: 70px;">
      <img src="images/percent.png" style="width: 33%; max-width: 450px;">
      <img src="images/regret.png" style="width: 33%; max-width: 450px;">
  </div>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 5. Putting it all together: Screening Results
{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center;">
  <div style="display: flex; justify-content: center; width: 100%; gap: 10px; margin-top: -20px;">
      <img src="images/time.png" style="width: 70%; max-width: 650px;">
  </div>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 5. Putting it all together: Next steps
<ul>
  <li class="fragment"> Run virtual screening on bigger library (100k, 1M) compounds</li>
  <li class="fragment"> Improve on performance of diffusion model</li>
</ul>