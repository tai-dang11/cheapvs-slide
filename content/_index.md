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
  <img src="images/SOM_vert_Web_Color_LG.png" alt="Main Logo" style="height: 85px; margin-left: 0px; margin-top: 200px"> 
  <img src="images/sail-logo.jpg" alt="Second Logo" style="height: 85px; margin-left: 50px; margin-top: 200px">
</div>

---
{{< slide auto-animate="" >}}
### 1.Introduction: Virtual Screening

For **a given protein** linked to a certain disease,
{{% fragment %}}the goal of virtual screening is to select a **few** small molecules (i.e., ligand){{% /fragment %}}
{{% fragment %}}from a library of **millions** candidates{{% /fragment %}}
{{% fragment %}}such that the selected candidate will have the **highest utility** in disease treating.{{% /fragment %}}

{{% fragment %}}For modern libraries, it is feasible scaling up to billions or even trillions of compounds enhances the reach and impact of virtual screening.{{% /fragment %}}


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
    <img src="images/lig2.png" style="width: 100%;" alt="Aff: -6.3, Solubility: 128.37">
    <figcaption style="text-align: left; font-size: 20px; margin-top: -50px;">Affinity: -6.3, Solubility: 128.37</figcaption>
  </figure>
</div>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition
These implicit expert knowledge, encoded as preferences over ligands, are valuable to elicit for effective virtual screening.
{{% fragment %}}We can leverage toolkits from the field of machine learning from human preferences to tackle this challenge.{{% /fragment %}}

<span class="fragment">
<table style="width: 90%; margin-top: 20px; border-collapse: collapse; text-align: center; font-size: 24px;">
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

<p style="font-size: 24px; text-align: center; margin-top: 15px;">
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

<!-- <p style="color: green;">Add a visualization for functions induced by different kernel here.</p> -->


---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition

{{% fragment %}}By modeling utility with a Gaussian Process, we effectively capture both the uncertainty and the non-linear relationships inherent in expert preferences.{{% /fragment %}}
{{% fragment %}}Learning GP Classifier can be done with standard machine learning toolbox such as `scikit-learn`.{{% /fragment %}}
{{% fragment %}}For example, when the synthetic oracle is the Auckley function, we obtain 85% train and test accuracy.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition

Learning chemical intuition is done in a close-loop, where the computer interacts with the chemist in an active manner.
{{% fragment %}}Starting with distribution over function $f$ condition on the current data, $p(f | D)$, our procedure includes 4 iterative steps:{{% /fragment %}}

{{% fragment %}}**Step 1:** Sample two candidate utility: $f_1 \sim p(f|D), f_2 \sim p(f|D)$ {{% /fragment %}}


{{% fragment %}}**Step 2:** Find the best ligand under each utility function:

```math
x_1 = \arg\max_{x \in \mathcal{L}} f_1(x), x_2 = \arg\max_{x \in \mathcal{L}} f_2 (x)
```
{{% /fragment %}}

{{% fragment %}}
**Step 3:** Present the two candidate ligands $x_1$ and $x_2$ to the expert to obtain preference $y$.
{{% /fragment %}}

{{% fragment %}}**Step 4:** Update the model in the present of new data $\mathcal{D} \leftarrow \mathcal{D} \cup \{(x_1, x_2, y)\}$.
{{% /fragment %}}

<!-- ---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition
{{% fragment %}}**Step 1:** Initialize
<ul>
  <li class="fragment">Collect a small set of expert preference data to train a GP surrogate model.</li>
</ul>
{{% /fragment %}}

{{% fragment %}}**Step 2:** Predict & Select
<ul>
  <li class="fragment">Use the GP to predict preferences for unlabeled candidates.</li>
  <li class="fragment">Select top options via acquisition function (i.e Thompson Sampling, UCB).</li>
</ul>
{{% /fragment %}}

{{% fragment %}}**Step 3:** Iterate
<ul>
  <li class="fragment">Refine the surrogate model with the limited initial data to efficiently identify the most preferred solutions.</li>
</ul>
{{% /fragment %}} -->

---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition

<!-- <p style="color: green;">Add a graph showing our active elicitation method can quickly find the best candidate $x$ with a minimal amount of query.
Add experiment showing that our method is robust with complex nonlinear function in high dimensional input.</p> -->
{{% fragment %}}
<img src="images/elicatation_acc.png" alt="Active Virtual Screening Diagram" style="display: block; margin: 0 auto; width: 55%;" class="fragment">
<p style="text-align: center; font-size: 20px">Observed accuracy plot for high-dimensional data with objectives: QED, affinity, polar surface area, and molecular weight.</p>

{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 2.Eliciting Chemical Intuition
We have demonstrated that our method can robustly identify candidate ligands that match a complex, latent utility function in a high-dimensional space with a minimal number of queries.

{{% fragment %}}For the next step, we aim to collaborate with experts in the lab to understand their latent utility preferences via pairwise preference elicitation for virtual screening applications.{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 3. Active Virtual Screening
{{% fragment %}}Even with the right trade-off objective elicited from expert, exhaustively screening millions of candidate from the virtual screening library is practically infeasible.{{% /fragment %}}
{{% fragment %}}To address this problem, we can choose to screen ligand that looks promising, while avoid ligand that are highly certain to be a bad candidate.{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 3. Active Virtual Screening
{{% fragment %}}To prioritize high-potential ligands, we use **Bayesian Optimization**,{{% /fragment %}}
{{% fragment %}}an approach that balances exploring new candidates and exploiting known promising ones to efficiently find optimal solutions.{{% /fragment %}}

<ul>
  <li class="fragment">Surrogate model: Gaussian Process, Neural Net, Random forest</li>
  <li class="fragment">Acquisition function: UCB, Greedy, Thompson sampling</li>
</ul>
  
{{% fragment %}}
<img src="images/avs.png" alt="Active Virtual Screening Diagram" style="display: block; margin: 0 auto; width: 65%;" class="fragment">
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search: Diffusion Model
{{% fragment %}}For docking tasks, we use **diffusion models**, rather than traditional tools like Glide or Smina, to improve speed and efficiency.{{% /fragment %}}

{{% fragment %}}**Diffusion models** are a type of machine learning model used to generate data by starting with noise and gradually creating a meaningful pattern.{{% /fragment %}}

<ul>
  <li class="fragment">Begin with a "noisy" version of data (e.g., a blurry image)</li>
  <li class="fragment"><b>Remove Noise step-by-step</b>, making the data clearer over time</li>
  <li class="fragment">The final result is a well-defined, meaningful pattern (e.g., a clear image)</li>
</ul>

{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 80%; margin-left: 50px;">
  <img src="images/cat.webp" alt="Diffusion Process" style="width: 30%;">
  <figcaption style="text-align: center; font-size: 24px; margin-top: 10px;">Diffusion sampling process.</figcaption>
</figure>
{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search: Diffusion Model
Why Use Diffusion Models for Molecules?
<ul>
  <li class="fragment">Starting with random points in 3D space, like a cloud of dots.</li>
  <li class="fragment">The diffusion model gradually arranges these points, step-by-step, into a meaningful 3D molecular structure.</li>
  <li class="fragment">The result is a 3D structure that represents a molecule.</li>
</ul>
{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center; width: 100%; margin-top: 0px;">
  <img src="images/molecular_diffusion.png" alt="Molecular Diffusion Process" style="width: 100%;">
  <figcaption style="text-align: center; font-size: 24px; margin-top: 0px;">From random points to a structured 3D molecule.</figcaption>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search: Training Data
{{% fragment %}}
The PDB database alone is limited:
<ul>
  <li class="fragment">Contains only ~17,000 protein-ligand pairs</li>
  <li class="fragment">Limited protein diversity, with around 5,000 unique proteins</li>
</ul>
{{% /fragment %}}

{{% fragment %}}
To train a robust diffusion model, **millions of diverse data points** are essential. Data augmentation expands:
<ul>
  <li class="fragment"><b>Ligand Diversity</b>: Includes a wider range of chemical structures and properties</li>
  <li class="fragment"><b>Protein Diversity</b>: Covers various binding sites, increasing model generalization</li>
</ul>
{{% /fragment %}}

{{% fragment %}}
Data augmentation techniques allow us to build a richer, more comprehensive dataset to improve model accuracy and performance.
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search: Training Data
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
        <img src="images/md1.gif" alt="MD Simulation Example" style="width: 55%; height: auto; margin-left: 100px; margin-top: -20px;">
        <p style="text-align: center; font-size: 20px; margin-left: -130px; margin-top: -40px;">Figure 1: MD Simulation Trajectories</p>
    </div>
    <div style="width: 49%;">
        <img src="images/pharmacophore.png" alt="Pharmacophore Model Example" style="width: 80%; height: auto;">
        <p style="text-align: center; font-size: 20px; margin-top: 80px;">Figure 2: Pharmacophore Modeling</p>
    </div>
</div>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search: Training Results
**Benchmark on Posebusters Dataset**:

{{% fragment %}}Posebusters Version 1 includes 428 protein-ligand structures, and Version 2 has 308, all released to the PDB after 2021. {{% /fragment %}}
{{% fragment %}}Performance is measured by the percentage of protein-ligand pairs with pocket-aligned ligand $RMSD<2 Å$.{{% /fragment %}}

{{% fragment %}}
<figure style="text-align: center; margin-top: -20px;">
  <img src="images/docking_results.png" alt="Docking Results" style="width: 100%; max-width: 1000px;">
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search
{{% fragment %}}Traditional docking tools are slow, making it challenging for virtual screening.{{% /fragment %}}
{{% fragment %}}For example, evaluating a single ligand's binding to a protein can take significant time.{{% /fragment %}}

<ul>
  <li class="fragment"> <b>Traditional Tools</b> (e.g., Glide, Smina): Up to 15 mins per pose</li>
  <li class="fragment"> <b>Chai</b>: 1 min for 5 poses</li>
</ul>

{{% fragment %}}
Our diffusion model accelerates docking by learning from patterns in ligand structures, making it possible to quickly evaluate many poses with high accuracy
{{% /fragment %}}

<ul>
  <li class="fragment"> <b>Neural Engine</b>: Leverages diffusion models to speed up docking by recognizing molecular patterns (~5 seconds for 64 poses)</li>
  <li class="fragment"> <b>Utility Scoring</b>: Evaluates ligands based on multiple criteria, such as affinity, solubility, and toxicity</li>
</ul>

---
{{< slide auto-animate="" >}}
### 4. Neural Diffusion Search
{{% fragment %}}To speed up virtual screening, we begin with blind docking to find potential binding sites, {{% /fragment %}}
{{% fragment %}}then switch to local docking on these areas during active screening. {{% /fragment %}}
{{% fragment %}}This approach, along with optimized ligand initialization at key positions, significantly reduces computation time while maintaining accuracy (~2 seconds for 64 poses).{{% /fragment %}}


{{% fragment %}}
<figure style="display: flex; flex-direction: column; align-items: center;">
  <div style="display: flex; justify-content: center; width: 100%; gap: 200px;">
      <img src="images/blind.gif" style="width: 30%; margin-right: 10px;" alt="Blind Docking">
      <img src="images/local.gif" style="width: 30%;" alt="Local Docking">
  </div>
  <figcaption style="text-align: center; font-size: 20px; margin-top: 0px;">Blind Docking (left) vs. Local Docking (right)</figcaption>
</figure>
{{% /fragment %}}

---
{{< slide auto-animate="" >}}
### 5. Putting it all together: Contrained settings
{{% fragment %}}In virtual screening, our goal is to find effective ligands efficiently. Traditional methods can be slow, especially with large, diverse libraries. {{% /fragment %}}
{{% fragment %}}If identified ligands are too structurally unique, they may be difficult or impossible to synthesize for chemists. {{% /fragment %}}
{{% fragment %}}By using **constrained settings**, we can focus on ligands with desirable features that are also more likely to be synthetically accessible.{{% /fragment %}}


---
{{< slide auto-animate="" >}}
### 5. Putting it all together: Contrained settings
{{% fragment %}}Using constrained settings, we can limit our search to clusters of chemically similar ligands, {{% /fragment %}}
{{% fragment %}}increasing the speed and accuracy of our screening while reducing computational demands.{{% /fragment %}}

{{% fragment %}}
<figure style="text-align: center;">
  <img src="images/similarity.png" alt="PCA-Based Virtual Screening" width="40%">
  <figcaption style="font-size: 20px">
    PC1 and PC2 calculated from PCA simplify ligand features in 2D; Tanimoto coefficient groups similar ligands for targeted screening.
  </figcaption>
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