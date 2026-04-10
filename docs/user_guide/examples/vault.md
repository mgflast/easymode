## Example 4: vault complexes in D. discoideum

In this example we used **easymode**, **Relion5**, and **M** to segment, pick, and average vault complexes in *Dictyostelium discoideum* cells.

??? note "Dataset and computational resources"

    For this test we used 150 tomograms of FIB-milled *D. discoideum* cells from [EMPIAR-11845](https://www.ebi.ac.uk/empiar/EMPIAR-11845). To avoid data leakage, we re-trained the vault network with this dataset excluded from the training collection.  
    We used 4 NVIDIA RTX 4090 GPUs for most processing steps.

Vault complexes are both common and rare: you tend to see them in many different cellular cryoET datasets, but in very low copy numbers. The slime mold *D. discoideum* contains an uncommonly large number of vault complexes - or more specifically, *D. discoideum* cells from Martin Beck's group in Frankfurt contain a lot of vaults, see: [Hoffmann et al., 2025](https://www.sciencedirect.com/science/article/pii/S1097276524009936), [Tuijtel et al., 2024](https://www.science.org/doi/10.1126/sciadv.adk6285), and [Geißler et al., 2025](https://www.biorxiv.org/content/10.64898/2025.12.12.693869v1.full.pdf) for some of the data sources and cool study of vaults.


### Step 1: vault segmentation
```
easymode segment vault --data denoised --output segmented --gpu 0,1,2,3
```

### Step 2: vault picking
```
easymode pick vault --data segmented --output coordinates/vault --spacing 700 --size 50000000
```
This yielded 393 vault coordinates across the 150 tomograms. It may be worth tweaking the spacing, size, and threshold parameters in your own data. When vault complexes cluster, they can be tricky to separate into individual picks, so this might take some fine-tuning.

### Step 3: export and refinement
```
WarpTools ts_export_particles --input_directory coordinates/vault --input_pattern "*.star" --coords_angpix 10.0 --output_star relion/vault/particles.star --output_angpix 10.0 --box 128 --diameter 800 --3d --relative_output_paths
```

A single round of Relion5 Refine3D with D39 symmetry imposed, followed by refinement in M, reached a resolution of **13.6 Å**. The very cap of the vault, where the D39 symmetry breaks (see [Lövestam & Scheres, Structure (2025)](https://www.cell.com/structure/fulltext/S0969-2126(25)00262-X)), was excluded from the refinement mask. We also made sure that the lumen of the vault complex was mostly excluded from the mask, as individual complexes contain a lot of varied density inside them which can throw off alignment. Averaging with C1 using a mask that includes the lumen, for example, produces a ribosome-like blob inside a heavily missing-wedge-affected vault shell.


<div style="text-align: center;">
<img src="../../assets/vault_map.png" alt="Vault subtomogram average" style="max-width:600px; width:100%; border-radius:8px;">
<p>Subtomogram average of the <em>D. discoideum</em> vault complex at 13.6 Å resolution with D39 symmetry, obtained from 348 easymode-detected particles.</p>
</div>
