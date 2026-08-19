## Example 6: Nucleosomes in human T cells

In this example we use **easymode** and Warp and Relion to segment chromatin in human T cells, and, via 3D classification, to isolate a particle set of nucleosomes.

??? note "Dataset"
    For this example we use the data from [EMPIAR-13566](https://www.ebi.ac.uk/empiar/EMPIAR-13566/) from the paper by [Kreysing et al.](https://www.nature.com/articles/s41467-026-75087-5) (MPI Frankfurt). It consists of 14 tilt series, for which we downloaded just the raw frames and .mdoc files so that we could reconstruct the tomograms ourselves using WarpTools.

As always, we reconstructed tomograms at 10 Å/px with WarpTools (via `easymode reconstruct`) and denoised them with the IsoNet2-style general denoiser (`easymode denoise --method iso`).

For picking, we will use the `chromatin` network, which is described in more detail [here](../../models/chromatin.md).

### Step 1: chromatin segmentations
With the iso-style denoised tomograms in `denoised/`, we call `easymode segment`:
```
easymode segment chromatin --data denoised/ --output segmented/ --tta 4
```

On average this took about 2 minutes per tomogram (on 4 RTX5090 GPUs).

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1280/1162; background:#fff; border-radius:8px;">
  <source src="../../../assets/nucleosome_segmentation.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
Example segmentation of chromatin, nucleus, nuclear envelope, and membranes in a human T cell tomogram.
{: .subtitle }

### Step 2: segmentation-based picking
Next, we overpick the segmentation output. This means that we will put in far more coordinates than the number of nucleosomes we expect to actually find. The main knob here is the `--spacing` value, which sets the minimum spacing between output coordinates. Approximating nucleosomes as disk-shaped, their diameter is about 110 Å. Enforcing a spacing of a little over half this diameter would ensure that two directly adjacent nucleosomes both get picked, but that we don't place multiple coordinates inside a single nucleosome. However, the thickness of a nucleosome 'disk' is only about 70 Å. To pick two hypothetical nucleosomes that are stacked right on top of each other, we would have to set the spacing a little bit below 70 Å. So let's use a spacing of 60 Å. For the minimum particle size `--size` we use 250.000 Å³, which is a bit less than half of our nucleosome disk's volume.

```
easymode pick chromatin --data segmented/ --spacing 60.0 --size 250000 --output coordinates/chromatin/
```

The command ran in about 30 seconds and yielded 120420 particles in the 14 tomograms.

### Step 3: 3D classification
We now need to filter our particle set by finding out which particles do indeed contribute to a nucleosome map, and which do not. We will use Relion5 3D classification, so the first step is to export the particles using WarpTools:

```
WarpTools ts_export_particles --settings warp_tiltseries.settings --input_directory coordinates/chromatin/ --coords_angpix 10.0 --output_star relion/nucleosome/particles.star --output_angpix 5.0 --box 64 --diameter 250 --relative_output_paths --3d
```

When this command completes we can start trying to average something in Relion. 

**Initial model**: to begin we prefer to use an InitialModel job. If this works to generate a low resolution map of your complex of interest, it allows you to do your processing without relying on an external reference. We set up a job with 4 classes, a relatively high `tau2_fudge` regularization parameter of 4, and a particle diameter and spherical mask of 150 Å.

```
mkdir InitialModel/job001 -p
relion_refine --o InitialModel/job001/run --iter 200 --grad --denovo_3dref --i particles.star --tomograms --trajectories --ctf --K 4 --sym C1 --flatten_solvent --zero_mask --pool 3 --pad 1 --particle_diameter 150 --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 2 --auto_sampling --tau2_fudge 4 --j 32 --gpu  --pipeline_control InitialModel/job001/
```

You don't need to run this job to completion - we are only after a low resolution, approximate model for a nucleosome. After 80 iterations we noticed that class 3 had become nucleosome-like, which was good enough for a start.

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1298/1162; background:#fff; border-radius:8px;">
  <source src="../../../assets/nucleosome_initial_model.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
Class 3 from the initial model job. It isn't a good map at all, but it resembled a nucleosome, and that is good enough for a start. The estimated resolution was 23 Å.
{: .subtitle }

**3D classification**: now that we have a rough model for a nucleosome, we can use it as a reference in 3D classification. Using the original particles.star (note: we do not use the particles and poses from the initial model job) we run a 3D classification job with 3 classes, a lower tau2 value of 1.0 (which tends to work better in tomography - the earlier 4 was just because we forgot to lower it), the same 150 Å diameter, and an initial low pass cutoff of 35 Å.

```
mkdir Class3D/job002 -p
mpirun -n 5 --oversubscribe relion_refine_mpi --o Class3D/job002/run --i particles.star --ref InitialModel/job001/run_it080_class003.mrc --firstiter_cc --trust_ref_size --ini_high 35 --pool 16 --pad 2 --ctf --iter 25 --tau2_fudge 1 --particle_diameter 150 --fast_subsets --K 3 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --offset_range 5 --offset_step 2 --sym C1 --norm --scale --j 32 --gpu  --pipeline_control Class3D/job002/
```

After 25 iterations, the particle set had split up into two nucleosome-like classes (21.6% and 29.2% of particles) and one kind of nucleosome-like but also kind of bad class (49.2% of particles). 

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:960px; aspect-ratio:1330/468; background:#fff; border-radius:8px;">
  <source src="../../../assets/nucleosome_class3d.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
3D classification results. The white map (21.6% of particles) and the green map (29.2%) resemble nucleosomes. The red map (49.2%) does look a bit nucleosome-like in the side view, but is overall of poor quality.
{: .subtitle }

We decided to continue with just the 29.2% of particles from class 3, calling that selection `nucleosomes.star`. For completeness, it might have been better to include class 1 as well, or to do multiple classification jobs and choose all particles that end up in any good class. Or to do a nested classification: take the particles from the discarded classes, run classification with the same parameters, see what floats up. But for now, 29.2% of 120.420 particles is 35.000, which is still more than enough.

### Step 4: refinement in Relion
Next, we use Relion5 3D auto-refinement to refine the selected 35.000 particles, using the class 3 average from before as the reference.

```
mkdir Refine3D/job003 -p
mpirun -n 5 --oversubscribe relion_refine_mpi --o Refine3D/job003/run --auto_refine --split_random_halves --i nucleosomes.star --ref Class3D/job002/run_it025_class003.mrc --firstiter_cc --trust_ref_size --ini_high 60 --pool 16 --pad 2 --ctf --particle_diameter 150 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 3 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale --j 32 --gpu  --pipeline_control Refine3D/job003/
```
This resulted in a map with a nominal resolution of about 16 Å. It looked a bit worse than the map from before, but if we create a mask and do some postprocessing:
```
mkdir MaskCreate/job004 -p
relion_mask_create --i Refine3D/job003/run_class001.mrc --o MaskCreate/job004/mask.mrc --lowpass 15 --ini_threshold 0.4 --extend_inimask 1 --width_soft_edge 4 --j 14  --pipeline_control MaskCreate/job004/

mkdir PostProcess/job005 -p
relion_postprocess --mask MaskCreate/job004/mask.mrc --i Refine3D/job003/run_half1_class001_unfil.mrc --o PostProcess/job005/postprocess  --angpix -1 --skip_fsc_weighting  --low_pass 5  --pipeline_control PostProcess/job005/
```
<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1280/1162; background:#fff; border-radius:8px;">
  <source src="../../../assets/nucleosome_postprocess.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
Postprocessed average of 35.000 particles at a resolution of 16.3 Å.
{: .subtitle }

We end up with a map that looks decent. From here on, our processing became a bit messy: we found that the particle set at the 5.0 Å/px used in the export did not refine to higher resolution. We tried some more classification and different masks in the refinement, but we were stuck at ~ 16 Å. Not sure why this was the case, we decided to re-export the selected particle subset at a smaller pixel size of 2.5 Å/px. 

### Step 5: re-exporting the particles at a smaller pixel size
As before we use WarpTools for the export. With the `run_data.star` particle file from the Refine3D job, the coordinates are in 5.0 Å/px units.
```
WarpTools ts_export_particles --settings warp_tiltseries.settings --input_star relion/Refine3D/job003/run_data.star --coords_angpix 5.0 --output_star relion/nucleosome_2/particles.star --output_angpix 2.5 --box 96 --diameter 250 --relative_output_paths --3d
```

We then ran a classification job without alignment, 4 classes, and a tau2 value of 4.0. Note that we're in a new `relion/nucleosome_2` directory, so the job numbers have reset.
```
mkdir Class3D/job001 -p
relion_refine --o Class3D/job001/run --i particles.star --ref relion/nucleosome/PostProcess/job005/postprocess.mrc --firstiter_cc --trust_ref_size --ini_high 30 --pool 16 --pad 2 --ctf --iter 25 --tau2_fudge 4.0 --particle_diameter 160 --fast_subsets --K 4 --flatten_solvent --zero_mask --solvent_mask relion/nucleosome/MaskCreate/job004/mask.mrc --skip_align --sym C1 --norm --scale --j 32 --pipeline_control Class3D/job001/
```

All four maps were now nucleosomes, and much more detail was visible than before. We're not sure why the previous run would have gotten stuck at 16 Å rather than a bit closer to Nyquist at ~10 Å. But clearly it was the pixel size that was the problem. The maps were now at an estimated resolution of 7.6 Å (but not gold-standard, as this was a classification job).

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:960px; aspect-ratio:1174/362; background:#fff; border-radius:8px;">
  <source src="../../../assets/nucleosome_class3d_2.5A.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
The four classes after re-extracting the particles at 2.5 Å/px.
{: .subtitle }

Since the particles all look fine, we can try the same refinement job as before, now using the previously created mask.

```
mkdir Refine3D/job002 -p
mpirun -n 5 --oversubscribe relion_refine_mpi --o Refine3D/job002/run --auto_refine --split_random_halves --i particles.star --ref Class3D/job001/run_it025_class001.mrc --firstiter_cc --trust_ref_size --ini_high 20 --pool 3 --pad 2 --ctf --particle_diameter 170 --flatten_solvent --zero_mask --solvent_mask relion/nucleosome/MaskCreate/job004/mask.mrc --oversampling 1 --healpix_order 2 --auto_local_healpix_order 2 --offset_range 2 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale --j 14 --gpu  --pipeline_control Refine3D/job002/
```

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1274/1048; background:#fff; border-radius:8px;">
  <source src="../../../assets/nucleosome_refined_9.3A.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
Nucleosome average at 9.3 Å resolution (35.000 particles).
{: .subtitle }

### Step 6: the final map
With resolution improved from ~16 Å to ~9 Å since the latest classification and subset selection, it may be worth doing another classification and select only the particles that contribute to the highest resolution class. We'll do it without any alignment:
```
mkdir Class3D/job003 -p
relion_refine --o Class3D/job003/run --i Refine3D/job002/run_data.star --ref Refine3D/job002/run_class001.mrc --firstiter_cc --trust_ref_size --ini_high 30 --pool 3 --pad 2 --ctf --iter 25 --tau2_fudge 2 --particle_diameter 200 --fast_subsets --K 2 --flatten_solvent --zero_mask --solvent_mask relion/nucleosome/MaskCreate/job004/mask.mrc --skip_align --sym C1 --norm --scale --j 64 --pipeline_control Class3D/job003/
```

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:700/479; background:#fff; border-radius:8px;">
  <source src="../../../assets/nucleosome_class3d_final.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
The two classes from the final classification job: class 1 in grey, class 2 in yellow.
{: .subtitle }

Finally, we selected the particles from the best class (class 2) only, re-exported these at 2.0 Å/px, ran another 3D refinement job, and postprocessed the result (all as before), giving a final map at 7.4 Å resolution.

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1274/966; background:#fff; border-radius:8px;">
  <source src="../../../assets/nucleosome_final_map.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
The final nucleosome map at 7.4 Å resolution.
{: .subtitle }