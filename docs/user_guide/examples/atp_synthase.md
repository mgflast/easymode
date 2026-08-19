## Example 5: ATP synthase in Polytomella spp.

In this example we use **easymode**, Warp/Relion/M, and subboxing to segment, pick, and average *Polytomella sp.* ATP synthase.

??? note "Dataset"
    For this example we used a curated set of 100 high quality tilt series of FIB-milled *Polytomella sp.* lamellae from [Dietrich et al., Science (2024)](https://www.science.org/doi/10.1126/science.adp4640), generously shared with us by Lea Dietrich and Andre Schwartz (Max Planck Institute for Brain Research, Frankfurt) for us to use in validation of the network. The data is not yet available online, but will soon be deposited onto EMPIAR by the original authors.

We reconstructed the tomograms at 10 Å/px as per usual with WarpTools (via `easymode reconstruct`) and denoised them with the IsoNet2-style general denoiser (`easymode denoise --method iso`).

ATP synthase in *Polytomella* is a dimer, with two F₁Fo complexes connected by the *peripheral stalk* which binds them together rigidly. This C2 symmetry is helpful to achieve a higher resolution in subtomogram averaging, and can be used in different ways. One way is to pick the dimer and refine it with C2 symmetry imposed. Another way is to pick a monomer first, generate an initial average, and then use 'subboxing' to add the second monomer into the particle selection (and then discard possible duplicates).

In this tutorial we use the subboxing approach. Because the ATP synthase dimer has a very non-globular shape, segmentation-based picking in blob mode (see the [picking page](../functions/picking.md)), which places coordinates at the deepest points of connected components in the segmentation-derived mesh, will yield picks for individual monomers, not for dimers. Subboxing is a very useful tool in many cases, so the demonstration here can also be helpful when averaging other targets. We developed a ChimeraX plugin to perform the subboxing, available at [chimerax-subboxer](https://github.com/mgflast/ChimeraX-subbox), which we use here.

### Step 1: ATP synthase segmentation
We use the pretrained ATP synthase segmentation network to segment our 100 IsoNet2-style denoised tomograms:
```
easymode segment atp_synthase --data denoised/ --output segmented/
```
Using 4 RTX 5090 GPUs this took about an hour and a half. 

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1280/804; background:#fff; border-radius:8px;">
  <source src="../../../assets/atp_synthase_segmentation.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
Example segmentation of ATP synthase (blue), mitochondrion (red), and membranes (gray) in a *Polytomella sp.* tomogram. (None of the networks were trained using any of the Polytomella data)
{: .subtitle }

### Step 2: segmentation-based picking
Next, we turn the segmentations into picks. After testing the 'hide dust' volume threshold a bit in ChimeraX we find that 250.000 Å³ is a good size threshold. For the spacing we will use 80 Å, which is just over half the distance between adjacent dimers. 
```
easymode pick atp_synthase --data segmented/ --size 250000 --spacing 80 --output coordinates/atp_synthase/
``` 
This yielded 32209 particles.

### Step 3: exporting particles to Relion5
For the subtomogram averaging, whenever it is possible we prefer to work without an external reference. This helps avoid any possibility of 'Einstein from noise'-ing your map ([Henderson, PNAS (2013)](https://www.pnas.org/doi/10.1073/pnas.1314449110)). It is also fun to see a structure appear without having put in any initial map.

We do of course have prior information here. We know that ATP synthase forms a dimer, and that the dimers form rows which shape the mitochondrial cristae. To be able to subbox and use the C2 symmetry later on, it would be helpful if our initial average contained a large enough field of view to feature the full dimer. So we extract relatively large boxes of 128 pixels, with a relatively large pixel size of 7 Å/px.

```
WarpTools ts_export_particles --settings warp_tiltseries.settings --input_directory coordinates/atp_synthase/ --coords_angpix 10.0 --output_angpix 7.0 --box 128 --diameter 600 --output_star relion/atp_synthase/particles.star --relative_output_paths --3d
```

### Step 4: reference-free initial model
We then use an InitialModel job in Relion5 to generate our initial reference for 3D refinement. We used most of the default settings, used no symmetry at this point, and ran the job with K=1 (one class) and --tau2_fudge 1. Initial model and classification jobs in Relion tomo can be very sensitive to the tau2_fudge value. The higher this is set, the more high-frequency information is used early on in processing; we find that in most cases a value around 1.0 (0.5 - 1.5) works well. It can often be useful to run these jobs with various tau2_fudge values. The diameter (400 Å) ended up being a little bit small so you might want to set it to 500.
```
mpirun -n 5 --oversubscribe relion_refine_mpi --o InitialModel/job001/run --iter 200 --grad --denovo_3dref --i particles.star --tomograms --trajectories --ctf --K 1 --sym C1 --flatten_solvent --zero_mask --pool 3 --pad 1 --particle_diameter 400 --oversampling 1 --healpix_order 1 --offset_range 6 --offset_step 2 --auto_sampling --tau2_fudge 1 --j 64 --gpu  --pipeline_control InitialModel/job001/
```
Initial model jobs are very fast in the early iterations, and get slower towards the end when more particles and a larger frequency range are used. Because we are only after an initial model, and we would use a strong low pass filter in the refinement anyway, we stopped this job after iteration 90 when it had already produced a useful ATP synthase map with an estimated resolution of 35 Å.

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1308/1162; background:#fff; border-radius:8px;">
  <source src="../../../assets/atp_synthase_initial_model.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
The initial ATP synthase model with estimated resolution of 35 Å.
{: .subtitle }

### Step 5: refinement in Relion5
Now that we have a reference we can use 3D auto-refine in Relion5. We used two jobs to do this. First, because not all particles have proper poses after the halfway-aborted InitialModel job, we use a global pose search with a spherical mask of 400 Å diameter:
```
mpirun -n 5 --oversubscribe relion_refine_mpi --o Refine3D/job001/run --auto_refine --split_random_halves --i InitialModel/job001/run_it090_data.star --ref InitialModel/job001/run_it090_class001.mrc --firstiter_cc --trust_ref_size --ini_high 60 --pool 3 --pad 2 --ctf --particle_diameter 400 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 4 --offset_range 5 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale --j 14 --gpu  --pipeline_control Refine3D/job001/
```
After iteration 10 this search became very slow (due to increased angular sampling). Because the map has a very distinct structure, with a clear membrane bound complex on a curved bilayer, we reasoned that this latest orientation estimate with its 7.5° search angle would have already oriented the particles correctly. We interrupted the job, and continued with a local search using the newly estimated poses:
```
mpirun -n 5 --oversubscribe relion_refine_mpi --o Refine3D/job002/run --auto_refine --split_random_halves --i Refine3D/job001/run_it010_data.star --ref Refine3D/job001/run_it010_half1_class001.mrc --trust_ref_size --ini_high 30 --pool 3 --pad 2 --ctf --particle_diameter 400 --flatten_solvent --zero_mask --oversampling 1 --healpix_order 2 --auto_local_healpix_order 2 --offset_range 5 --offset_step 2 --sym C1 --low_resol_join_halves 40 --norm --scale --j 14 --gpu  --pipeline_control Refine3D/job002/
```
Notice the difference: we used --healpix_order 2 with --auto_local_healpix_order 4 initially, but this time changed that to --healpix_order 2 and --auto_local_healpix_order 2. This means that the 2nd job starts with local searches only and a search angle of 7.5°. Instead of 2 hours per iteration, this 2nd job ran in just over 2 hours and yielded a map with an estimated resolution of 14 Å, limited by the pixel size of 7 Å/px that we used during extraction (in step 3). (In hindsight, it would have probably been more practical to just run the first job with --auto_local_healpix_order 3)

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1308/1162; background:#fff; border-radius:8px;">
  <source src="../../../assets/atp_synthase_refined_14A.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
The refined map at 14 Å.
{: .subtitle }

### Step 6: increasing the particle number by subboxing
The map now clearly showed a row of ATP synthase dimers. We are not guaranteed that the initial picking step placed a coordinate at each and every ATP synthase monomer in every tomogram. So to add missing particles to the particle set, we can now use _subboxing_. This works as follows: if we define multiple per-monomer transformations in local space (particle space) and then, for every particle in our particle set, make one copy of that particle per monomer and apply every per-monomer transformation, we end with a larger particle set that covers more unique monomers in real space (the tomogram volume). This may be easier to explain visually - the below video shows the process of using the [ChimeraX-subboxer](https://github.com/mgflast/ChimeraX-subbox) to subbox the ATP synthase particle set.

<div style="text-align: center;">
<video muted playsinline controls style="width:100%; max-width:960px; aspect-ratio:16/10; background:#fff; border-radius:8px;">
  <source src="../../../assets/atp_synthase_subboxing.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
Subboxing the ATP synthase particle set with the ChimeraX-subboxer plugin.
{: .subtitle }

In this video we do the following steps: first, we crop a single ATP synthase monomer from the parent map. Second, we make copies of this monomer and place them correctly in the parent map. We then use the ChimeraX-subboxer plugin to measure the relative position and orientation of each monomer, to load the starfile from the latest Relion5 refinement job, and to add the new particles to that starfile.

The output from subboxing is thus a new starfile, with more particles than before. Because we were doing validation of the automated picking with the easymode atp_synthase network, we only used the C2 symmetry here and sub-boxed the 2nd monomer alone, not any of the particles in the adjacent dimers (if the network had completely missed some dimers, we did not want to artificially add those in - outside of validation, we would of course have wanted to include as many particles as possible). Because we discarded duplicates (particles within 50 Å of each other) and many dimers had already had both monomers picked, the set only increased from 32209 to 39100 particles. Outside of validation, we might have also wanted to use 3D classification to discard any possible false picks (especially if you subbox particles from adjacent dimers this would be useful), but again we did not use it here.

### Step 7: refinement in M
We now had our initial poses, a 14 Å resolution map, and 39k particles. To improve the resolution we use M. This requires a mask - we made one quickly by opening the map in ChimeraX, setting an appropriate threshold, and using `volume onesmask #1 onGrid #1 valueType float32`, then `vop gaussian #2 sDev 5.0`, then adjusting the threshold on the resulting model so that it is slightly larger than the original map, and finally running `volume onesmask #3 onGrid #3 valueType float32` and saving the map resulting from all of that as `mask.mrc`.
```
MTools create_population -d m -n easymode
MTools create_source -p m/easymode.population -n easymode -s warp_tiltseries.settings
MTools create_species -p m/easymode.population -n atp_synthase -d 250 -s C1 --half1 relion/atp_synthase/Refine3D/job002/run_half1_class001_unfil.mrc --half2 relion/atp_synthase/Refine3D/job002/run_half2_class001_unfil.mrc --mask mask.mrc --particles_relion relion/atp_synthase/Refine3D/job002/run_data_subboxed.star --angpix 6.0
MCore --population m/easymode.population --iter 0
```

After the import, the estimated resolution of the map in M was 12.3 Å. After repeated rounds of pose refinement (--refine_particles), refining the image warp (--refine_imagewarp) with increasing level of detail (1x1, then 2x2, then at most 4x4), and lowering the pixel size incrementally (we prefer to start coarse, so that refinement is faster, and decrease the pixel size only when resolution appears to be limited by sampling rather than by other factors), and one round of (exhaustive) CTF refinement (--ctf_defocus --ctf_defocusexhaustive), we ended up with a map at 7.8 Å global resolution. At this point you could use focused refinement and classification to try to improve the resolution of parts of the map, but we didn't do that.

<div style="text-align: center;">
<video autoplay loop muted playsinline controls style="width:100%; max-width:720px; aspect-ratio:1200/1162; background:#fff; border-radius:8px;">
  <source src="../../../assets/atp_synthase_final_map.mp4" type="video/mp4">
  Video failed to load.
</video>
</div>
The final map at 7.8 Å global resolution.
{: .subtitle }