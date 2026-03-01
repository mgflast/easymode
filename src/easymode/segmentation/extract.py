import os

TRAINING_COLLECTION_APIX = 10.0


def extract_training_data(features, apix):
    import glob, mrcfile
    import pickle
    import numpy as np
    import json
    import hashlib
    from multiprocessing import Pool

    binning_xy = int(round(apix / TRAINING_COLLECTION_APIX))
    if binning_xy < 1:
        binning_xy = 1

    BOX_Z = 160
    BOX_XY = 160 * binning_xy

    flavours = ['even', 'odd', 'raw', 'cryocare', 'ddw']
    annotated_tomograms = glob.glob('/cephfs/mlast/compu_projects/easymode/volumes_cryocare/*.scns')

    with open('/cephfs/mlast/compu_projects/easymode/datasets/dataset_contents.json', 'r') as jf:
        dataset_tomo_map = json.load(jf)

    feature_dataset_count_map = dict()
    for f in features:
        feature_dataset_count_map[f] = dict()
        for dataset in dataset_tomo_map:
            feature_dataset_count_map[f][dataset] = 0
        os.makedirs(f'/cephfs/mlast/compu_projects/easymode/training/3d/data/{f}/label', exist_ok=True)
        os.makedirs(f'/cephfs/mlast/compu_projects/easymode/training/3d/data/{f}/validity', exist_ok=True)
        for flavour in flavours:
            os.makedirs(f'/cephfs/mlast/compu_projects/easymode/training/3d/data/{f}/{flavour}', exist_ok=True)

    for f in features:
        for subdir in ['label', 'validity'] + flavours:
            d = f'/cephfs/mlast/compu_projects/easymode/training/3d/data/{f}/{subdir}'
            for old_file in glob.glob(os.path.join(d, '*')):
                os.remove(old_file)

    tomo_dataset_map = dict()
    for dataset in dataset_tomo_map:
        for tomo in dataset_tomo_map[dataset]:
            tomo_dataset_map[tomo] = dataset

    def make_id(tomo, j, k, l, feature):
        key = f"{tomo}_{feature}_{j}_{k}_{l}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def get_box(j, k, l, tomo, flavour):
        dataset = tomo_dataset_map[tomo]
        if flavour == 'even':
            path = f'/cephfs/mlast/compu_projects/easymode/datasets/{dataset}/warp_tiltseries/reconstruction/even/{tomo}'
        elif flavour == 'odd':
            path = f'/cephfs/mlast/compu_projects/easymode/datasets/{dataset}/warp_tiltseries/reconstruction/odd/{tomo}'
        elif flavour == 'raw':
            path = f'/cephfs/mlast/compu_projects/easymode/datasets/{dataset}/warp_tiltseries/reconstruction/{tomo}'
        elif flavour == 'cryocare':
            path = f'/cephfs/mlast/compu_projects/easymode/volumes_cryocare/{tomo}'
        else:
            path = f'/cephfs/mlast/compu_projects/easymode/volumes_ddw/{tomo}'

        if not os.path.exists(path):
            return False, None, None

        hz, hxy = BOX_Z // 2, BOX_XY // 2
        with mrcfile.mmap(path, permissive=True) as m:
            v = m.data
            Z, Y, X = v.shape
            out = np.zeros((BOX_Z, BOX_XY, BOX_XY), dtype=v.dtype)
            msk = np.zeros((BOX_Z, BOX_XY, BOX_XY), dtype=np.uint8)

            z0, z1 = j - hz, j + (BOX_Z - hz)
            y0, y1 = k - hxy, k + (BOX_XY - hxy)
            x0, x1 = l - hxy, l + (BOX_XY - hxy)

            zs0, zs1 = max(0, z0), min(Z, z1)
            ys0, ys1 = max(0, y0), min(Y, y1)
            xs0, xs1 = max(0, x0), min(X, x1)

            zd0, yd0, xd0 = zs0 - z0, ys0 - y0, xs0 - x0
            dz, dy, dx = zs1 - zs0, ys1 - ys0, xs1 - xs0

            if dz > 0 and dy > 0 and dx > 0:
                out[zd0:zd0+dz, yd0:yd0+dy, xd0:xd0+dx] = v[zs0:zs1, ys0:ys1, xs0:xs1]
                msk[zd0:zd0+dz, yd0:yd0+dy, xd0:xd0+dx] = 1

            # Bin XY only, leave Z unbinned
            if binning_xy > 1:
                out = out.reshape((BOX_Z, 160, binning_xy, 160, binning_xy)).mean(axis=(2, 4))
                msk = msk.reshape((BOX_Z, 160, binning_xy, 160, binning_xy)).min(axis=(2, 4))

        return True, out, msk

    def get_box_save_box(j, k, l, tomo, flavour, sample_id, feature):
        if flavour == 'cryocare':
            with open(f'/cephfs/mlast/compu_projects/easymode/training/3d/data/{feature}/cryocare/{sample_id}.aislink', 'w') as lf:
                lf.write(f'{tomo}')
        valid, vol, msk = get_box(j, k, l, tomo, flavour)
        if not valid:
            return
        with mrcfile.new(f'/cephfs/mlast/compu_projects/easymode/training/3d/data/{feature}/{flavour}/{sample_id}.mrc', overwrite=True) as m:
            m.set_data(vol.astype(np.float32))
            m.voxel_size = (apix, apix, TRAINING_COLLECTION_APIX)
        if flavour == 'cryocare':
            with mrcfile.new(f'/cephfs/mlast/compu_projects/easymode/training/3d/data/{feature}/validity/{sample_id}.mrc', overwrite=True) as m:
                m.set_data(msk.astype(np.uint8))
                m.voxel_size = (apix, apix, TRAINING_COLLECTION_APIX)

    def process_particle(args):
        j, k, l, tomo, feature, particle_id = args
        get_box_save_box(j, l, k, tomo, 'raw', particle_id, feature)
        get_box_save_box(j, l, k, tomo, 'even', particle_id, feature)
        get_box_save_box(j, l, k, tomo, 'odd', particle_id, feature)
        get_box_save_box(j, l, k, tomo, 'cryocare', particle_id, feature)
        if 'Junk' in feature or 'Not' in feature:
            with mrcfile.new(f'/cephfs/mlast/compu_projects/easymode/training/3d/data/{feature}/label/{particle_id}.mrc', overwrite=True) as m:
                m.set_data(np.zeros((160, 160, 160), dtype=np.float32))
                m.voxel_size = (apix, apix, TRAINING_COLLECTION_APIX)
        return feature, tomo_dataset_map[tomo]

    tasks = []
    print(f"Scanning {len(annotated_tomograms)} annotated tomograms...")
    print(f"Target pixel size: {apix} A/px (XY binning: {binning_xy}x, Z unbinned at {TRAINING_COLLECTION_APIX} A/px)")
    print(f"Extracting boxes: Z={BOX_Z}px, XY={BOX_XY}px -> binned to 160x160x160")
    for n, annotation in enumerate(annotated_tomograms, start=1):
        try:
            with open(annotation, 'rb') as pf:
                se_frame = pickle.load(pf)
        except Exception as e:
            print(f"\terror loading {annotation}\n\t{e}")
            continue

        tomo = os.path.basename(se_frame.path).split('\\')[-1]
        if tomo not in tomo_dataset_map:
            print(f"\tskipping {annotation}: tomogram {tomo} not in dataset map")
            continue
        dataset = tomo_dataset_map[tomo]
        for f in se_frame.features:
            if f.title not in features:
                continue
            print(f'\tparsing {n}/{len(annotated_tomograms)} - {dataset} - {annotation} - {f.title}')
            box_coordinates = [(z, box[0], box[1]) for z in f.boxes for box in f.boxes[z]]
            for j, k, l in box_coordinates:
                particle_id = make_id(tomo, j, k, l, f.title)
                tasks.append((j, k, l, tomo, f.title, particle_id))

    print(f"Processing {len(tasks)} particles across {os.cpu_count()} cores")
    with Pool() as pool:
        results = pool.map(process_particle, tasks)

    for feature, dataset in results:
        feature_dataset_count_map[feature][dataset] += 1

    with open('/cephfs/mlast/compu_projects/easymode/training/3d/n_annotations.json', 'w') as jf:
        json.dump(feature_dataset_count_map, jf, indent=4)

    print("Done.")
