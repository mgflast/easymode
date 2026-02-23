import os, glob, time, multiprocessing, psutil
import tensorflow as tf
import starfile
import mrcfile
import numpy as np
import shutil
from easymode.core.distribution import get_model, load_model
from easymode.core.util import fourier_bin
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

class TiltSelectionJob:
    TRAINED_AT_BIN = 16

    def __init__(self, tiltstack, tomostar_file=None, output_directory=None, xml_directory=None):
        self.tiltstack = tiltstack
        self.tomostar_file = tomostar_file
        self.output_directory = output_directory
        self.tiltstack_file = None
        self.xml_directory = xml_directory
        self.xml_file = None
        self.output_file = None
        self.init_time = time.time()
        self.parse_inputs()
        self.started = False
        self.report = False

    def parse_inputs(self):
        # Find tiltstack file
        if os.path.isfile(self.tiltstack):
            self.tiltstack_file = self.tiltstack
            tomo = os.path.splitext(os.path.basename(self.tiltstack_file))[0]
        else:
            tomo = os.path.splitext(os.path.basename(self.tomostar_file))[0]
            candidates = [
                os.path.join(self.tiltstack, f'{tomo}.st'),
                os.path.join(self.tiltstack, f'{tomo}.mrc'),
                os.path.join(self.tiltstack, tomo, '*.st'),     # not doing *.mrc because that's often something like {tomo}*_CTF.mrc
                os.path.join('warp_tiltseries', 'tiltstack', tomo, f'{tomo}.st')
            ]

            for c in candidates:
                if "*" in c:
                    if hits :=glob.glob(c):
                        self.tiltstack_file = hits[0]
                else:
                    if os.path.isfile(c):
                        self.tiltstack_file = c

        # Find xml file
        if self.xml_directory is not None:
            self.xml_file = os.path.join(self.xml_directory, f'{tomo}.xml')
            if not os.path.exists(self.xml_file):
                self.xml_file = None

        # Determine output file path
        if self.output_directory is None:
            if self.tomostar_file is None:
                self.output_directory = os.path.dirname(self.tiltstack_file)
            else:
                self.output_directory = os.path.dirname(self.tomostar_file)

        os.makedirs(self.output_directory, exist_ok=True)
        self.output_file = os.path.join(self.output_directory, f'{tomo}.tomostar')
        print(self.output_file)

    def get_reference_index(self, stack):
        if self.tomostar_file is not None:
            df = starfile.read(self.tomostar_file)
            return int(df["wrpDose"].astype(float).idxmin())
        else:
            return stack.shape[0] // 2

    @staticmethod
    def preprocess(image):
        image = fourier_bin(image, 16)
        image -= np.mean(image)
        image /= np.std(image) + 1e-8
        return image[None, ..., None]

    @staticmethod
    def predict_tta(x0, x1, model, tta):
        TTA_ROTATE = [0, 1, 2, 3, 0, 1, 2, 3]
        TTA_FLIP = [0, 0, 0, 0, 1, 1, 1, 1]
        tta = min([8, tta])

        y = 0
        for t in range(tta):
            _x0 = np.rot90(x0, k=TTA_ROTATE[t], axes=(1, 2))
            _x1 = np.rot90(x1, k=TTA_ROTATE[t], axes=(1, 2))
            if TTA_FLIP[t]:
                _x0 = np.flip(_x0, axis=1)
                _x1 = np.flip(_x1, axis=1)

            instance_y = np.squeeze(model.predict([_x0, _x1]))
            y += instance_y

        return y / tta

    def set_xml_flags(self, df, xml_file):
        from lxml import etree
        import os

        parser = etree.XMLParser(remove_blank_text=False)
        tree = etree.parse(xml_file, parser)
        root = tree.getroot()

        movie_el = root.find('.//{*}MoviePath')
        use_el = root.find('.//{*}UseTilt')
        if movie_el is None or use_el is None:
            return

        xml_movies = movie_el.text.split()
        use_text = use_el.text
        xml_use = [line.strip() for line in use_text.split('\n') if line.strip()]

        if len(xml_movies) != len(xml_use):
            return

        if 'wrpMovieName' not in df.columns or 'tiltInclude' not in df.columns:
            return

        inc_by_base = {os.path.basename(p): bool(int(v))
                       for p, v in zip(df['wrpMovieName'].astype(str), df['tiltInclude'].astype(int))}

        missing = 0
        n_false = 0
        for i, p in enumerate(xml_movies):
            b = os.path.basename(p)
            if b in inc_by_base:
                v = inc_by_base[b]
                xml_use[i] = "True" if v else "False"
                if not v:
                    n_false += 1
            else:
                missing += 1

        use_el.text = '\n'.join(xml_use)
        tree.write(xml_file, encoding="utf-8", xml_declaration=True, pretty_print=False)

    def process(self, model, tta, overwrite, threshold=0.5):
        if os.path.exists(self.output_file):
            if os.path.getmtime(self.output_file) > self.init_time:  # output exists, written after process started - another thread did it
                return None
            if not overwrite:  # output exists, and we don't want to overwrite
                return f'skipped (output already exists at {self.output_file})' if self.report else None

        with open(self.output_file, 'w') as f:
            pass # placeholder file made to signal this stack is being processed.

        stack = mrcfile.read(self.tiltstack_file)
        proc_stack = list()
        reference_index = self.get_reference_index(stack)

        if self.tomostar_file is not None:
            df = starfile.read(self.tomostar_file)  # we're assuming there is one block in the starfile.
            df = df.reset_index(drop=True)
            if len(df) != stack.shape[0]:
                raise Exception(f"Number of rows in {self.tomostar_file} ({len(df)}) does not match then number of images in {self.tiltstack_file} ({stack.shape[0]}) - skipping.")
        else:
            import pandas as pd
            df = pd.DataFrame({"tiltIndex": range(stack.shape[0])})

        df["tiltInclude"] = 1
        df["tiltScore"] = "n/a"

        reference_tilt = TiltSelectionJob.preprocess(stack[reference_index, :, :])
        for j in range(stack.shape[0]):
            if j == reference_index:
                df.loc[j, 'tiltInclude'] = 1
                df.loc[j, 'tiltScore'] = "reference"
                proc_stack.append(reference_tilt)
                continue

            query_tilt = TiltSelectionJob.preprocess(stack[j, :, :])
            proc_stack.append(query_tilt)
            y = TiltSelectionJob.predict_tta(reference_tilt, query_tilt, model, tta)
            df.loc[j, 'tiltInclude'] = int(y > threshold)
            df.loc[j, 'tiltScore'] = f"{y:.4f}"

        # Write final .tomostar files.
        starfile.write(df, self.output_file)

        # Apply result to XML file if available
        if self.xml_file is not None:
            self.set_xml_flags(df, self.xml_file)

        # Save stack (for debugging)
        # proc_stack = np.squeeze(np.array(proc_stack, dtype=np.float32))
        # mrcfile.new(self.output_file.replace('.tomostar', '.mrc'), overwrite=overwrite, data=proc_stack)

        return f"{df['tiltInclude'].sum()} tilts discarded"

def inference_thread(job_list, model_path, gpu, tta, overwrite, threshold=0.5, is_reporter_thread=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)

    process_start_time = psutil.Process().create_time()

    print(f'GPU {gpu} - loading model ({model_path}).')
    model = load_model(model_path)

    for j, job in enumerate(job_list, 1):
        job.report = is_reporter_thread
        job_out = job.process(model, tta, overwrite, threshold=threshold)
        etc = time.strftime('%H:%M:%S', time.gmtime((time.time() - process_start_time) / j * (len(job_list) - j)))
        if job_out is not None:
            print(f"{j}/{len(job_list)} (on GPU {gpu}) - {job.output_file} - etc {etc}" + f" - {job_out}")


def parse_inputs(input_tiltstack, input_tomostar, input_xml, output_directory, extension='*.tomostar'):
    if not "." in extension:
        extension = f".{extension}"
    if not "*" in extension:
        extension = f"*{extension}"

    jobs = list()


    if not os.path.isdir(input_tiltstack):
        if os.path.isfile(input_tiltstack):                                                                             # A: single tilt stack
            job = TiltSelectionJob(tiltstack=input_tiltstack, output_directory=output_directory, xml_directory=input_xml)
            jobs.append(job)
        else:                                                                                                           # B: tiltstack glob pattern
            for f in glob.glob(input_tiltstack):
                job = TiltSelectionJob(tiltstack=f, output_directory=output_directory, xml_directory=input_xml)
                jobs.append(job)
    elif os.path.isdir(input_tiltstack) and input_tomostar is None:                                                     # C: tiltstack directory, no tomostar
        for f in glob.glob(os.path.join(input_tiltstack, '*.st')):
            job = TiltSelectionJob(tiltstack=f, output_directory=output_directory, xml_directory=input_xml)
            jobs.append(job)
    else:
        if os.path.isfile(input_tomostar):                                                                              # D: tiltstack directory, single tomostar file
            job = TiltSelectionJob(tiltstack=input_tiltstack, tomostar_file=input_tomostar, output_directory=output_directory, xml_directory=input_xml)
            jobs.append(job)
        elif os.path.isdir(input_tomostar):                                                                             # E: tiltstack directory, tomostar directory
            for f in glob.glob(os.path.join(input_tomostar, extension)):
                job = TiltSelectionJob(tiltstack=input_tiltstack, tomostar_file=f, output_directory=output_directory, xml_directory=input_xml)
                jobs.append(job)

            # Create a backup of the original tomostar files.
            if os.path.abspath(output_directory) == os.path.abspath(input_tomostar):
                timestamp = datetime.now().strftime("%Y%m%d%H%M")
                backup_directory = os.path.join(input_tomostar, f'backup_{timestamp}')
                os.makedirs(backup_directory, exist_ok=True)
                for j in jobs:
                    shutil.copy(j.tomostar_file, os.path.join(backup_directory, os.path.basename(j.tomostar_file)))
                    j.tomostar_file = os.path.join(backup_directory, os.path.basename(j.tomostar_file))
                print(f'A backup of the original .tomostar files in {input_tomostar} has been saved at {backup_directory}')

    return jobs


def dispatch(input_tiltstack, input_tomostar, input_xml, output_directory, tta=1, gpus=None, overwrite=False, extension="*.tomostar", threshold=0.5):
    if gpus is None:
        gpus = list(range(0, len(tf.config.list_physical_devices('GPU'))))
    else:
        gpus = [int(g) for g in gpus.split(',') if g.strip().isdigit()]

    if len(gpus) == 0:
        print("\033[93m" + "warning: no GPUs detected. processing will continue, but using CPUs only!" + "\033[0m")

    if input_tiltstack is None and input_tomostar is None:
        print(f'Please specify a value for either --tiltstack or --tomostar (or both). For example: --tomostar tomostar --tiltstack warp_tiltseries/tiltstack')
    if output_directory is None and input_tomostar is not None and os.path.isdir(input_tomostar):
        output_directory = input_tomostar

    jobs = parse_inputs(input_tiltstack, input_tomostar, input_xml, output_directory, extension)
    if len(jobs) == 0:
        print(f'No tomostars and/or tilt stacks found with --tomostar {input_tomostar} and --tiltstack {input_tiltstack}. Exiting.')
        return

    print(f'Found {len(jobs)} tilt stacks to process.')

    model_path, metadata = get_model('tilt')
    if model_path is None:
        print(f'Could not find model for tilt selection. Exiting.')
        return

    # back up original xmls
    if input_xml is not None and os.path.exists(input_xml):
        backup_location = os.path.join(input_xml, 'backup_' + datetime.now().strftime("%Y%m%d%H%M"))
        print(f'backing up original xml files to {backup_location}')
        os.makedirs(backup_location, exist_ok=True)
        for j in jobs:
            if os.path.exists(j.xml_file):
                shutil.copy(j.xml_file, os.path.join(backup_location, os.path.basename(j.xml_file)))


    multiprocessing.set_start_method("spawn", force=True)

    processes = []
    for gpu in gpus:
        p = multiprocessing.Process(
            target=inference_thread,
            args=(
                jobs,
                model_path,
                gpu,
                tta,
                overwrite,
                threshold,
                (gpu == gpus[0])
            )
        )
        processes.append(p)
        p.start()
        time.sleep(2)

    for p in processes:
        p.join()


