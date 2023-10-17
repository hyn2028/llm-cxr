from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from .mimiccxr_vq_dataset import CXR_VQ_TOKENIZER_LEN, CXR_VQ_VQ_LEN, CXR_VQ_CODE_BOOK_SIZE, bcolors

from pathlib import Path
import _pickle as cPickle


class MimicCxrVqaDataset(Dataset): 
    def _load_vqa_datset(self, path: Path) -> dict:
        def _vqa_file_path_to_pid_sid(path: Path) -> tuple:
            pid = path.parent.name[1:]
            sid = path.name.split(".")[0][1:]
            return int(pid), int(sid)
        
        def _check_vqa(vqa: dict) -> bool:
            try:
                if len(vqa) != 4:
                    raise ValueError(vqa)
                vqa["answer"]
                vqa["question"]
            except:
                return False
            return True

        vqa_files = path.glob("**/*.txt")
        vqas_dump = []
        for vqa_file in vqa_files:
            with open(vqa_file, "r") as f:
                vqas = eval(f.readline())
                assert f.readline() == ""
            for vqa in vqas:
                vqa["subject_id"], vqa["study_id"] = _vqa_file_path_to_pid_sid(vqa_file)
            vqas_dump.extend(vqas)
        
        print("MimicCxrVqaDataset: # of vqas: ", len(vqas_dump))
        vqas_dump = list(filter(_check_vqa, vqas_dump))
        print("MimicCxrVqaDataset: # of vqas after ill-formed filtering: ", len(vqas_dump))
        return pd.DataFrame(vqas_dump)
        

    def __init__(self, report_path: str, vq_path: str, tokenizer_len: int, mode: str):
        assert tokenizer_len == CXR_VQ_TOKENIZER_LEN
        
        report_path = Path(report_path)

        # Load dataset
        self.mimic_db = pd.read_csv(report_path / "mimic-cxr-2.0.0-split.csv", 
                              index_col="dicom_id", 
                              dtype={"subject_id": int, "study_id": int, "split": str})
        self.vqa_db = self._load_vqa_datset(report_path / "mimic-cxr-jpg_medvqa_v1")

        # Select dataset
        if mode == "train":
            self.mimic_db = self.mimic_db.loc[self.mimic_db['split'] == 'train']
        elif mode == "test":
            self.mimic_db = self.mimic_db.loc[(self.mimic_db['split'] == 'test') | (self.mimic_db['split'] == 'validate')]
        else:
            raise ValueError(mode)

        # Load vector quantized images
        with open(vq_path, "rb") as f:
            self.cxr_vq = cPickle.load(f)
        error_dicom_ids = self._check_vq()
        dicom_ids = set(self.mimic_db.index) - error_dicom_ids

        # Load selected dicom ids (pa/ap & earliest study)
        with open(report_path / "mimic-cxr-2.0.0-selected-pa-ap-earlist-study.pickle", "rb") as f:
            selected_dicom_ids = set(cPickle.load(f).keys())

        # Filter out dicom ids
        possible_dicom_ids = dicom_ids & selected_dicom_ids
        self.mimic_db = self.mimic_db.loc[list(possible_dicom_ids)].reset_index()

        # Merge mimic_db and vqa_db
        self.final_db = self.mimic_db.merge(self.vqa_db, how="right", on=["subject_id", "study_id"])
        self.final_db = self.final_db.dropna()

        # Generate outputs
        self.outputs = [] 
        for index, row in tqdm(self.final_db.iterrows(), colour="green", unit='pair', total=len(self.final_db)):
            io_type = "input"
            dicom_id = row["dicom_id"]
            cxr_vq = self.cxr_vq[dicom_id]
            cxr_vq_shifted = [x + tokenizer_len for x in cxr_vq]
            report = row["answer"]
            instruction = row["question"]
            self.outputs.append({"report": report,
                                 "cxr_vq_shifted": cxr_vq_shifted,
                                 "io_type": io_type,
                                 "instruction": instruction})
        
        del self.mimic_db
        del self.vqa_db
        del self.final_db
        del self.cxr_vq


    def __len__(self) -> int: 
        return len(self.outputs)


    def __getitem__(self, idx: int): 
        return self.outputs[idx]
    

    def _check_vq(self):
        error_ids = set()
        for dicom_id, cxr_vq in self.cxr_vq.items():
            if len(cxr_vq) != CXR_VQ_VQ_LEN or max(cxr_vq) >= CXR_VQ_CODE_BOOK_SIZE:
                error_ids.add(dicom_id)
        print(f"{bcolors.FAIL}[Warning] # of error dicom vq(s): {len(error_ids)}{bcolors.ENDC}")
        return error_ids
