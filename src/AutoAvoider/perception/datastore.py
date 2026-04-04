"""Dataset utilities for training (Tub/TubGroup)."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd

from AutoAvoider.common.logging import setup_logging
from AutoAvoider.tools.path_utils import expand_path_arg

logger = setup_logging(name="autoavoider.perception.datastore")


class Tub:
    """Handle dataset directory and records."""

    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(path)
        logger.info("path_in_tub: %s", self.path)
        self.meta_path = os.path.join(self.path, "meta.json")
        self.df: pd.DataFrame | None = None

        if os.path.exists(self.path):
            logger.info("Tub exists: %s", self.path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            self.current_ix = self.get_last_ix() + 1
        else:
            self.meta = {"inputs": [], "types": []}
            self.current_ix = 0

        self.start_time = time.time()

    def get_last_ix(self) -> int:
        index = self.get_index()
        return max(index) if len(index) >= 1 else -1

    def update_df(self) -> None:
        df = pd.DataFrame([self.get_json_record(i) for i in self.get_index(shuffled=False)])
        self.df = df

    def get_df(self) -> pd.DataFrame:
        if self.df is None:
            self.update_df()
        return self.df

    def get_index(self, shuffled: bool = True) -> List[int]:
        files = next(os.walk(self.path))[2]
        record_files = [f for f in files if f[:6] == "record"]

        def get_file_ix(file_name: str) -> int:
            try:
                name = file_name.split(".")[0]
                num = int(name.split("_")[1])
            except Exception:
                num = 0
            return num

        nums = [get_file_ix(f) for f in record_files]
        nums = sorted(nums) if not shuffled else random.sample(nums, len(nums))
        return nums

    @property
    def inputs(self) -> List[str]:
        return list(self.meta["inputs"])

    @property
    def types(self) -> List[str]:
        return list(self.meta["types"])

    def get_input_type(self, key: str) -> str | None:
        input_types = dict(zip(self.inputs, self.types))
        return input_types.get(key)

    def get_num_records(self) -> int:
        import glob
        files = glob.glob(os.path.join(self.path, "record_*.json"))
        return len(files)

    def make_record_paths_absolute(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in record_dict.items():
            if isinstance(v, str) and "." in v:
                v = os.path.join(self.path, v)
            d[k] = v
        return d

    def check(self, fix: bool = False) -> None:
        logger.info("Checking tub: %s", self.path)
        logger.info("Found: %s records", self.get_num_records())
        problems = False
        for ix in self.get_index(shuffled=False):
            try:
                self.get_record(ix)
            except Exception:
                problems = True
                if not fix:
                    logger.warning("problems with record %s : %s", ix, self.path)
                else:
                    logger.warning("problems with record %s, removing: %s", ix, self.path)
                    self.remove_record(ix)
        if not problems:
            logger.info("No problems found.")

    def remove_record(self, ix: int) -> None:
        record = self.get_json_record_path(ix)
        try:
            os.remove(record)
        except FileNotFoundError:
            return

    def get_json_record_path(self, ix: int) -> str:
        return os.path.join(self.path, f"record_{ix}.json")

    def get_json_record(self, ix: int) -> Dict[str, Any]:
        path = self.get_json_record_path(ix)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                json_data = json.load(fp)
        except UnicodeDecodeError as exc:
            raise Exception(
                f"bad record: {ix}. You may want to run a data check/fix step"
            ) from exc
        except FileNotFoundError:
            raise
        except Exception:
            logger.error("Unexpected error: %s", sys.exc_info()[0])
            raise

        record_dict = self.make_record_paths_absolute(json_data)
        return record_dict

    def get_record(self, ix: int) -> Dict[str, Any]:
        json_data = self.get_json_record(ix)
        data = self.read_record(json_data)
        return data

    def read_record(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key, val in record_dict.items():
            typ = self.get_input_type(key)
            if typ == "image_array":
                img = cv2.imread(val)
                val = np.array(img)
            data[key] = val
        return data

    def delete(self) -> None:
        import shutil
        shutil.rmtree(self.path)

    def get_record_gen(
        self,
        record_transform: Any = None,
        shuffle: bool = True,
        df: pd.DataFrame | None = None,
    ) -> Iterable[Dict[str, Any]]:
        if df is None:
            df = self.get_df()

        while True:
            for _ in self.df.iterrows():
                if shuffle:
                    record_dict = df.sample(n=1).to_dict(orient="records")[0]
                else:
                    record_dict = df.sample(n=1).to_dict(orient="records")[0]

                record_dict = self.read_record(record_dict)
                if record_transform:
                    record_dict = record_transform(record_dict)
                yield record_dict

    def get_batch_gen(
        self,
        keys: List[str] | None = None,
        batch_size: int = 128,
        record_transform: Any = None,
        shuffle: bool = True,
        df: pd.DataFrame | None = None,
    ) -> Iterable[Dict[str, np.ndarray]]:
        record_gen = self.get_record_gen(record_transform=record_transform, shuffle=shuffle, df=df)

        if df is None:
            df = self.get_df()
        if keys is None:
            keys = list(self.df.columns)

        while True:
            record_list = [next(record_gen) for _ in range(batch_size)]
            batch_arrays: Dict[str, np.ndarray] = {}
            for k in keys:
                arr = np.array([r[k] for r in record_list])
                batch_arrays[k] = arr
            yield batch_arrays

    def get_train_gen(
        self,
        X_keys: List[str],
        Y_keys: List[str],
        batch_size: int = 128,
        record_transform: Any = None,
        df: pd.DataFrame | None = None,
    ) -> Iterable[Tuple[List[np.ndarray], List[np.ndarray]]]:
        batch_gen = self.get_batch_gen(
            X_keys + Y_keys,
            batch_size=batch_size,
            record_transform=record_transform,
            df=df,
        )

        while True:
            batch = next(batch_gen)
            X = [batch[k] for k in X_keys]
            Y = [batch[k] for k in Y_keys]
            yield X, Y

    def get_train_val_gen(
        self,
        X_keys: List[str],
        Y_keys: List[str],
        batch_size: int = 128,
        train_frac: float = 0.8,
        train_record_transform: Any = None,
        val_record_transform: Any = None,
    ) -> Tuple[Iterable[Tuple[List[np.ndarray], List[np.ndarray]]], Iterable[Tuple[List[np.ndarray], List[np.ndarray]]]]:
        if self.df is None:
            self.update_df()

        train_df = self.df.sample(frac=train_frac, random_state=200)
        val_df = self.df.drop(train_df.index)

        train_gen = self.get_train_gen(
            X_keys=X_keys,
            Y_keys=Y_keys,
            batch_size=batch_size,
            record_transform=train_record_transform,
            df=train_df,
        )
        val_gen = self.get_train_gen(
            X_keys=X_keys,
            Y_keys=Y_keys,
            batch_size=batch_size,
            record_transform=val_record_transform,
            df=val_df,
        )

        return train_gen, val_gen


class TubGroup(Tub):
    """Combine multiple tubs."""

    def __init__(self, tub_paths_arg: str) -> None:
        tub_paths = expand_path_arg(tub_paths_arg)
        logger.info("TubGroup:tubpaths: %s", tub_paths)
        self.tubs = [Tub(path) for path in tub_paths]
        self.input_types: Dict[str, str] = {}

        record_count = 0
        for t in self.tubs:
            t.update_df()
            record_count += len(t.df)
            self.input_types.update(dict(zip(t.inputs, t.types)))

        logger.info(
            "joining the tubs %s records together. This could take %s minutes.",
            record_count,
            int(record_count / 300000),
        )

        self.meta = {"inputs": list(self.input_types.keys()), "types": list(self.input_types.values())}
        self.df = pd.concat([t.df for t in self.tubs], axis=0, join="inner")

    @property
    def inputs(self) -> List[str]:
        return list(self.meta["inputs"])

    @property
    def types(self) -> List[str]:
        return list(self.meta["types"])

    def get_num_tubs(self) -> int:
        return len(self.tubs)

    def get_num_records(self) -> int:
        return len(self.df)
